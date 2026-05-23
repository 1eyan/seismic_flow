#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GatedSeisDiT — 基于 Gated Transformer 的地震数据生成模型（Diffusion / Flow Matching）

架构以 gated_transformer_v5.py 为主:
  - Encoder: SimpleGatedEncoder (stacked GatedTransformerEncoderBlock + 6D RoPE)
  - Bottleneck: GatedDiTBlock (adaLN-Zero, 6D RoPE, flatten 全局注意力)
  - Decoder: SimpleGatedDecoder (stacked GatedTransformerEncoderBlock + 6D RoPE)
  - Tokenizer: Conv2d — 处理 chunk 后的 4D I/O 边界

组件全部复用自 gated_transformer_v5.py:
  - RMSNorm, Qwen3MLP, GatedMultiHeadAttention
  - GatedTransformerEncoderBlock, SimpleGatedEncoder, SimpleGatedDecoder
  - GatedDiTBlock, RotaryPositionEmbeddingND, get_norm_layer
  - trace_time_chunk, trace_time_unchunk

Forward 签名与训练代码 train_fpmV3_ddp.py 兼容:
  forward(self, x, t, condL=None, log_tau=None, time_axis=None, training=False)
  输入: x (B, 2, H, W) -- x[:,0:1]=data, x[:,1:2]=condition
  输出: (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .gated_transformer_v5 import (
    RMSNorm,
    SimpleGatedEncoder,
    SimpleGatedDecoder,
    GatedDiTBlock,
    get_norm_layer,
    trace_time_chunk,
    trace_time_unchunk,
)


# =============================================================================
# Utility
# =============================================================================


def modulate(x, shift, scale):
    """adaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class GenTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding -> MLP."""
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


# =============================================================================
# GatedSeisDiT — Full generation model
# =============================================================================


class GatedSeisDiT(nn.Module):
    """
    Generation model using gated Transformer framework.
    Encoder/Decoder: SimpleGatedEncoder/Decoder (stacked GatedTransformerEncoderBlock)
    Bottleneck: GatedDiTBlock with adaLN-Zero conditioning
    RoPE: 6D coordinate-based (RotaryPositionEmbeddingND)

    Pipeline:
      1. trace_time_chunk: (B,H,W) -> (B, H*n_chunks, chunk_length)
      2. Conv2d fuse: merge data+condition chunks
      3. Linear embed: chunk_length -> d_model
      4. Encoder -> Bottleneck -> Decoder
      5. Head: d_model -> 1 (per token)
      6. trace_time_unchunk: (B, H*n_chunks, 1) -> (B, H, W) -> (B, 1, H, W)

    Compatible with train_fpmV3_ddp.py:
        forward(x, t, condL=...) where x:(B,2,H,W), t:(B,), condL:(rx,ry,sx,sy)
        output: (B,1,H,W)
    """
    def __init__(
        self,
        image_channels=2,
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_bottleneck_layers=4,
        num_decoder_layers=4,
        d_ff=None,
        dropout=0.1,
        output_channels=1,
        headwise_attn_output_gate=True,
        elementwise_attn_output_gate=False,
        use_qk_norm=True,
        qkv_bias=False,
        rms_norm_eps=1e-8,
        hidden_act='gelu',
        norm_type='rms',
        chunk_length=128,
        chunk_overlap=0.5,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 4)
        self.d_model = d_model
        self.image_channels = image_channels

        # Chunk params
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap

        # Conv2d tokenizer: fuse data + condition chunks
        # After chunking, we reshape to (B, n_chunks, H, chunk_length) for both data and condition
        # n_chunks is dynamic (depends on W and chunk_length), so we use 1x1 conv with groups
        # We'll build the conv layers lazily in forward once n_chunks is known,
        # OR use a fixed max_chunks approach. For simplicity, use 1x1 conv on concatenated features.
        # Actually, chunk_length is fixed, so we can use:
        #   cat(data_chunks, cond_chunks) -> (B, 2, H, chunk_length) per chunk
        # But chunks are processed together. Better: treat each chunk independently first,
        # then fuse. Let's use per-token processing:
        #   Each token = one chunk of one trace = (chunk_length,) features
        #   data + cond concatenated -> (2 * chunk_length,) -> Linear -> d_model
        # This avoids the dynamic n_chunks problem entirely.

        # Embedding: Linear(2 * chunk_length, d_model) — data_chunk + cond_chunk per token
        self.embed = nn.Linear(2 * chunk_length, d_model)
        self.embed_norm = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)

        # Encoder: SimpleGatedEncoder (stacked GatedTransformerEncoderBlock)
        self.encoder = SimpleGatedEncoder(
            input_dim=d_model,
            embed_dim=d_model,
            num_layers=num_encoder_layers,
            num_heads=nhead,
            d_ff=d_ff,
            dropout=dropout,
            norm_type=norm_type,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps,
            hidden_act=hidden_act,
            use_rope=True,
        )

        # Bottleneck: GatedDiTBlock with adaLN-Zero
        self.bottleneck_layers = nn.ModuleList([
            GatedDiTBlock(
                d_model=d_model,
                n_heads=nhead,
                d_ff=d_ff,
                dropout=dropout,
                headwise_attn_output_gate=headwise_attn_output_gate,
                elementwise_attn_output_gate=elementwise_attn_output_gate,
                use_qk_norm=use_qk_norm,
                qkv_bias=qkv_bias,
                rms_norm_eps=rms_norm_eps,
                hidden_act=hidden_act,
            )
            for _ in range(num_bottleneck_layers)
        ])

        # Decoder: SimpleGatedDecoder (stacked GatedTransformerEncoderBlock)
        self.decoder = SimpleGatedDecoder(
            input_dim=d_model,
            embed_dim=d_model,
            num_layers=num_decoder_layers,
            num_heads=nhead,
            d_ff=d_ff,
            dropout=dropout,
            norm_type=norm_type,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps,
            hidden_act=hidden_act,
            use_rope=True,
        )

        # Final adaLN
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # Output head: project to chunk_length (reconstruct the time chunk)
        self.head = nn.Linear(d_model, chunk_length, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Time embedding
        self.time_emb = GenTimeEmbedding(d_model)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            pass
        elif isinstance(m, nn.Parameter):
            if m.data.abs().max() > 100:
                m.data.clamp_(-100, 100)

    def _build_position_ids(self, coords_chunked, time_bounds):
        """
        Build 6D position_ids for RoPE from coords and time bounds.
        Args:
            coords_chunked: (B, H*n_chunks, 4) — [rx, ry, sx, sy] per token
            time_bounds: (B, H*n_chunks, 2) — [start_idx, end_idx] per token
        Returns: (B, H*n_chunks, 6) where 6 = [rx, ry, sx, sy, time_start_norm, time_end_norm]
        """
        # Normalize time bounds to [0, 1] globally (across all tokens)
        max_t = time_bounds.max().clamp(min=1.0)
        time_idx = (time_bounds / max_t).float().clamp(0.0, 1.0)
        # Concatenate: (B, seq_len, 6) = [rx, ry, sx, sy, time_start, time_end]
        pos = torch.cat([coords_chunked, time_idx], dim=-1)
        return pos

    def _build_conditioning(self,t,pos):
        """
        Build conditioning embedding: (B, 1, d_model)
        """
        t_emb = self.time_emb(t)
        c = t_emb.unsqueeze(1)
        return c

    def forward(self, x: torch.Tensor, t: torch.Tensor, condL=None, log_tau=None, time_axis=None, training=False):
        """
        Args:
            x: (B, 2, H, W) -- x[:,0:1]=data, x[:,1:2]=condition/mask
            t: (B,) diffusion timestep
            condL: tuple (rx, ry, sx, sy) each (B, H) — receiver/source coordinates
        Returns:
            (B, 1, H, W) predicted output
        """
        B, C_in, H, W = x.shape

        # Build coords: (B, H, 4) with centering
        if condL is not None:
            rx, ry, sx, sy = condL
            x_mean = sx.mean(dim=-1, keepdim=True)
            y_mean = sy.mean(dim=-1, keepdim=True)
            sx_c = sx - x_mean
            sy_c = sy - y_mean
            rx_c = rx - x_mean
            ry_c = ry - y_mean
            coords = torch.stack([rx_c, ry_c, sx_c, sy_c], dim=-1)
        else:
            coords = torch.zeros(B, H, 4, device=x.device, dtype=x.dtype)

        x_in, x_cond = x[:, 0:1], x[:, 1:2]

        # Chunk along time axis: (B, H, W) -> (B, H*n_chunks, chunk_length)
        x_chunked, coords_chunked, time_bounds, chunk_info = trace_time_chunk(
            x_in.squeeze(1), coords, chunk_length=self.chunk_length, overlap_ratio=self.chunk_overlap
        )
        x_cond_chunked, _, _, _ = trace_time_chunk(
            x_cond.squeeze(1), coords, chunk_length=self.chunk_length, overlap_ratio=self.chunk_overlap
        )

        seq_len = x_chunked.shape[1]  # H * n_chunks

        # Embed: concat data chunk + condition chunk per token -> Linear -> d_model
        # x_chunked: (B, seq_len, chunk_length), x_cond_chunked: (B, seq_len, chunk_length)
        x = torch.cat([x_chunked, x_cond_chunked], dim=-1)  # (B, seq_len, 2*chunk_length)
        x = self.embed_norm(self.embed(x))  # (B, seq_len, d_model)

        # Build 6D position_ids for RoPE: (B, seq_len, 6)
        position_ids = self._build_position_ids(coords_chunked, time_bounds)

        # Build conditioning
        c = self._build_conditioning(t)

        # Encoder
        encoder_features, encoder_out = self.encoder(x, skip_initial_proj=True, position_ids=position_ids)

        # Bottleneck (adaLN-Zero conditioned)
        x = encoder_out
        for block in self.bottleneck_layers:
            x = block(x, c, position_ids=position_ids)

        # Final adaLN modulation
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)

        # Decoder (with skip connection from encoder)
        x = self.decoder(x, skip_features=encoder_features, position_ids=position_ids)

        # Output head: (B, seq_len, d_model) -> (B, seq_len, chunk_length)
        x = self.head(x)

        # Unchunk: (B, H*n_chunks, chunk_length) -> (B, H, W)
        x = trace_time_unchunk(x, chunk_info, overlap_ratio=self.chunk_overlap)

        # Reshape to (B, 1, H, W)
        x = x.unsqueeze(1)
        return x


def create_gated_seisdit(
    image_channels=2,
    d_model=576,
    d_ff=2048,
    nhead=8,
    num_encoder_layers=4,
    num_bottleneck_layers=4,
    num_decoder_layers=4,
    dropout=0.1,
    headwise_attn_output_gate=True,
    elementwise_attn_output_gate=False,
    use_qk_norm=True,
    qkv_bias=False,
    chunk_length=128,
    chunk_overlap=0.5,
    **kwargs,
):
    """Factory function for GatedSeisDiT generation model.

    6D RoPE 要求 head_dim (=d_model//n_heads) 能被 12 整除。
    推荐组合: d_model=384/nhead=8, d_model=576/nhead=8, d_model=576/nhead=12
    """
    head_dim = d_model // nhead
    if head_dim % 12 != 0:
        raise ValueError(
            f"6D RoPE 要求 head_dim={head_dim} (d_model={d_model}//nhead={nhead}) "
            f"能被 12 整除。推荐: d_model=384/nhead=8 或 d_model=576/nhead=8"
        )
    return GatedSeisDiT(
        image_channels=image_channels,
        chunk_length=chunk_length,
        chunk_overlap=chunk_overlap,
        d_model=d_model,
        d_ff=d_ff,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_bottleneck_layers=num_bottleneck_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        headwise_attn_output_gate=headwise_attn_output_gate,
        elementwise_attn_output_gate=elementwise_attn_output_gate,
        use_qk_norm=use_qk_norm,
        qkv_bias=qkv_bias,
        **kwargs,
    )


if __name__ == "__main__":
    print("=== 测试 GatedSeisDiT (gated_transformer_v5 架构) ===")
    B, H, W = 2, 64, 128
    chunk_length = 64
    chunk_overlap = 0.5
    model = create_gated_seisdit(
        d_model=576,
        nhead=8,
        chunk_length=chunk_length,
        chunk_overlap=chunk_overlap,
        num_encoder_layers=4,
        num_bottleneck_layers=4,
        num_decoder_layers=4,
        dropout=0.1,
        headwise_attn_output_gate=True,
        elementwise_attn_output_gate=False,
        use_qk_norm=True,
        qkv_bias=False,
    )
    x = torch.randn(B, 2, H, W)
    t = torch.rand(B)
    rx = torch.randn(B, H)
    ry = torch.randn(B, H)
    sx = torch.randn(B, H)
    sy = torch.randn(B, H)

    out = model(x, t, condL=(rx, ry, sx, sy))
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # Test without condL
    out2 = model(x, t, condL=None)
    print(f"无 condL 输出形状: {out2.shape}")
