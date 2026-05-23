#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GatedSeisDiT — 基于 Gated Transformer 的地震数据生成模型（Diffusion / Flow Matching）

架构：
  - Encoder: SimpleGatedEncoder (stacked GatedTransformerEncoderBlock + 6D RoPE)
  - Bottleneck: GatedDiTBlock (adaLN-Zero, 6D RoPE, flatten 全局注意力)
  - Decoder: SimpleGatedDecoder (stacked GatedTransformerEncoderBlock + 6D RoPE)
  - Tokenizer: 分离的 data/cond embedding + 能量感知 + 缺失状态 embedding

融合方式（正确的四路相加）：
  data_emb + cond_emb + energy_emb + missing_emb -> embed_norm -> transformer

Forward 签名与训练代码 train_fpmV3_ddp.py 兼容:
  forward(x, t, condL=...) where x:(B,2,H,W), t:(B,), condL:(rx,ry,sx,sy)
  output: (B,1,H,W)
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
        self.lin1 = nn.Linear(self.n_channels // 4, n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, n_channels)

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
      2. 分离 embedding: embed_data + embed_cond + energy_emb + missing_emb
      3. Encoder -> Bottleneck -> Decoder
      4. Head: d_model -> chunk_length (per token)
      5. trace_time_unchunk: (B, H*n_chunks, chunk_length) -> (B, H, W) -> (B, 1, H, W)
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
        use_energy_stats=True,
        use_missing_embed=True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 4)
        self.d_model = d_model
        self.image_channels = image_channels
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap
        self.use_energy_stats = use_energy_stats
        self.use_missing_embed = use_missing_embed

        # ---- 分离的 data/cond embedding（不再 concat）----
        self.embed_data = nn.Linear(chunk_length, d_model)
        self.embed_cond = nn.Linear(chunk_length, d_model)
        self.embed_norm = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)

        # ---- 能量感知 MLP：输入 chunk 级别的 RMS energy -> d_model ----
        if use_energy_stats:
            self.energy_mlp = nn.Sequential(
                nn.Linear(3, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )

        # ---- 缺失状态 embedding：0=缺失, 1=观测 ----
        if use_missing_embed:
            self.missing_embed = nn.Embedding(2, d_model)
        
        self.fuse_mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # ---- Encoder ----
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

        # ---- Bottleneck ----
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

        # ---- Decoder ----
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

        # ---- Final adaLN ----
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # ---- Output head ----
        self.head = nn.Linear(d_model, chunk_length, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # ---- Time embedding ----
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
        max_t = time_bounds.max().clamp(min=1.0)
        time_idx = (time_bounds / max_t).float().clamp(0.0, 1.0)
        pos = torch.cat([coords_chunked, time_idx], dim=-1)
        return pos
    
    def _build_cond_statistics(self, x_cond_chunked: torch.Tensor):
        """
        从条件输入块中提取推理期可见的统计量。

        Returns:
            cond_stats: (B, seq_len, 3) = [log_rms_obs, log_mean_abs_obs]
        """
        obs_mask = (x_cond_chunked.abs() > 1e-6).float()
        obs_count = obs_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        coverage = obs_mask.mean(dim=-1, keepdim=True)

        sq_sum = ((x_cond_chunked ** 2) * obs_mask).sum(dim=-1, keepdim=True)
        abs_sum = (x_cond_chunked.abs() * obs_mask).sum(dim=-1, keepdim=True)

        rms_obs = torch.sqrt(sq_sum / obs_count + 1e-8)
        mean_abs_obs = abs_sum / obs_count

        cond_stats = torch.cat(
            [
                torch.log1p(rms_obs),
                torch.log1p(mean_abs_obs),
                coverage,
            ],
            dim=-1,
        )
        return cond_stats

    def _build_conditioning(self, t):
        """Build conditioning embedding: (B, d_model) -> broadcast to seq"""
        t_emb = self.time_emb(t)
        return t_emb

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

        # ---- Build coords: (B, H, 4) with centering ----
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

        # ---- Per-trace missing mask (before chunking) ----
        # missing_mask: (B, H, W) — 1=观测, 0=缺失
        #missing_mask = (x_cond.abs().max(dim=-1, keepdim=True)[0].squeeze(-1) > 0).float()
        missing_mask = (x_cond.squeeze(1).abs().max(dim=-1)[0] > 0).float()
        #print('missing_mask.shape', missing_mask.shape)
        missing_mask = missing_mask.unsqueeze(-1).repeat(1,1,W)
        # ---- Chunk along time axis: (B, H, W) -> (B, H*n_chunks, chunk_length) ----
        x_chunked, coords_chunked, time_bounds, chunk_info = trace_time_chunk(
            x_in.squeeze(1), coords, chunk_length=self.chunk_length, overlap_ratio=self.chunk_overlap
        )
        x_cond_chunked, _, _, _ = trace_time_chunk(
            x_cond.squeeze(1), coords, chunk_length=self.chunk_length, overlap_ratio=self.chunk_overlap
        )
        missing_chunked, _, _, _ = trace_time_chunk(
            missing_mask, coords, chunk_length=self.chunk_length, overlap_ratio=self.chunk_overlap
        )
        #print(x_chunked.shape, x_cond_chunked.shape, missing_chunked.shape)
        # ---- 四路分离 embedding + 融合 ----
        data_emb = self.embed_data(x_chunked)          # (B, seq, d_model)
        cond_emb = self.embed_cond(x_cond_chunked)     # (B, seq, d_model)

        # ---- Energy: 直接对每个 chunk 计算 RMS（有真正的时间分辨率）----
        if self.use_energy_stats:
            cond_stats = self._build_cond_statistics(x_cond_chunked)
            energy_emb = self.energy_mlp(cond_stats)  # (B, seq, d_model)
        else:
            energy_emb = 0.0

        if self.use_missing_embed:
            # missing_chunked: (B, seq, chunk_length) -> 每时刻取平均 -> (B, seq)
            missing_per_token = missing_chunked.mean(dim=-1).clamp(0, 1).long()  # (B, seq)
            missing_emb = self.missing_embed(missing_per_token)  # (B, seq, d_model)
        else:
            missing_emb = 0.0

        # 融合：data + cond + energy + missing（全部在 d_model 维度）
        #print(data_emb.shape, cond_emb.shape, energy_emb.shape, missing_emb.shape)
        cond_emb = cond_emb + energy_emb + missing_emb
        x = torch.cat([data_emb, cond_emb], dim=-1)
        x = self.fuse_mlp(x)
        x = self.embed_norm(x)  # (B, seq_len, d_model)

        # ---- 6D RoPE position_ids ----
        position_ids = self._build_position_ids(coords_chunked, time_bounds)

        # ---- Conditioning ----
        c = self._build_conditioning(t)  # (B, d_model)
        # broadcast to seq_len
        c = c.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, seq_len, d_model)

        # ---- Encoder ----
        encoder_features, encoder_out = self.encoder(x, skip_initial_proj=True, position_ids=position_ids)

        # ---- Bottleneck ----
        x = encoder_out
        for block in self.bottleneck_layers:
            x = block(x, c, position_ids=position_ids)

        # ---- Final adaLN + Decoder ----
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.decoder(x, skip_features=encoder_features, position_ids=position_ids)

        # ---- Output head ----
        x = self.head(x)  # (B, seq_len, chunk_length)

        # ---- Unchunk: (B, H*n_chunks, chunk_length) -> (B, H, W) ----
        x = trace_time_unchunk(x, chunk_info, overlap_ratio=self.chunk_overlap)
        x_out = x.unsqueeze(1)  # (B, 1, H, W)
        return x_out


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
    chunk_length=256,
    chunk_overlap=0.5,
    use_energy_stats=True,
    use_missing_embed=True,
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
        use_energy_stats=use_energy_stats,
        use_missing_embed=use_missing_embed,
        **kwargs,
    )


if __name__ == "__main__":
    print("=== 测试 GatedSeisDiT (正确的四路融合) ===")
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
        use_energy_stats=True,
        use_missing_embed=True,
    )
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(B, 2, H, W)
    t = torch.rand(B)
    rx = torch.randn(B, H)
    ry = torch.randn(B, H)
    sx = torch.randn(B, H)
    sy = torch.randn(B, H)

    out = model(x, t, condL=(rx, ry, sx, sy))
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")

    # Test ablation: 关闭 energy 和 missing embed
    model_no_ab = create_gated_seisdit(
        d_model=576, nhead=8, chunk_length=chunk_length, chunk_overlap=chunk_overlap,
        num_encoder_layers=4, num_bottleneck_layers=4, num_decoder_layers=4, dropout=0.1,
        use_energy_stats=True, use_missing_embed=True,
    )
    out2 = model_no_ab(x, t, condL=(rx, ry, sx, sy))
    print(f"[关闭 ablations] 输出形状: {out2.shape}")
