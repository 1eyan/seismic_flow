#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V9 Encoder-Decoder 地震插值 Transformer.

将 V9 的 Block Attention Residuals (Kimi, arXiv:2603.15031) 引入
encoder-decoder 架构（observed-only encoder + cross-attention decoder）。

Encoder: 直接复用 V9 的 SimpleGatedEncoder(use_attn_res=True)
Decoder: Cross-attention decoder（使用 V9 GatedMultiHeadAttention + FeedForward 构建）

与 gated_transformer_v9.py 保持 encoder 组件完全一致。
所有组件均来自 V9，零外部依赖。

前向接口:
    forward(x, coords, time_bounds, mask) -> (B, L, output_dim)
"""

import torch
import torch.nn as nn

# V9 组件 —— 直接复用
from .gated_transformer_v9 import (
    SimpleGatedEncoder,
    GatedMultiHeadAttention,
    FeedForward,
    get_norm_layer,
    RMSNorm,
    AbsoluteCoordinateEncoding,
    TimeSegmentEncoding,
)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# =============================================================================
# Cross-Attention Decoder（使用 V9 组件构建）
# =============================================================================

class CrossAttnDecoderBlock(nn.Module):
    """Cross-Attention 解码器块 —— 镜像 V9 GatedTransformerEncoderBlock 的 Pre-LN 结构。

    子层:
      1. Self-attention (Q=K=V=decoder input)
      2. Cross-attention (Q=decoder, K=V=encoder memory)
      3. FeedForward

    所有子模块均使用 V9 的 GatedMultiHeadAttention / FeedForward / get_norm_layer。
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False,
                 qkv_bias=False,
                 norm_type='rms',
                 rms_norm_eps=1e-8,
                 hidden_act='gelu',
                 use_rope=True,
                 use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        # Self-attention sub-layer
        self.norm1 = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)
        self.self_attn = GatedMultiHeadAttention(
            d_model, n_heads, dropout,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps,
            use_rope=use_rope,
        )
        self.drop1 = nn.Dropout(dropout)

        # Cross-attention sub-layer
        self.norm2 = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)
        self.memory_norm = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)
        self.cross_attn = GatedMultiHeadAttention(
            d_model, n_heads, dropout,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps,
            use_rope=use_rope,
        )
        self.drop2 = nn.Dropout(dropout)

        # FFN sub-layer
        self.norm3 = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)
        self.ffn = FeedForward(d_model, d_ff, dropout, hidden_act=hidden_act)
        self.drop3 = nn.Dropout(dropout)

        if self.use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, 9 * d_model, bias=True),
            )
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            self.adaLN_modulation = None

    def forward(self, x, memory, self_mask=None, memory_mask=None,
                position_ids=None, memory_position_ids=None,
                self_attention_bias=None, memory_attention_bias=None,
                missing_mask=None, conditioning=None, cross_query_gate=None):
        if self.use_adaln and conditioning is not None:
            (
                shift_self,
                scale_self,
                gate_self,
                shift_cross,
                scale_cross,
                gate_cross,
                shift_ffn,
                scale_ffn,
                gate_ffn,
            ) = self.adaLN_modulation(conditioning).chunk(9, dim=-1)
        else:
            shift_self = scale_self = shift_cross = scale_cross = shift_ffn = scale_ffn = 0.0
            gate_self = gate_cross = gate_ffn = 1.0

        # 1. Self-attention
        x_norm = modulate(self.norm1(x), shift_self, scale_self)
        self_out = self.self_attn(
            x_norm, x_norm, x_norm,
            mask=self_mask,
            position_ids=position_ids,
            key_position_ids=position_ids,
            attention_bias=self_attention_bias,
            missing_mask=missing_mask,
        )
        x = x + gate_self * self.drop1(self_out)

        # 2. Cross-attention: Q=decoder, K/V=memory
        x_norm = modulate(self.norm2(x), shift_cross, scale_cross)
        mem_norm = self.memory_norm(memory)
        cross_out = self.cross_attn(
            x_norm, mem_norm, mem_norm,
            mask=memory_mask,
            position_ids=position_ids,
            key_position_ids=memory_position_ids,
            attention_bias=memory_attention_bias,
        )
        if cross_query_gate is not None:
            cross_out = cross_out * cross_query_gate.to(dtype=cross_out.dtype, device=cross_out.device)
        x = x + gate_cross * self.drop2(cross_out)

        # 3. FFN
        x_norm = modulate(self.norm3(x), shift_ffn, scale_ffn)
        x = x + gate_ffn * self.drop3(self.ffn(x_norm))
        return x


class SimpleGatedCrossDecoder(nn.Module):
    """Cross-Attention 解码器 —— 镜像 V9 SimpleGatedDecoder 结构。

    与 SimpleGatedDecoder 的对应关系:
      SimpleGatedDecoder.layers      -> self.layers (CrossAttnDecoderBlock 替代 GatedTransformerEncoderBlock)
      SimpleGatedDecoder.final_proj  -> self.final_proj (Linear + norm, 完全相同)
      + initial_proj / initial_norm（与 V9 encoder 的 initial 结构保持一致）
    """

    def __init__(self, input_dim, embed_dim=1024, num_layers=4,
                 num_heads=16, d_ff=2048, dropout=0.1, norm_type='rms',
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False, qkv_bias=False,
                 rms_norm_eps=1e-8, hidden_act='gelu',
                 use_rope=True,
                 use_adaln=False):
        super().__init__()
        self.initial_proj = nn.Linear(input_dim, embed_dim)
        self.initial_norm = get_norm_layer(norm_type, embed_dim, eps=rms_norm_eps)

        self.layers = nn.ModuleList([
            CrossAttnDecoderBlock(
                embed_dim, num_heads, d_ff, dropout,
                headwise_attn_output_gate=headwise_attn_output_gate,
                elementwise_attn_output_gate=elementwise_attn_output_gate,
                use_qk_norm=use_qk_norm, qkv_bias=qkv_bias,
                norm_type=norm_type, rms_norm_eps=rms_norm_eps,
                hidden_act=hidden_act,
                use_rope=use_rope,
                use_adaln=use_adaln,
            )
            for _ in range(num_layers)
        ])

        # 镜像 V9 SimpleGatedDecoder 的 final_proj: Linear + Norm
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, input_dim),
            get_norm_layer(norm_type, input_dim, eps=rms_norm_eps),
        )

    def forward(self, x, memory, skip_initial_proj=False,
                self_mask=None, memory_mask=None,
                position_ids=None, memory_position_ids=None,
                self_attention_bias=None, memory_attention_bias=None,
                missing_mask=None, conditioning=None, cross_query_gate=None):
        if not skip_initial_proj:
            x = self.initial_proj(x)
            x = self.initial_norm(x)
        for layer in self.layers:
            x = layer(x, memory,
                      self_mask=self_mask, memory_mask=memory_mask,
                      position_ids=position_ids,
                      memory_position_ids=memory_position_ids,
                      self_attention_bias=self_attention_bias,
                      memory_attention_bias=memory_attention_bias,
                      missing_mask=missing_mask,
                      conditioning=conditioning,
                      cross_query_gate=cross_query_gate)
        return self.final_proj(x)


class GatedSeismicInterpolationTransformerV9EncDec(nn.Module):
    """V9 Encoder-Decoder 地震插值 Transformer。

    Encoder: 直接复用 V9 的 SimpleGatedEncoder(use_attn_res=True)
    Decoder: Cross-attention decoder（使用 V9 GatedMultiHeadAttention + FeedForward 构建）
    """

    def __init__(
        self,
        input_dim,
        d_model=512,
        n_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_ff=2048,
        dropout=0.1,
        output_dim=None,
        norm_type="rms",
        headwise_attn_output_gate=False,
        elementwise_attn_output_gate=False,
        use_qk_norm=False,
        qkv_bias=False,
        rms_norm_eps=1e-8,
        hidden_act="gelu",
        use_coord_encoding=True,
        use_rope=True,
        coord_dim=6,
        coord_max_freq=1.0,
        encode_observed_only=True,
        num_attn_res_blocks=2,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.norm_type = norm_type
        self.use_coord_encoding = use_coord_encoding
        self.use_rope = use_rope
        self.encode_observed_only = encode_observed_only

        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)

        if self.use_coord_encoding:
            self.coord_encoding = AbsoluteCoordinateEncoding(
                d_model, coord_dim=coord_dim, max_freq=coord_max_freq
            )
        if not self.use_rope:
            self.time_segment_encoding = TimeSegmentEncoding(d_model)

        # V9 encoder —— 直接复用，use_attn_res=True
        self.encoder = SimpleGatedEncoder(
            input_dim=d_model,
            embed_dim=d_model,
            num_layers=num_encoder_layers,
            num_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            norm_type=norm_type,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps,
            hidden_act=hidden_act,
            use_rope=use_rope,
            use_attn_res=True,
            num_attn_res_blocks=num_attn_res_blocks,
        )

        # Cross-attention decoder
        self.decoder = SimpleGatedCrossDecoder(
            input_dim=d_model,
            embed_dim=d_model,
            num_layers=num_decoder_layers,
            num_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            norm_type=norm_type,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps,
            hidden_act=hidden_act,
            use_rope=use_rope,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            get_norm_layer(norm_type, output_dim, eps=rms_norm_eps),
        )

        self.dropout = nn.Dropout(dropout)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            pass
        elif isinstance(m, nn.Parameter):
            if m.data.abs().max() > 100:
                m.data.clamp_(-100, 100)

    @staticmethod
    def _normalize_observed_mask(mask, x):
        """将 mask 标准化为 (B, L) 的 observed-token mask (1=观测, 0=缺失)。"""
        if mask is None:
            return (~torch.all(x == 0, dim=-1)).float()

        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.dim() != 2:
            raise ValueError(
                f"mask should be (B, L) or (B, L, 1) observed-token mask, "
                f"got shape={tuple(mask.shape)}"
            )
        if mask.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match token shape "
                f"{tuple(x.shape[:2])}. "
                "EncDec 需要 (B, seq_len) 的 token 级 mask，"
                "seq_len 须与 x 一致（trace_time_chunk 后为 n_traces*n_chunks）。"
            )
        return mask.float()

    @staticmethod
    def _gather_observed_tokens(tokens, observed_mask, position_ids=None):
        """将观测 token 紧凑排列，返回 (memory_tokens, memory_mask, memory_pos)。"""
        batch_size, _, dim = tokens.shape
        max_obs = max(1, int(observed_mask.sum(dim=1).max().item()))

        memory_tokens = tokens.new_zeros(batch_size, max_obs, dim)
        memory_mask = tokens.new_zeros(batch_size, max_obs)
        memory_pos = None
        if position_ids is not None:
            memory_pos = position_ids.new_zeros(batch_size, max_obs, position_ids.shape[-1])

        for batch_idx in range(batch_size):
            obs_idx = observed_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            if obs_idx.numel() == 0:
                memory_tokens[batch_idx, 0] = tokens[batch_idx, 0]
                memory_mask[batch_idx, 0] = 1.0
                if memory_pos is not None:
                    memory_pos[batch_idx, 0] = position_ids[batch_idx, 0]
                continue

            count = obs_idx.numel()
            memory_tokens[batch_idx, :count] = tokens[batch_idx, obs_idx]
            memory_mask[batch_idx, :count] = 1.0
            if memory_pos is not None:
                memory_pos[batch_idx, :count] = position_ids[batch_idx, obs_idx]

        return memory_tokens, memory_mask, memory_pos

    def forward(self, x, coords=None, time_bounds=None, mask=None):
        """
        Args:
            x: (B, L, input_dim) — 已切块 token
            coords: (B, L, 4) — 空间坐标 [sx, sy, rx, ry]，已归一化到 [0, 1]
            time_bounds: (B, L, 2) — 时间边界 [start, end]，已归一化到 [0, 1]
            mask: (B, L) or (B, L, 1) — 1=观测 token, 0=缺失 token
        Returns:
            (B, L, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        input_x = x

        if coords is None:
            coords = torch.zeros(batch_size, seq_len, 4, dtype=x.dtype, device=x.device)
        if time_bounds is None:
            time_bounds = torch.zeros(batch_size, seq_len, 2, dtype=torch.float32, device=x.device)

        time_norm = time_bounds.float().clamp(0.0, 1.0)

        # Step 1: Input projection + coordinate encoding
        x = self.input_norm(self.input_proj(x))
        coords_6d = None
        if self.use_coord_encoding:
            coords_6d = torch.cat([coords.float(), time_norm], dim=-1)
            x = self.coord_encoding(x, coords_6d)
        if not self.use_rope:
            max_time = time_bounds.max().item() + 1.0
            x = self.time_segment_encoding(x, time_bounds, max_time=max_time)
        x = self.dropout(x)

        # Step 2: Build position_ids & observed mask
        position_ids = coords_6d if (self.use_rope and coords_6d is not None) else None
        observed_mask = self._normalize_observed_mask(mask, input_x).bool()

        # Step 3: Encoder —— 复用 V9 的 SimpleGatedEncoder(use_attn_res=True)
        if self.encode_observed_only:
            memory_tokens, memory_mask, memory_pos = self._gather_observed_tokens(
                x, observed_mask, position_ids=position_ids
            )
        else:
            memory_tokens = x
            memory_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype)
            memory_pos = position_ids

        _, memory = self.encoder(
            memory_tokens,
            skip_initial_proj=True,
            mask=memory_mask,
            position_ids=memory_pos,
        )

        # Step 4: Cross-attention decoder
        decoded = self.decoder(
            x,
            memory,
            skip_initial_proj=True,
            memory_mask=memory_mask,
            position_ids=position_ids,
            memory_position_ids=memory_pos,
        )

        # Step 5: Output projection
        output = self.output_proj(decoded)
        return output


def create_gated_model_v9_encdec(
    input_dim,
    d_model=512,
    n_heads=8,
    num_layers=4,
    num_encoder_layers=None,
    num_decoder_layers=None,
    d_ff=2048,
    dropout=0.1,
    output_dim=None,
    norm_type="rms",
    headwise_attn_output_gate=False,
    elementwise_attn_output_gate=False,
    use_qk_norm=False,
    qkv_bias=False,
    rms_norm_eps=1e-8,
    hidden_act="gelu",
    use_coord_encoding=True,
    use_rope=True,
    coord_dim=6,
    coord_max_freq=1.0,
    encode_observed_only=True,
    num_attn_res_blocks=2,
):
    """创建 V9 Encoder-Decoder 模型。

    encoder 直接复用 V9 的 SimpleGatedEncoder(use_attn_res=True)。
    decoder 使用本地定义的 cross-attention decoder（基于 V9 GatedMultiHeadAttention + FeedForward）。
    """
    if output_dim is None:
        output_dim = input_dim

    if num_encoder_layers is None:
        num_encoder_layers = num_layers
    if num_decoder_layers is None:
        num_decoder_layers = num_layers

    if use_rope and coord_dim == 6:
        head_dim = d_model // n_heads
        if head_dim % coord_dim != 0 or (head_dim // coord_dim) % 2 != 0:
            raise ValueError(
                f"6D RoPE 要求 head_dim={head_dim} (d_model={d_model}//n_heads={n_heads}) "
                f"能被 {coord_dim} 整除且每份 {head_dim//coord_dim} 为偶数"
            )

    return GatedSeismicInterpolationTransformerV9EncDec(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
        output_dim=output_dim,
        norm_type=norm_type,
        headwise_attn_output_gate=headwise_attn_output_gate,
        elementwise_attn_output_gate=elementwise_attn_output_gate,
        use_qk_norm=use_qk_norm,
        qkv_bias=qkv_bias,
        rms_norm_eps=rms_norm_eps,
        hidden_act=hidden_act,
        use_coord_encoding=use_coord_encoding,
        use_rope=use_rope,
        coord_dim=coord_dim,
        coord_max_freq=coord_max_freq,
        encode_observed_only=encode_observed_only,
        num_attn_res_blocks=num_attn_res_blocks,
    )


if __name__ == "__main__":
    print("=== 测试 V9 Encoder-Decoder (复用 V9 encoder) ===")

    batch_size = 2
    seq_len = 120
    input_dim = 64

    model = create_gated_model_v9_encdec(
        input_dim=input_dim,
        d_model=384,
        n_heads=8,
        num_layers=4,
        num_attn_res_blocks=2,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    x[:, 30:50] = 0.0
    coords = torch.rand(batch_size, seq_len, 4)
    time_bounds = torch.rand(batch_size, seq_len, 2)
    time_bounds = torch.sort(time_bounds, dim=-1)[0]
    obs_mask = (~torch.all(x == 0, dim=-1)).float()

    y = model(x, coords=coords, time_bounds=time_bounds, mask=obs_mask)

    print(f"input shape:  {x.shape}")
    print(f"output shape: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 验证梯度
    loss = y.mean()
    loss.backward()
    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"NaN 梯度: {has_nan}")

    # 对比: V9 EncDec encoder vs V9 原始 encoder
    from .gated_transformer_v9 import create_gated_model_v9

    model_v9 = create_gated_model_v9(
        input_dim=input_dim, d_model=384, n_heads=8, num_layers=4,
        use_attn_res=True, num_attn_res_blocks=2,
    )

    # Encoder 类型对比
    print(f"V9 EncDec encoder type:  {type(model.encoder).__name__}")
    print(f"V9 原始 encoder type:     {type(model_v9.encoder).__name__}")
    print(f"Encoder class 相同:      {type(model.encoder) is type(model_v9.encoder)}")

    print("测试通过!")
