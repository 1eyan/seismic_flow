#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4-aligned encoder-decoder seismic interpolation transformer.

对齐 GatedSeismicInterpolationTransformerV4 的接口与数据流：
- 输入仍是外部已经 chunk 好的 token 序列 x:(B, L, input_dim)
- coords:(B, L, 4), time_bounds:(B, L, 2) 的用法保持一致
- 输出仍是 (B, L, output_dim)，默认 output_dim=input_dim

和原 V4 的区别：
- 原 V4: 单流 self-attention encoder -> self-attention decoder(skip)
- 本文件: observed-only encoder(memory) -> full-query decoder(self-attn + cross-attn)
"""

import torch
import torch.nn as nn

from .gated_transformer_v5_ed import (
    AbsoluteCoordinateEncoding,
    RMSNorm,
    SimpleGatedCrossDecoder,
    SimpleGatedEncoder,
    TimeSegmentEncoding,
    get_norm_layer,
)


class GatedSeismicInterpolationTransformerV4EncDec(nn.Module):
    """对齐原 V4 接口的 encoder-decoder 版本。"""

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
        )

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
        if mask is None:
            return (~torch.all(x == 0, dim=-1)).float()

        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.dim() != 2:
            raise ValueError(
                f"mask should be (B, L) or (B, L, 1) observed-token mask, got shape={tuple(mask.shape)}"
            )
        if mask.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match token shape {tuple(x.shape[:2])}. "
                "EncDec 需要 (B, seq_len) 的 token 级 mask，seq_len 须与 x 一致（trace_time_chunk 后为 n_traces*n_chunks）。"
                "若你传入的是道级 (B, n_traces)，请在训练脚本中按 chunk 顺序展开到每个 token（见 train_dongfang_gated_v5.expand_trace_mask_to_token_mask）。"
            )
        return mask.float()

    @staticmethod
    def _gather_observed_tokens(tokens, observed_mask, position_ids=None):
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
        输入:
            x: (B, L, input_dim) 已切块 token
            coords: (B, L, 4)，值建议已归一化到 [0, 1]
            time_bounds: (B, L, 2)，值建议已归一化到 [0, 1]
            mask: (B, L) 或 (B, L, 1)，1=observed token, 0=missing token
        输出:
            (B, L, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        input_x = x

        if coords is None:
            coords = torch.zeros(batch_size, seq_len, 4, dtype=x.dtype, device=x.device)
        if time_bounds is None:
            time_bounds = torch.zeros(batch_size, seq_len, 2, dtype=torch.float32, device=x.device)

        time_norm = time_bounds.float().clamp(0.0, 1.0)

        x = self.input_norm(self.input_proj(x))
        coords_6d = None
        if self.use_coord_encoding:
            coords_6d = torch.cat([coords.float(), time_norm], dim=-1)
            x = self.coord_encoding(x, coords_6d)
        if not self.use_rope:
            max_time = time_bounds.max().item() + 1.0
            x = self.time_segment_encoding(x, time_bounds, max_time=max_time)
        x = self.dropout(x)

        position_ids = coords_6d if (self.use_rope and coords_6d is not None) else None
        observed_mask = self._normalize_observed_mask(mask, input_x).bool()

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

        decoded = self.decoder(
            x,
            memory,
            skip_initial_proj=True,
            memory_mask=memory_mask,
            position_ids=position_ids,
            memory_position_ids=memory_pos,
        )
        output = self.output_proj(decoded)
        # 观测道直接复用输入值，缺失道用模型预测
        obs = observed_mask.float().unsqueeze(-1)  # (B, L, 1)
        return output


def create_gated_model_v4_encdec(
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
):
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
                f"能被 {coord_dim} 整除且每份 {head_dim // coord_dim} 为偶数"
            )

    return GatedSeismicInterpolationTransformerV4EncDec(
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
    )


if __name__ == "__main__":
    print("=== 测试 V4-aligned Encoder-Decoder ===")
    batch_size = 2
    seq_len = 120
    input_dim = 64

    model = create_gated_model_v4_encdec(
        input_dim=input_dim,
        d_model=384,
        n_heads=8,
        num_layers=2,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    x[:, 30:50] = 0.0
    coords = torch.rand(batch_size, seq_len, 4)
    time_bounds = torch.rand(batch_size, seq_len, 2)
    time_bounds = torch.sort(time_bounds, dim=-1)[0]
    obs_mask = (~torch.all(x == 0, dim=-1)).float()

    y = model(x, coords=coords, time_bounds=time_bounds, mask=obs_mask)
    print(f"input shape: {x.shape}")
    print(f"output shape: {y.shape}")
