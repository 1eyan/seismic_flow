#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generative V9 encoder-decoder backbone for conditional Flow Matching.

The model is compatible with FlowMatchingModel:
    forward(x, t, condL=...) where x[:, 0:1] is x_t and x[:, 1:2] is x_cond.

Encoder:
    observed condition tokens only.
Decoder:
    all x_t tokens as queries, cross-attending to observed condition memory.

Positioning:
    - 6D absolute coordinates are passed to V9 RoPE.
    - Patch-local observed-system geometry from sx/sy/rx/ry is converted to
      an attention bias, primarily for decoder cross-attention.

The conditioning path is intentionally minimal: x_cond itself, observed-only
memory selection, and geometry bias already carry the missing/energy cues.
"""

import math

import torch
import torch.nn as nn

from .gated_transformer_v9 import (
    RMSNorm,
    SimpleGatedEncoder,
    get_norm_layer,
    trace_time_chunk,
    trace_time_unchunk,
)
from .gated_transformer_v9_encdec import SimpleGatedCrossDecoder


class GenTimeEmbedding(nn.Module):
    """Sinusoidal flow timestep embedding followed by an MLP."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.n_channels // 8
        emb_scale = math.log(10_000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None].float() * freqs[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.lin2(self.act(self.lin1(emb)))


class ObservedSystemRelativeBias(nn.Module):
    """Pairwise attention bias from patch-local observed-system geometry."""

    def __init__(
        self,
        n_heads: int,
        hidden_dim: int = 64,
        init_scale: float = 0.0,
        prior_init: float = 0.05,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.mlp = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_heads),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.prior_scale = nn.Parameter(torch.tensor(float(prior_init)))

    def forward(self, geom_q: torch.Tensor, geom_k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geom_q: (B, Lq, 10) = local src/rec/mid/offset + time geometry.
            geom_k: (B, Lk, 10)
        Returns:
            (B, n_heads, Lq, Lk)
        """
        src_q, rec_q = geom_q[..., 0:2], geom_q[..., 2:4]
        mid_q, off_q = geom_q[..., 4:6], geom_q[..., 6:8]
        time_q = geom_q[..., 8:10]

        src_k, rec_k = geom_k[..., 0:2], geom_k[..., 2:4]
        mid_k, off_k = geom_k[..., 4:6], geom_k[..., 6:8]
        time_k = geom_k[..., 8:10]

        delta_src = src_q[:, :, None, :] - src_k[:, None, :, :]
        delta_rec = rec_q[:, :, None, :] - rec_k[:, None, :, :]
        delta_mid = mid_q[:, :, None, :] - mid_k[:, None, :, :]
        delta_off = off_q[:, :, None, :] - off_k[:, None, :, :]
        delta_time = time_q[:, :, None, :] - time_k[:, None, :, :]

        norm_delta_mid = torch.linalg.norm(delta_mid, dim=-1, keepdim=True)

        pair_features = torch.cat(
            [
                delta_src,
                delta_rec,
                delta_mid,
                delta_off,
                norm_delta_mid,
                delta_time,
                delta_time.abs(),
            ],
            dim=-1,
        )
        learned_bias = self.mlp(pair_features).permute(0, 3, 1, 2).contiguous()
        distance_prior = -norm_delta_mid.permute(0, 3, 1, 2).contiguous()
        return self.prior_scale * distance_prior + self.scale * learned_bias


class GatedSeisDiTEncDecGen(nn.Module):
    """Conditional generative encoder-decoder for seismic trace interpolation."""

    def __init__(
        self,
        image_channels=2,
        d_model=576,
        nhead=8,
        num_encoder_layers=4,
        num_memory_bottleneck_layers=2,
        num_decoder_layers=4,
        d_ff=None,
        dropout=0.1,
        output_channels=1,
        chunk_length=256,
        chunk_overlap=0.5,
        headwise_attn_output_gate=True,
        elementwise_attn_output_gate=False,
        use_qk_norm=True,
        qkv_bias=False,
        rms_norm_eps=1e-8,
        hidden_act="gelu",
        norm_type="rms",
        use_relative_bias=True,
        use_encoder_relative_bias=False,
        use_decoder_self_relative_bias=False,
        use_cross_relative_bias=True,
        relative_bias_hidden_dim=64,
        relative_bias_init_scale=0.0,
        relative_bias_prior_scale=0.05,
        use_time_adaln=True,
        use_cross_query_gate=True,
        num_attn_res_blocks=2,
        output_head_zero_init=True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 4)

        self.image_channels = image_channels
        self.d_model = d_model
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap
        self.output_channels = output_channels
        self.use_relative_bias = use_relative_bias
        self.use_encoder_relative_bias = use_encoder_relative_bias
        self.use_decoder_self_relative_bias = use_decoder_self_relative_bias
        self.use_cross_relative_bias = use_cross_relative_bias
        self.use_cross_query_gate = use_cross_query_gate
        self.missing_eps = 1e-6

        self.xt_embed = nn.Linear(chunk_length, d_model)
        self.cond_embed = nn.Linear(chunk_length, d_model)
        self.cond_query_embed = nn.Linear(chunk_length, d_model)
        self.query_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.query_norm = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)
        self.memory_norm = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)

        self.time_emb = GenTimeEmbedding(d_model)

        if self.use_cross_query_gate:
            self.cross_query_gate = nn.Embedding(2, 1)
            with torch.no_grad():
                self.cross_query_gate.weight[0].fill_(1.0)   # missing query: rely on memory
                self.cross_query_gate.weight[1].fill_(0.35)  # observed query: lighter memory residual
        else:
            self.cross_query_gate = None

        if use_relative_bias:
            self.relative_bias = ObservedSystemRelativeBias(
                n_heads=nhead,
                hidden_dim=relative_bias_hidden_dim,
                init_scale=relative_bias_init_scale,
                prior_init=relative_bias_prior_scale,
            )
        else:
            self.relative_bias = None

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
            use_attn_res=True,
            num_attn_res_blocks=num_attn_res_blocks,
        )

        if num_memory_bottleneck_layers and num_memory_bottleneck_layers > 0:
            self.memory_bottleneck = SimpleGatedEncoder(
                input_dim=d_model,
                embed_dim=d_model,
                num_layers=num_memory_bottleneck_layers,
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
                use_attn_res=True,
            )
        else:
            self.memory_bottleneck = None

        self.decoder = SimpleGatedCrossDecoder(
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
            use_adaln=use_time_adaln,
        )

        self.head = nn.Linear(d_model, chunk_length * output_channels, bias=True)
        self.initialize_weights()
        if output_head_zero_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            pass
        elif isinstance(m, nn.Parameter):
            if m.data.abs().max() > 100:
                m.data.clamp_(-100, 100)

    def _build_abs_coords(self, condL, batch_size, n_traces, device, dtype):
        if condL is None:
            trace_pos = torch.linspace(0.0, 1.0, n_traces, device=device, dtype=dtype)
            rx = sx = trace_pos.unsqueeze(0).expand(batch_size, -1)
            ry = sy = torch.zeros_like(rx)
        else:
            rx, ry, sx, sy = [v.to(device=device, dtype=dtype) for v in condL]
        return torch.stack([rx, ry, sx, sy], dim=-1)

    def _build_position_ids(self, coords_chunked, time_bounds):
        max_t = time_bounds.max().clamp(min=1.0)
        time_idx = (time_bounds / max_t).float().clamp(0.0, 1.0)
        return torch.cat([coords_chunked.float(), time_idx], dim=-1)

    def _infer_observed_mask(self, x_cond, missing_mask_input=None):
        if missing_mask_input is not None:
            if missing_mask_input.dim() == 3 and missing_mask_input.shape[-1] == 1:
                missing_mask_input = missing_mask_input.squeeze(-1)
            return missing_mask_input.float()
        trace_max = x_cond.abs().amax(dim=-1).squeeze(1)
        return (trace_max > self.missing_eps).float()

    @staticmethod
    def _expand_trace_mask(trace_mask, n_chunks):
        return trace_mask.unsqueeze(1).expand(-1, n_chunks, -1).reshape(trace_mask.shape[0], -1)

    @staticmethod
    def _expand_trace_features(trace_features, n_chunks):
        return trace_features.unsqueeze(1).expand(-1, n_chunks, -1, -1).reshape(
            trace_features.shape[0], -1, trace_features.shape[-1]
        )

    def _build_local_trace_geometry(self, condL, observed_trace_mask, batch_size, n_traces, device, dtype):
        if condL is None:
            trace_pos = torch.linspace(0.0, 1.0, n_traces, device=device, dtype=dtype)
            sx = rx = trace_pos.unsqueeze(0).expand(batch_size, -1)
            sy = ry = torch.zeros_like(sx)
        else:
            rx, ry, sx, sy = [v.to(device=device, dtype=dtype) for v in condL]

        src = torch.stack([sx, sy], dim=-1)
        rec = torch.stack([rx, ry], dim=-1)
        mid = 0.5 * (src + rec)

        obs = observed_trace_mask.to(device=device, dtype=dtype)
        obs_sum = obs.sum(dim=1, keepdim=True)
        has_obs = (obs_sum > 0).unsqueeze(-1)
        obs_sum_safe = obs_sum.clamp(min=1.0).unsqueeze(-1)

        obs_center = (mid * obs.unsqueeze(-1)).sum(dim=1, keepdim=True) / obs_sum_safe
        all_center = mid.mean(dim=1, keepdim=True)
        center = torch.where(has_obs, obs_center, all_center)

        src_dist2 = ((src - center) ** 2).sum(dim=-1)
        rec_dist2 = ((rec - center) ** 2).sum(dim=-1)
        obs_scale_sq = ((src_dist2 + rec_dist2) * obs).sum(dim=1, keepdim=True) / (
            2.0 * obs_sum.clamp(min=1.0)
        )
        all_scale_sq = (src_dist2 + rec_dist2).mean(dim=1, keepdim=True) / 2.0
        scale_sq = torch.where(has_obs.squeeze(-1), obs_scale_sq, all_scale_sq)
        scale = torch.sqrt(scale_sq + 1e-8).clamp(min=1e-3).unsqueeze(-1)

        src_l = (src - center) / scale
        rec_l = (rec - center) / scale
        mid_l = 0.5 * (src_l + rec_l)
        off_l = rec_l - src_l
        off_len = torch.linalg.norm(off_l, dim=-1, keepdim=True)
        azimuth = torch.atan2(off_l[..., 1:2], off_l[..., 0:1] + 1e-8)
        return torch.cat(
            [
                src_l,
                rec_l,
                mid_l,
                off_l,
                off_len,
                torch.sin(azimuth),
                torch.cos(azimuth),
            ],
            dim=-1,
        )

    def _build_token_rel_geometry(self, trace_geometry, n_chunks, time_bounds):
        geom = self._expand_trace_features(trace_geometry, n_chunks)
        max_t = time_bounds.max().clamp(min=1.0)
        time_norm = (time_bounds / max_t).float().clamp(0.0, 1.0).to(dtype=geom.dtype)
        return torch.cat([geom, time_norm], dim=-1)

    @staticmethod
    def _gather_observed_tokens(tokens, abs_pos, rel_geom, token_obs_mask):
        batch_size, _, dim = tokens.shape
        max_obs = max(1, int(token_obs_mask.sum(dim=1).max().item()))
        memory_tokens = tokens.new_zeros(batch_size, max_obs, dim)
        memory_abs = abs_pos.new_zeros(batch_size, max_obs, abs_pos.shape[-1])
        memory_rel = rel_geom.new_zeros(batch_size, max_obs, rel_geom.shape[-1])
        memory_mask = tokens.new_zeros(batch_size, max_obs)

        for batch_idx in range(batch_size):
            obs_idx = token_obs_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            if obs_idx.numel() == 0:
                memory_tokens[batch_idx, 0] = tokens[batch_idx, 0]
                memory_abs[batch_idx, 0] = abs_pos[batch_idx, 0]
                memory_rel[batch_idx, 0] = rel_geom[batch_idx, 0]
                memory_mask[batch_idx, 0] = 1.0
                continue

            count = obs_idx.numel()
            memory_tokens[batch_idx, :count] = tokens[batch_idx, obs_idx]
            memory_abs[batch_idx, :count] = abs_pos[batch_idx, obs_idx]
            memory_rel[batch_idx, :count] = rel_geom[batch_idx, obs_idx]
            memory_mask[batch_idx, :count] = 1.0

        return memory_tokens, memory_abs, memory_rel, memory_mask

    def _relative_attention_bias(self, geom_q, geom_k, enabled):
        if not (self.use_relative_bias and enabled and self.relative_bias is not None):
            return None
        return self.relative_bias(geom_q, geom_k)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condL=None, log_tau=None,
                time_axis=None, training=False, missing_mask_input=None):
        """
        Args:
            x: (B, 2, H, W), x[:,0:1] is x_t and x[:,1:2] is x_cond.
            t: (B,) flow timestep.
            condL: tuple(rx, ry, sx, sy), each (B, H).
            missing_mask_input: optional (B, H), 1=observed and 0=missing.
        Returns:
            (B, 1, H, W) vector field prediction.
        """
        if x.dim() != 4 or x.shape[1] < 2:
            raise ValueError(f"expect x shape (B, 2, H, W), got {tuple(x.shape)}")

        x_t = x[:, 0:1]
        x_cond = x[:, 1:2]
        batch_size, _, n_traces, _ = x_t.shape

        coords = self._build_abs_coords(condL, batch_size, n_traces, x_t.device, x_t.dtype)
        observed_trace_mask = self._infer_observed_mask(x_cond, missing_mask_input=missing_mask_input)

        x_t_chunked, coords_chunked, time_bounds, chunk_info = trace_time_chunk(
            x_t.squeeze(1),
            coords,
            chunk_length=self.chunk_length,
            overlap_ratio=self.chunk_overlap,
        )
        x_cond_chunked, _, _, _ = trace_time_chunk(
            x_cond.squeeze(1),
            coords,
            chunk_length=self.chunk_length,
            overlap_ratio=self.chunk_overlap,
        )

        n_chunks = chunk_info["n_chunks"]
        token_obs_mask = self._expand_trace_mask(observed_trace_mask, n_chunks).bool()
        abs_pos = self._build_position_ids(coords_chunked, time_bounds)

        trace_geometry = self._build_local_trace_geometry(
            condL,
            observed_trace_mask,
            batch_size,
            n_traces,
            x_t.device,
            x_t.dtype,
        )
        rel_geom = self._build_token_rel_geometry(trace_geometry, n_chunks, time_bounds)

        cond_tokens = self.cond_embed(x_cond_chunked)
        query_data = self.xt_embed(x_t_chunked)
        query_cond = self.cond_query_embed(x_cond_chunked)
        query_tokens = self.query_fuse(torch.cat([query_data, query_cond], dim=-1))

        t_emb = self.time_emb(t).unsqueeze(1)
        query_tokens = query_tokens + t_emb

        cond_tokens = self.memory_norm(cond_tokens)
        query_tokens = self.query_norm(query_tokens)

        memory_tokens, memory_abs, memory_rel, memory_mask = self._gather_observed_tokens(
            cond_tokens,
            abs_pos,
            rel_geom,
            token_obs_mask,
        )

        encoder_bias = self._relative_attention_bias(
            memory_rel,
            memory_rel,
            enabled=self.use_encoder_relative_bias,
        )
        _, memory = self.encoder(
            memory_tokens,
            skip_initial_proj=True,
            position_ids=memory_abs,
            mask=memory_mask,
            attention_bias=encoder_bias,
        )
        if self.memory_bottleneck is not None:
            _, memory = self.memory_bottleneck(
                memory,
                skip_initial_proj=True,
                position_ids=memory_abs,
                mask=memory_mask,
                attention_bias=encoder_bias,
            )

        self_bias = self._relative_attention_bias(
            rel_geom,
            rel_geom,
            enabled=self.use_decoder_self_relative_bias,
        )
        cross_bias = self._relative_attention_bias(
            rel_geom,
            memory_rel,
            enabled=self.use_cross_relative_bias,
        )
        cross_query_gate = None
        if self.cross_query_gate is not None:
            cross_query_gate = self.cross_query_gate(token_obs_mask.long())
        decoded = self.decoder(
            query_tokens,
            memory,
            skip_initial_proj=True,
            memory_mask=memory_mask,
            position_ids=abs_pos,
            memory_position_ids=memory_abs,
            self_attention_bias=self_bias,
            memory_attention_bias=cross_bias,
            conditioning=t_emb.expand(-1, query_tokens.shape[1], -1),
            cross_query_gate=cross_query_gate,
        )

        out_tokens = self.head(decoded)
        out_tokens = out_tokens.view(
            batch_size,
            x_t_chunked.shape[1],
            self.output_channels,
            self.chunk_length,
        )
        out_tokens = out_tokens.squeeze(2)
        out = trace_time_unchunk(out_tokens, chunk_info, overlap_ratio=self.chunk_overlap)
        return out.unsqueeze(1)


def create_gated_seisdit_gen_encdec(
    image_channels=2,
    d_model=576,
    d_ff=2048,
    nhead=8,
    num_encoder_layers=4,
    num_bottleneck_layers=None,
    num_memory_bottleneck_layers=None,
    num_decoder_layers=4,
    dropout=0.1,
    chunk_length=256,
    chunk_overlap=0.5,
    headwise_attn_output_gate=True,
    elementwise_attn_output_gate=False,
    use_qk_norm=True,
    qkv_bias=False,
    use_energy_stats=None,
    use_missing_embed=None,
    use_relative_bias=True,
    use_encoder_relative_bias=False,
    use_decoder_self_relative_bias=False,
    use_cross_relative_bias=True,
    relative_bias_hidden_dim=64,
    relative_bias_init_scale=0.0,
    relative_bias_prior_scale=0.05,
    use_time_adaln=True,
    use_cross_query_gate=True,
    num_attn_res_blocks=2,
    **kwargs,
):
    head_dim = d_model // nhead
    if head_dim % 12 != 0:
        raise ValueError(
            f"6D RoPE requires head_dim={head_dim} (d_model={d_model}//nhead={nhead}) "
            "to be divisible by 12"
        )
    if num_memory_bottleneck_layers is None:
        num_memory_bottleneck_layers = 2 if num_bottleneck_layers is None else num_bottleneck_layers
    return GatedSeisDiTEncDecGen(
        image_channels=image_channels,
        d_model=d_model,
        d_ff=d_ff,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_memory_bottleneck_layers=num_memory_bottleneck_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        chunk_length=chunk_length,
        chunk_overlap=chunk_overlap,
        headwise_attn_output_gate=headwise_attn_output_gate,
        elementwise_attn_output_gate=elementwise_attn_output_gate,
        use_qk_norm=use_qk_norm,
        qkv_bias=qkv_bias,
        use_relative_bias=use_relative_bias,
        use_encoder_relative_bias=use_encoder_relative_bias,
        use_decoder_self_relative_bias=use_decoder_self_relative_bias,
        use_cross_relative_bias=use_cross_relative_bias,
        relative_bias_hidden_dim=relative_bias_hidden_dim,
        relative_bias_init_scale=relative_bias_init_scale,
        relative_bias_prior_scale=relative_bias_prior_scale,
        use_time_adaln=use_time_adaln,
        use_cross_query_gate=use_cross_query_gate,
        num_attn_res_blocks=num_attn_res_blocks,
        **kwargs,
    )


if __name__ == "__main__":
    model = create_gated_seisdit_gen_encdec(
        d_model=576,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        chunk_length=64,
    )
    batch_size, n_traces, time_len = 2, 48, 256
    x_t = torch.randn(batch_size, 1, n_traces, time_len)
    x_cond = torch.randn(batch_size, 1, n_traces, time_len)
    x_cond[:, :, 8:16] = 0.0
    rx = torch.linspace(0.0, 1.0, n_traces).repeat(batch_size, 1)
    ry = torch.zeros(batch_size, n_traces)
    sx = torch.zeros(batch_size, n_traces)
    sy = torch.ones(batch_size, n_traces)
    t = torch.rand(batch_size)
    y = model(torch.cat([x_t, x_cond], dim=1), t, condL=(rx, ry, sx, sy))
    print(f"output shape: {tuple(y.shape)}")
