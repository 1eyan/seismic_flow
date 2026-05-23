#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地震数据Transformer模型 V9 - Block Attention Residuals 插值网络（纯网络部分）

基于 QWEN 论文的 Gated Attention 机制 + Kimi Attention Residuals (Block 模式)，
不包含 chunk/unchunk。
输入: (B, seq_len, input_dim) 已切块数据 + coords + time_bounds
输出: (B, seq_len, input_dim)

切块与重建由调用方在外层完成，可使用本模块提供的 trace_time_chunk / trace_time_unchunk。

V9 相对 V5 的核心改进:
- Encoder 支持 Block Attention Residuals (use_attn_res=True)
- 默认 use_attn_res=False，行为与 V5 完全一致（向后兼容）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Qwen3 style)"""
    def __init__(self, hidden_size, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size) 或 (batch_size, hidden_size)
        # 计算 RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并应用可学习权重
        return (x / rms) * self.weight


class RMSNormNoWeight(nn.Module):
    """无参 RMS Normalization（Attention Residuals 的 key 归一化用）

    参考: Kimi Team, "Attention Residuals", arXiv:2603.15031
    使用 parameter-free RMSNorm 防止深层 magnitude 过大导致后期层主导注意力。
    """
    def __init__(self, eps=1e-6):
        super(RMSNormNoWeight, self).__init__()
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms


class AttnResOperator(nn.Module):
    """Attention Residuals 核心算子

    将固定深度的累加替换为基于 softmax attention 的动态选择。
    参考: Kimi Team, "Attention Residuals", arXiv:2603.15031

    关键设计:
    - zero-initialized pseudo-query: 初始化时 softmax 退化为均匀权重（安全启动）
    - parameter-free RMSNorm on keys: 防止 magnitude 差异导致的不公平竞争
    """
    def __init__(self, d_model, eps=1e-6):
        super(AttnResOperator, self).__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNormNoWeight(eps=eps)

    def forward(self, sources):
        """
        Args:
            sources: (N_src, batch_size, seq_len, d_model) - 之前所有 block 的聚合表示
        Returns:
            output: (batch_size, seq_len, d_model) - 动态聚合后的当前层输入
        """
        # K: (N_src, B, T, d)
        K = self.key_norm(sources)
        # logits: (N_src, B, T)
        logits = torch.einsum("d,nbtd->nbt", self.pseudo_query, K)
        # weights: (N_src, B, T), softmax over depth dimension (dim=0)
        weights = F.softmax(logits, dim=0)
        # output: (B, T, d)
        output = torch.einsum("nbt,nbtd->btd", weights, sources)
        return output


def trace_time_chunk(
    x, coords, chunk_length, overlap_ratio=0.0
):
    """
    将每个 trace 沿时间轴切成多个重叠/非重叠的时间段。

    Args:
        x: 地震数据 (batch_size, num_traces, time_length)
        coords: 四维坐标 (batch_size, num_traces, 4)
        chunk_length: 每段时间长度
        overlap_ratio: 重叠率 [0, 1)，0 表示无重叠

    Returns:
        x_chunked: (batch_size, num_traces * num_chunks, chunk_length)
        coords_chunked: (batch_size, num_traces * num_chunks, 4)
        time_bounds: (batch_size, num_traces * num_chunks, 2) - [start_idx, end_idx] 每段的首尾时间索引
        chunk_info: dict with n_chunks, step, for unchunking
    """
    B, N_trace, T_time = x.shape
    step = max(1, int(chunk_length * (1.0 - overlap_ratio)))
    n_chunks = max(1, (T_time - chunk_length) // step + 1)

    chunks_list = []
    coords_list = []
    bounds_list = []

    last_end = -1
    for c in range(n_chunks):
        start = c * step
        end = start + chunk_length
        if end > T_time:
            # 最后一段若超出，则从末尾对齐
            end = T_time
            start = max(0, end - chunk_length)
        chunk = x[:, :, start:end]  # (B, N_trace, chunk_len)
        if chunk.shape[2] < chunk_length:
            chunk = F.pad(chunk, (0, chunk_length - chunk.shape[2]))
        chunks_list.append(chunk)
        coords_list.append(coords)
        bounds = torch.zeros(B, N_trace, 2, dtype=torch.float32, device=x.device)
        bounds[:, :, 0] = start
        bounds[:, :, 1] = end - 1  # 末尾索引（含）
        bounds_list.append(bounds)
        last_end = end

    # 若最后一块未覆盖到 T_time，补一块末尾对齐的
    if last_end < T_time:
        start = max(0, T_time - chunk_length)
        end = T_time
        chunk = x[:, :, start:end]
        if chunk.shape[2] < chunk_length:
            chunk = F.pad(chunk, (0, chunk_length - chunk.shape[2]))
        chunks_list.append(chunk)
        coords_list.append(coords)
        bounds = torch.zeros(B, N_trace, 2, dtype=torch.float32, device=x.device)
        bounds[:, :, 0] = start
        bounds[:, :, 1] = end - 1
        bounds_list.append(bounds)
        n_chunks += 1

    # (B, N_trace * n_chunks, chunk_length)
    x_chunked = torch.cat(chunks_list, dim=1)
    coords_chunked = torch.cat(coords_list, dim=1)  # (B, N_trace * n_chunks, 4)
    time_bounds = torch.cat(bounds_list, dim=1)     # (B, N_trace * n_chunks, 2)

    chunk_info = {
        "n_chunks": n_chunks,
        "step": step,
        "chunk_length": chunk_length,
        "n_traces": N_trace,
        "time_length": T_time,
    }
    return x_chunked, coords_chunked, time_bounds, chunk_info


def trace_time_unchunk(x_chunked, chunk_info, overlap_ratio=0.0):
    """
    将切块后的输出重建回原始形状，重叠区域做平均。

    Args:
        x_chunked: (batch_size, num_traces * num_chunks, chunk_length)
        chunk_info: 来自 trace_time_chunk 的 dict

    Returns:
        x: (batch_size, num_traces, time_length)
    """
    B, tokens, chunk_len = x_chunked.shape
    n_chunks = chunk_info["n_chunks"]
    step = chunk_info["step"]
    n_traces = chunk_info["n_traces"]
    T_time = chunk_info["time_length"]

    out = torch.zeros(B, n_traces, T_time, dtype=x_chunked.dtype, device=x_chunked.device)
    count = torch.zeros(B, n_traces, T_time, dtype=torch.float32, device=x_chunked.device)

    for c in range(n_chunks):
        start = c * step
        end = start + chunk_len
        if end > T_time:
            end = T_time
            start = max(0, end - chunk_len)  # 与 chunk 一致的末尾对齐
        seg_len = end - start
        idx = c * n_traces
        seg = x_chunked[:, idx : idx + n_traces, :seg_len]
        out[:, :, start:end] = out[:, :, start:end] + seg
        count[:, :, start:end] = count[:, :, start:end] + 1.0

    count = count.clamp(min=1.0)
    out = out / count
    return out


def get_norm_layer(norm_type, d_model, eps=1e-8):
    """获取归一化层

    Args:
        norm_type: 归一化类型，'rms' 或 'layer'
        d_model: 模型维度
        eps: 数值稳定性参数

    Returns:
        归一化层模块
    """
    if norm_type.lower() == 'rms':
        return RMSNorm(d_model, eps=eps)
    elif norm_type.lower() == 'layer' or norm_type.lower() == 'layernorm':
        return nn.LayerNorm(d_model, eps=eps)
    else:
        raise ValueError(f"不支持的归一化类型: {norm_type}。请使用 'rms' 或 'layer'")


class AbsoluteCoordinateEncoding(nn.Module):
    """基于正余弦的绝对坐标编码器 - 支持 2/4/6 维坐标

    6D: [source_x, source_y, receiver_x, receiver_y, time_start_norm, time_end_norm]
    使用标准 Transformer 风格的 sincos 编码：
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    对于归一化坐标 [0,1]，自动映射到 [0, 10000] 范围
    """
    def __init__(self, d_model, coord_dim=4, max_freq=1.0):
        super(AbsoluteCoordinateEncoding, self).__init__()
        self.d_model = d_model
        self.coord_dim = coord_dim  # 4维坐标: [source_x, source_y, receiver_x, receiver_y]
        self.max_freq = max_freq  # 保留参数以兼容旧接口，但不再使用
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # 根据 coord_dim 分配每维编码长度，dim_per_coord 需为偶数（sin/cos 成对）
        self.dim_per_coord = max(2, (d_model // coord_dim) // 2 * 2)
        encoded_dim = coord_dim * self.dim_per_coord
        self.proj = nn.Linear(encoded_dim, d_model) if encoded_dim != d_model else None

        # 为每个坐标维度预计算频率
        # 频率: 10000^(2i/dim_per_coord)，其中 i 是维度索引
        # 对于归一化坐标[0,1]，我们使用标准的Transformer频率计算
        self.freq_bands = []
        for dim in range(coord_dim):
            dim_t = torch.arange(0, self.dim_per_coord // 2, dtype=torch.float32)
            # 标准 Transformer 频率: 10000^(2i/d_model)
            # 这个频率设计用于位置索引（0, 1, 2, ...），对于归一化坐标映射后的位置也适用
            freqs = 10000.0 ** (2.0 * dim_t / self.dim_per_coord)
            self.freq_bands.append(freqs)

    def _encode_1d_coord(self, coord_values, freqs):
        """
        对单个坐标维度进行1D sincos编码

        Args:
            coord_values: (batch_size, seq_len, 1) 坐标值
            freqs: (num_freqs,) 频率数组

        Returns:
            encoded: (batch_size, seq_len, dim_per_coord) 编码结果
        """
        batch_size, seq_len, _ = coord_values.shape

        # 如果坐标在 [0,1] 范围内，假设是归一化坐标
        # sin/cos的输出范围是[-1, 1]，所以位置编码值本身不会很大
        # 但我们需要确保角度范围合理，使得sin/cos能够充分变化
        coord_max = coord_values.max().item()
        coord_min = coord_values.min().item()
        if coord_max <= 1.0 + 1e-6 and coord_min >= -1e-6:
            # 对于归一化坐标[0,1]，映射到合理的位置范围
            # 使用1000作为最大位置，这样角度范围大约是[0, 1000/freq_min]
            # 对于最小频率1，最大角度是1000，sin/cos可以充分变化
            max_pos = 1000.0
            pos = coord_values * max_pos  # (batch_size, seq_len, 1) 范围 [0, 1000]
        else:
            # 否则认为是原始位置索引，直接使用
            pos = coord_values

        # 计算角度: pos / freq
        # pos: (batch_size, seq_len, 1), freqs: (num_freqs,)
        # 广播后: (batch_size, seq_len, num_freqs)
        # 添加小的epsilon避免除零
        angles = pos / (freqs.unsqueeze(0).unsqueeze(0) + 1e-8)

        # 生成位置编码: [sin, cos, sin, cos, ...]
        emb = torch.zeros(batch_size, seq_len, self.dim_per_coord,
                         dtype=coord_values.dtype, device=coord_values.device)
        emb[:, :, 0::2] = torch.sin(angles)  # 偶数位置: sin
        emb[:, :, 1::2] = torch.cos(angles)  # 奇数位置: cos

        return emb

    def _encode_coordinates(self, coords):
        """
        使用标准 Transformer 风格的 sincos 编码坐标
        输入: coords (batch_size, seq_len, coord_dim) - coord_dim=4: [source_x, source_y, receiver_x, receiver_y]
        输出: encoded (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, coord_dim = coords.shape
        encoded_list = []

        for dim in range(coord_dim):
            # 获取该维度的坐标
            coord_values = coords[:, :, dim:dim+1]  # (batch_size, seq_len, 1)

            # 获取该维度的频率
            freqs = self.freq_bands[dim].to(coords.device)  # (num_freqs,)

            # 对单个维度进行编码
            dim_encoded = self._encode_1d_coord(coord_values, freqs)  # (batch_size, seq_len, dim_per_coord)

            encoded_list.append(dim_encoded)

        # 拼接所有维度的编码
        encoded = torch.cat(encoded_list, dim=-1)  # (batch_size, seq_len, coord_dim * dim_per_coord)
        if self.proj is not None:
            encoded = self.proj(encoded)  # -> (batch_size, seq_len, d_model)
        return encoded

    def forward(self, x, coords):
        """
        输入:
            x: 地震数据 (batch_size, seq_len, d_model)
            coords: 坐标 (batch_size, seq_len, coord_dim)，coord_dim=4 或 6
                    - 4D: [source_x, source_y, receiver_x, receiver_y]
                    - 6D: [source_x, source_y, receiver_x, receiver_y, time_start, time_end]（后两维需已归一化）
        输出:
            encoded: 编码后的数据 (batch_size, seq_len, d_model)
        """
        coord_encoded = self._encode_coordinates(coords)
        encoded = x + coord_encoded * self.gamma
        return encoded


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[RoPE 辅助] 将维度对 (x0,x1,x2,x3,...) 变为 (-x1,x0,-x3,x2,...)。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


# =============================================================================
# 6D RoPE（用于空间与时序坐标 coords）
# =============================================================================


def apply_rotary_pos_emb_nd(
    q: torch.Tensor,
    k: torch.Tensor,
    coords: torch.Tensor,
    coord_dim: int = 6,
    base: float = 10000.0,
) -> tuple:
    """
    [RoPE 4D/6D] 对 Query 和 Key 施加基于多维坐标的旋转位置编码。
    将 head_dim 均分为 coord_dim 份，每份用对应坐标维度的值作为「位置」施加旋转。
    支持 4D (sx, sy, rx, ry) 或 6D (sx, sy, rx, ry, start_time, end_time)。
    坐标值需归一化到 [0, 1]，内部会映射到合理的位置范围。

    Args:
        q: Query，(batch, num_heads, seq_len, head_dim)
        k: Key，形状同上
        coords: 坐标 (batch, seq_len, coord_dim)，值在 [0, 1]
        coord_dim: 坐标维度，4 或 6
        base: 频率基数

    Returns:
        (q_rotated, k_rotated)
    """
    B, H, L, D = q.shape
    if D % coord_dim != 0:
        raise ValueError(f"head_dim={D} 必须能被 coord_dim={coord_dim} 整除")
    if (D // coord_dim) % 2 != 0:
        raise ValueError(f"每个坐标维度的 head 子维度 {D//coord_dim} 必须为偶数")

    part_dim = D // coord_dim
    coords = coords.to(q.dtype).to(q.device)
    max_coord = 1000.0
    pos = coords * max_coord

    q_parts, k_parts = [], []
    for i in range(coord_dim):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, part_dim, 2, dtype=torch.float32, device=q.device) / part_dim)
        )
        angles = torch.einsum("bl,d->bld", pos[..., i], inv_freq)
        emb = torch.cat([angles, angles], dim=-1)
        cos_i = emb.cos()
        sin_i = emb.sin()

        q_i = q[:, :, :, i * part_dim : (i + 1) * part_dim]
        k_i = k[:, :, :, i * part_dim : (i + 1) * part_dim]
        cos_i = cos_i.unsqueeze(1)
        sin_i = sin_i.unsqueeze(1)
        q_i_rot = (q_i * cos_i) + (_rotate_half(q_i) * sin_i)
        k_i_rot = (k_i * cos_i) + (_rotate_half(k_i) * sin_i)
        q_parts.append(q_i_rot)
        k_parts.append(k_i_rot)

    return torch.cat(q_parts, dim=-1), torch.cat(k_parts, dim=-1)


class RotaryPositionEmbeddingND(nn.Module):
    """
    [RoPE 4D/6D] 基于多维坐标的旋转位置编码。
    支持 4D (sx, sy, rx, ry) 或 6D (sx, sy, rx, ry, start_time, end_time)。
    coords 在 forward 时传入，根据坐标实时计算旋转并作用于 Q、K。
    """

    def __init__(self, head_dim: int, coord_dim: int = 6, base: float = 10000.0):
        super().__init__()
        if head_dim % coord_dim != 0 or (head_dim // coord_dim) % 2 != 0:
            raise ValueError(
                f"head_dim={head_dim} 必须能被 coord_dim={coord_dim} 整除，"
                f"且每份 {head_dim//coord_dim} 为偶数"
            )
        self.head_dim = head_dim
        self.coord_dim = coord_dim
        self.base = base

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple:
        """
        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_heads, seq_len, head_dim)
            coords: (batch, seq_len, coord_dim)，与序列逐位置对应
        """
        return apply_rotary_pos_emb_nd(q, k, coords, self.coord_dim, self.base)


# =============================================================================
# 标量 / 序列索引 RoPE（position_ids 为 None 或 (B, L)）
# =============================================================================


class RotaryPositionEmbedding(nn.Module):
    """RoPE（标量/序列索引）
    支持 position_ids=None（用 0..L-1）或 position_ids (B, L) 标量位置。
    6D 坐标请用 apply_rotary_pos_emb_nd / RotaryPositionEmbeddingND。
    """
    def __init__(self, head_dim, max_seq_len=8192, base=10000):
        super(RotaryPositionEmbedding, self).__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids=None, seq_len=None):
        """
        Args:
            x: 输入 tensor，用于获取 device
            position_ids: None 或 (batch, seq_len) 标量位置
            seq_len: 序列长度（position_ids 为 None 时使用）
        Returns:
            cos, sin: (seq_len, head_dim) 或 (batch, seq_len, head_dim)
        """
        if seq_len is None:
            seq_len = x.shape[1]
        if position_ids is None:
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
        else:
            t = position_ids.float()  # (B, L)
            freqs = torch.einsum('bl,d->bld', t, self.inv_freq)
            freqs = torch.cat([freqs, freqs], dim=-1)
            return freqs.cos(), freqs.sin()
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin):
        """对 Q、K 应用 RoPE 旋转

        Args:
            q, k: (batch, num_heads, seq_len, head_dim)
            cos, sin: (seq_len, head_dim) 或 (batch, seq_len, head_dim)
        """
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos.unsqueeze(1)   # (B, 1, L, head_dim)
            sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def apply_rotary_single(query, cos, sin):
        """对单个张量应用 RoPE 旋转（用于 cross-attention 中 Q/K 位置不同时）

        Args:
            query: (batch, num_heads, seq_len, head_dim)
            cos, sin: (seq_len, head_dim) 或 (batch, seq_len, head_dim)
        """
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos.unsqueeze(1)   # (B, 1, L, head_dim)
            sin = sin.unsqueeze(1)
        return (query * cos) + (_rotate_half(query) * sin)


class TimeSegmentEncoding(nn.Module):
    """对每段的首尾时间索引 (start, end) 进行 sincos 编码，输出 d_model 维"""

    def __init__(self, d_model, max_time=10000):
        super(TimeSegmentEncoding, self).__init__()
        self.d_model = d_model
        self.max_time = max_time
        assert d_model % 2 == 0, "d_model 必须为偶数"
        self.dim_per_val = d_model // 2
        self.gamma = nn.Parameter(torch.tensor(0.1))

        dim_t = torch.arange(0, self.dim_per_val // 2, dtype=torch.float32)
        self.register_buffer("freqs_start", 10000.0 ** (2.0 * dim_t / self.dim_per_val))
        self.register_buffer("freqs_end", 10000.0 ** (2.0 * dim_t / self.dim_per_val))

    def _encode_1d(self, vals, freqs):
        B, L, _ = vals.shape
        pos = vals / (self.max_time + 1e-8)
        angles = pos / (freqs.unsqueeze(0).unsqueeze(0) + 1e-8)
        emb = torch.zeros(B, L, self.dim_per_val, dtype=vals.dtype, device=vals.device)
        emb[:, :, 0::2] = torch.sin(angles)
        emb[:, :, 1::2] = torch.cos(angles)
        return emb

    def forward(self, x, time_bounds, max_time=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            time_bounds: (batch_size, seq_len, 2) - [start_idx, end_idx]
            max_time: 用于归一化的最大时间索引（默认用 time_bounds 最大值）
        """
        B, L, _ = time_bounds.shape
        if max_time is None:
            max_time = time_bounds.max().item() + 1.0
        bounds_norm = time_bounds.float() / max_time

        start_enc = self._encode_1d(bounds_norm[:, :, 0:1], self.freqs_start)
        end_enc = self._encode_1d(bounds_norm[:, :, 1:2], self.freqs_end)
        time_encoded = torch.cat([start_enc, end_enc], dim=-1)
        return x + time_encoded * self.gamma


class GatedMultiHeadAttention(nn.Module):
    """带门控的多头注意力机制（基于 QWEN3 标准库实现）

    支持两种门控方式：
    - headwise_attn_output_gate: 每个注意力头一个门控值
    - elementwise_attn_output_gate: 每个元素一个门控值

    支持 RoPE 位置编码（对 Q、K 施加旋转）
    """
    def __init__(self, d_model, n_heads, dropout=0.1,
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False,
                 qkv_bias=False,
                 rms_norm_eps=1e-8,
                 use_rope=True):
        super(GatedMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.hidden_size = d_model
        self.num_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = dropout
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate

        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim)
            self.rope_nd = RotaryPositionEmbeddingND(self.head_dim, coord_dim=6)
        else:
            self.rope = None
            self.rope_nd = None

        # 根据门控类型设置 q_proj 的输出维度
        if self.headwise_attn_output_gate:
            # 每个头一个门控值: d_model + n_heads
            q_proj_out_dim = d_model + n_heads
        elif self.elementwise_attn_output_gate:
            # 每个元素一个门控值: d_model * 2
            q_proj_out_dim = d_model * 2
        else:
            # 无门控: d_model
            q_proj_out_dim = d_model

        self.q_proj = nn.Linear(d_model, q_proj_out_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        # QK normalization (可选)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, query, key, value, mask=None, position_ids=None, key_position_ids=None, missing_mask=None, attention_bias=None):
        batch_size, seq_len, _ = query.size()

        # 投影并提取 query, key, value
        query_states = self.q_proj(query)
        key_states = self.k_proj(key)
        value_states = self.v_proj(value)

        # 处理门控分数（如果启用）
        if self.headwise_attn_output_gate:
            # q_proj 输出: (batch_size, seq_len, num_heads * head_dim + num_heads)
            # 先 reshape 成 (batch_size, seq_len, num_heads, head_dim + 1)
            query_states = query_states.view(batch_size, seq_len, self.num_heads, -1)
            query_states, gate_score = torch.split(
                query_states,
                [self.head_dim, 1],
                dim=-1
            )
            gate_score = gate_score.reshape(batch_size, seq_len, self.num_heads, 1)
            query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        elif self.elementwise_attn_output_gate:
            # q_proj 输出: (batch_size, seq_len, num_heads * head_dim * 2)
            # 先 reshape 成 (batch_size, seq_len, num_heads, head_dim * 2)
            query_states = query_states.view(batch_size, seq_len, self.num_heads, -1)
            query_states, gate_score = torch.split(
                query_states,
                [self.head_dim, self.head_dim],
                dim=-1
            )
            gate_score = gate_score.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            gate_score = None

        key_states = key_states.view(batch_size, key.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, value.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # QK normalization (可选)
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # RoPE 位置编码：6D 坐标用 RotaryPositionEmbeddingND，否则用标量/序列 RoPE
        if self.rope is not None:
            if position_ids is not None and position_ids.dim() == 3:
                # 6D coordinate RoPE
                if key_position_ids is not None:
                    query_states, _ = self.rope_nd(query_states, query_states, position_ids)
                    _, key_states = self.rope_nd(key_states, key_states, key_position_ids)
                else:
                    query_states, key_states = self.rope_nd(query_states, key_states, position_ids)
            else:
                # Scalar / sequence RoPE
                cos_q, sin_q = self.rope(query_states, position_ids=position_ids, seq_len=seq_len)
                if key_position_ids is not None:
                    kv_len = key_states.shape[2]
                    cos_k, sin_k = self.rope(key_states, position_ids=key_position_ids, seq_len=kv_len)
                    query_states = RotaryPositionEmbedding.apply_rotary_single(query_states, cos_q, sin_q)
                    key_states = RotaryPositionEmbedding.apply_rotary_single(key_states, cos_k, sin_k)
                else:
                    query_states, key_states = RotaryPositionEmbedding.apply_rotary_pos_emb(
                        query_states, key_states, cos_q, sin_q
                    )

        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用 mask
        if mask is not None:
            # 允许 [B, L] 或 [B, 1, 1, L] 或 [B, L, L]
            if mask.dim() == 2:         # [B, L]
                mask = mask[:, None, None, :]       # -> [B, 1, 1, L]
            elif mask.dim() == 3:       # [B, L, L]
                mask = mask[:, None, :, :]          # -> [B, 1, L, L]
            # attention_scores: [B, H, L, L]，让 mask 可广播
            neg = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(mask == 0, neg)

        if attention_bias is not None:
            attn_weights = attn_weights + attention_bias

        # Softmax 归一化
        # 先进行数值稳定性处理：减去最大值避免溢出
        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, value_states)

        # 转置并重塑
        attn_output = attn_output.transpose(1, 2).contiguous()

        # 应用门控（如果启用）
        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            # gate_score shape: (batch_size, seq_len, num_heads, ...)
            attn_output = attn_output * torch.sigmoid(gate_score)

        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # 输出投影
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen3MLP(nn.Module):
    """Qwen3 风格的 MLP（前馈网络）

    使用门控机制: down_proj(act_fn(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, hidden_size, intermediate_size, dropout=0.1, hidden_act='gelu'):
        super(Qwen3MLP, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # 激活函数映射
        act_fn_map = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
        }
        self.act_fn = act_fn_map.get(hidden_act.lower(), nn.GELU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state):
        # Qwen3 MLP: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        gate_output = self.act_fn(self.gate_proj(hidden_state))
        up_output = self.up_proj(hidden_state)
        output = self.down_proj(gate_output * up_output)
        return self.dropout(output)


class FeedForward(nn.Module):
    """前馈网络 - 兼容旧接口，内部使用 Qwen3MLP"""
    def __init__(self, d_model, d_ff, dropout=0.1, hidden_act='gelu'):
        super(FeedForward, self).__init__()
        self.mlp = Qwen3MLP(d_model, d_ff, dropout, hidden_act)

    def forward(self, x):
        return self.mlp(x)


class GatedTransformerEncoderBlock(nn.Module):
    """带Gated Attention的Transformer编码器块 - True Pre-LN + 残差

    支持 Qwen3 风格的配置：
    - headwise_attn_output_gate: 每个注意力头一个门控值
    - elementwise_attn_output_gate: 每个元素一个门控值
    - use_qk_norm: 使用 QK normalization
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False,
                 qkv_bias=False,
                 norm_type='rms',
                 rms_norm_eps=1e-8,
                 hidden_act='gelu',
                 use_rope=True):
        super().__init__()
        self.norm1 = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)
        self.attn = GatedMultiHeadAttention(
            d_model, n_heads, dropout,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps,
            use_rope=use_rope
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)
        self.ffn = FeedForward(d_model, d_ff, dropout, hidden_act=hidden_act)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, position_ids=None, missing_mask=None, attention_bias=None):
        # Pre-LN: 先归一化，再子层，再残差；子层后不再归一化
        x_norm = self.norm1(x)
        x = x + self.drop1(self.attn(x_norm, x_norm, x_norm, mask, position_ids=position_ids, missing_mask=missing_mask, attention_bias=attention_bias))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class BlockAttnResEncoderBlock(nn.Module):
    """Block Attention Residuals 编码器块

    将标准残差连接替换为 AttnResOperator 动态聚合。
    层内仍然保留 Pre-LN + 残差，确保梯度稳定。

    参考: Kimi Team, "Attention Residuals", arXiv:2603.15031
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False,
                 qkv_bias=False,
                 norm_type='rms',
                 rms_norm_eps=1e-8,
                 hidden_act='gelu',
                 use_rope=True):
        super().__init__()
        self.attn_res = AttnResOperator(d_model, eps=rms_norm_eps)
        self.norm1 = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)
        self.attn = GatedMultiHeadAttention(
            d_model, n_heads, dropout,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            rms_norm_eps=rms_norm_eps,
            use_rope=use_rope
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = get_norm_layer(norm_type, d_model, eps=rms_norm_eps)
        self.ffn = FeedForward(d_model, d_ff, dropout, hidden_act=hidden_act)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, sources, mask=None, position_ids=None, missing_mask=None, attention_bias=None):
        """
        Args:
            sources: (N_src, batch_size, seq_len, d_model) - 之前所有 block 的聚合表示
            mask, position_ids, missing_mask, attention_bias: 与 GatedMultiHeadAttention 一致
        Returns:
            out: (batch_size, seq_len, d_model) - 当前层输出
        """
        # 动态聚合之前所有 block 的表示
        h = self.attn_res(sources)  # (B, T, d)

        # 层内仍然保留 Pre-LN + 残差（确保深层梯度流动）
        h = h + self.drop1(self.attn(self.norm1(h), self.norm1(h), self.norm1(h), mask, position_ids=position_ids, missing_mask=missing_mask, attention_bias=attention_bias))
        out = h + self.drop2(self.ffn(self.norm2(h)))
        return out


class SimpleGatedEncoder(nn.Module):
    """单阶段 Gated 编码器：仅堆叠 GatedTransformerEncoderBlock，无下采样和通道扩展

    V9 新增: 支持 Block Attention Residuals (use_attn_res=True)
    """

    def __init__(self, input_dim, embed_dim=1024, num_layers=4,
                 num_heads=16, d_ff=2048, dropout=0.1, norm_type='rms',
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False, qkv_bias=False,
                 rms_norm_eps=1e-8, hidden_act='gelu',
                 use_rope=True,
                 use_attn_res=False,
                 num_attn_res_blocks=2):
        super().__init__()
        self.initial_proj = nn.Linear(input_dim, embed_dim)
        self.initial_norm = get_norm_layer(norm_type, embed_dim, eps=rms_norm_eps)
        self.num_layers = num_layers
        self.use_attn_res = use_attn_res
        self.num_attn_res_blocks = num_attn_res_blocks

        if self.use_attn_res:
            self.layers = nn.ModuleList([
                BlockAttnResEncoderBlock(
                    embed_dim, num_heads, d_ff, dropout,
                    headwise_attn_output_gate=headwise_attn_output_gate,
                    elementwise_attn_output_gate=elementwise_attn_output_gate,
                    use_qk_norm=use_qk_norm, qkv_bias=qkv_bias,
                    norm_type=norm_type, rms_norm_eps=rms_norm_eps,
                    hidden_act=hidden_act,
                    use_rope=use_rope
                )
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                GatedTransformerEncoderBlock(
                    embed_dim, num_heads, d_ff, dropout,
                    headwise_attn_output_gate=headwise_attn_output_gate,
                    elementwise_attn_output_gate=elementwise_attn_output_gate,
                    use_qk_norm=use_qk_norm, qkv_bias=qkv_bias,
                    norm_type=norm_type, rms_norm_eps=rms_norm_eps,
                    hidden_act=hidden_act,
                    use_rope=use_rope
                )
                for _ in range(num_layers)
            ])
        self.final_norm = get_norm_layer(norm_type, embed_dim, eps=rms_norm_eps)

    def forward(self, x, skip_initial_proj=False, position_ids=None, mask=None, missing_mask=None, attention_bias=None):
        if not skip_initial_proj:
            x = self.initial_proj(x)
            x = self.initial_norm(x)

        if not self.use_attn_res:
            # 标准前向传播（与 V5 完全一致）
            for layer in self.layers:
                x = layer(x, position_ids=position_ids, mask=mask, missing_mask=missing_mask, attention_bias=attention_bias)
            x = self.final_norm(x)
            return [x], x
        else:
            # Block Attention Residuals 前向传播
            # blocks[0] = embedding (经过 initial_proj 和 initial_norm)
            blocks = [x]
            partial_block = None

            for i, layer in enumerate(self.layers):
                source_list = blocks.copy()
                if partial_block is not None:
                    source_list.append(partial_block)

                # sources: (N_src, B, T, d)
                sources = torch.stack(source_list, dim=0)
                out = layer(sources, position_ids=position_ids, mask=mask, missing_mask=missing_mask, attention_bias=attention_bias)  # (B, T, d)

                # Block 内累加子层输出
                if partial_block is None:
                    partial_block = out
                else:
                    partial_block = partial_block + out

                # 到达 block 边界，封存当前 block
                if (i + 1) % self.num_attn_res_blocks == 0:
                    blocks.append(partial_block)
                    partial_block = None

            # 处理剩余的 partial_block（当 num_layers 不是 num_attn_res_blocks 整数倍时）
            if partial_block is not None:
                blocks.append(partial_block)

            # blocks[-1] 是最后一个 block 的聚合输出
            x = self.final_norm(blocks[-1])
            return [x], x


class SimpleGatedDecoder(nn.Module):
    """单阶段 Gated 解码器：仅堆叠 GatedTransformerEncoderBlock + 输出投影，无上采样和通道扩展"""

    def __init__(self, input_dim, embed_dim=1024, num_layers=4,
                 num_heads=16, d_ff=2048, dropout=0.1, norm_type='rms',
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False, qkv_bias=False,
                 rms_norm_eps=1e-8, hidden_act='gelu',
                 use_rope=True):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedTransformerEncoderBlock(
                embed_dim, num_heads, d_ff, dropout,
                headwise_attn_output_gate=headwise_attn_output_gate,
                elementwise_attn_output_gate=elementwise_attn_output_gate,
                use_qk_norm=use_qk_norm, qkv_bias=qkv_bias,
                norm_type=norm_type, rms_norm_eps=rms_norm_eps,
                hidden_act=hidden_act,
                use_rope=use_rope
            )
            for _ in range(num_layers)
        ])
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, input_dim),
            get_norm_layer(norm_type, input_dim, eps=rms_norm_eps)
        )

    def forward(self, x, skip_features=None, position_ids=None):
        for layer in self.layers:
            x = layer(x, position_ids=position_ids)
        return self.final_proj(x)



class GatedSeismicInterpolationTransformerV9(nn.Module):
    """Gated 地震插值 Transformer V9（纯网络部分）

    V9 核心改进（相对 V5）:
    - Encoder 支持 Block Attention Residuals (Kimi, arXiv:2603.15031)
    - 默认 use_attn_res=False，行为与 V5 完全一致

    不包含 chunk/unchunk，调用方需在外层完成切块与重建。

    输入: x (B, seq_len, input_dim), coords (B, seq_len, 4) 空间坐标, time_bounds (B, seq_len, 2) 起止时间
    输出: (B, seq_len, input_dim)

    其中 seq_len = num_traces * num_chunks，input_dim = 每段时间切片长度
    """
    def __init__(self, input_dim, d_model=512, n_heads=8, num_layers=4, d_ff=2048,
                 dropout=0.1, output_dim=None, norm_type='rms',
                 headwise_attn_output_gate=False,
                 elementwise_attn_output_gate=False,
                 use_qk_norm=False,
                 qkv_bias=False,
                 rms_norm_eps=1e-8,
                 hidden_act='gelu',
                 use_coord_encoding=True,
                 use_rope=True,
                 coord_dim=6,
                 coord_max_freq=1.0,
                 use_attn_res=False,
                 num_attn_res_blocks=2):
        super(GatedSeismicInterpolationTransformerV9, self).__init__()
        self.d_model = d_model
        self.norm_type = norm_type
        self.use_coord_encoding = use_coord_encoding  # 6D: 4 空间 + 2 时间，加在输入上
        self.use_rope = use_rope
        self.input_dim = input_dim

        if self.use_coord_encoding:
            self.coord_encoding = AbsoluteCoordinateEncoding(
                d_model, coord_dim=coord_dim, max_freq=coord_max_freq
            )
        if not use_rope:
            self.time_segment_encoding = TimeSegmentEncoding(d_model)

        self.encoder = SimpleGatedEncoder(
            input_dim=input_dim,
            embed_dim=d_model,
            num_layers=num_layers,
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
            use_attn_res=use_attn_res,
            num_attn_res_blocks=num_attn_res_blocks
        )

        self.decoder = SimpleGatedDecoder(
            input_dim=input_dim,
            embed_dim=d_model,
            num_layers=num_layers,
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
            use_rope=use_rope
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
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            pass
        elif isinstance(m, nn.Parameter):
            if m.data.abs().max() > 100:
                m.data.clamp_(-100, 100)

    def forward(self, x, coords=None, time_bounds=None, mask=None):
        """
        前向传播（不含 chunk/unchunk，由调用方在外层处理）
        输入:
            x: (batch_size, seq_len, input_dim) - 已切块的数据，seq_len = num_traces * num_chunks
            coords: (batch_size, seq_len, 4)，值需归一化到 [0, 1]
            time_bounds: (batch_size, seq_len, 2) 起止时间，需归一化到 [0, 1]（6D RoPE 用）
        输出:
            (batch_size, seq_len, input_dim)
        """
        B, seq_len, _ = x.shape
        if coords is None:
            coords = torch.zeros(B, seq_len, 4, dtype=x.dtype, device=x.device)
        if time_bounds is None:
            time_bounds = torch.zeros(B, seq_len, 2, dtype=torch.float32, device=x.device)

        # time_bounds 应由调用方归一化到 [0, 1]（训练脚本中 build_masked_dataset 已处理）
        time_norm = time_bounds.float().clamp(0.0, 1.0)

        x = self.encoder.initial_proj(x)
        x = self.encoder.initial_norm(x)
        coords_6d = None
        if self.use_coord_encoding:
            coords_6d = torch.cat([coords.float(), time_norm], dim=-1)  # (B, L, 6)
            x = self.coord_encoding(x, coords_6d)
        if not self.use_rope:
            max_time = time_bounds.max().item() + 1.0
            x = self.time_segment_encoding(x, time_bounds, max_time=max_time)
        x = self.dropout(x)

        # 6D RoPE: 传入 coords_6d (B, L, 6)，RotaryPositionEmbedding 分段调制频率
        position_ids = coords_6d if (self.use_rope and coords_6d is not None) else None

        features, latent = self.encoder(x, skip_initial_proj=True, position_ids=position_ids)
        x = self.decoder(latent, features, position_ids=position_ids)
        return x


def create_gated_model_v9(input_dim, d_model=512, n_heads=8, num_layers=4, d_ff=2048,
                         dropout=0.1, output_dim=None, norm_type='rms',
                         headwise_attn_output_gate=False,
                         elementwise_attn_output_gate=False,
                         use_qk_norm=False,
                         qkv_bias=False,
                         rms_norm_eps=1e-8,
                         hidden_act='gelu',
                         use_coord_encoding=True,
                         use_rope=True,
                         coord_dim=6,
                         coord_max_freq=1.0,
                         use_attn_res=False,
                         num_attn_res_blocks=2):
    """创建 V9 模型（纯网络，不含 chunk/unchunk）

    input_dim: 每段 token 的特征维度（即 time_slice/chunk_length）
    6D RoPE 要求 head_dim (=d_model//n_heads) 能被 coord_dim 整除且每份为偶数。

    V9 新增参数:
        use_attn_res: 是否启用 Block Attention Residuals（默认 False，向后兼容）
        num_attn_res_blocks: 每个 block 包含多少层（默认 2）
    """
    if output_dim is None:
        output_dim = input_dim
    if use_rope and coord_dim == 6:
        head_dim = d_model // n_heads
        if head_dim % coord_dim != 0 or (head_dim // coord_dim) % 2 != 0:
            raise ValueError(
                f"6D RoPE 要求 head_dim={head_dim} (d_model={d_model}//n_heads={n_heads}) "
                f"能被 {coord_dim} 整除且每份 {head_dim//coord_dim} 为偶数"
            )
    return GatedSeismicInterpolationTransformerV9(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
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
        use_attn_res=use_attn_res,
        num_attn_res_blocks=num_attn_res_blocks
    )


if __name__ == "__main__":
    # 简单测试 V9 模型（纯网络，模拟已切块数据）
    print("=== 测试 Gated Transformer V9（纯网络部分）===")

    chunk_length = 128
    num_traces = 201
    n_chunks = 5
    seq_len = num_traces * n_chunks

    # 测试1: 默认行为（use_attn_res=False），应与 V5 完全一致
    model_base = create_gated_model_v9(
        input_dim=chunk_length,
        d_model=512,
        n_heads=8,
        num_layers=4,
        use_attn_res=False
    )

    # 测试2: Block Attention Residuals（use_attn_res=True）
    model_attn_res = create_gated_model_v9(
        input_dim=chunk_length,
        d_model=512,
        n_heads=8,
        num_layers=4,
        use_attn_res=True,
        num_attn_res_blocks=2
    )

    batch_size = 2
    x = torch.randn(batch_size, seq_len, chunk_length)
    coords = torch.rand(batch_size, seq_len, 4)
    time_bounds = torch.zeros(batch_size, seq_len, 2)
    for c in range(n_chunks):
        start = c * 64
        time_bounds[:, c * num_traces:(c + 1) * num_traces, 0] = start
        time_bounds[:, c * num_traces:(c + 1) * num_traces, 1] = start + chunk_length - 1

    output_base = model_base(x, coords, time_bounds)
    output_attn_res = model_attn_res(x, coords, time_bounds)

    print(f"输入形状: {x.shape}")
    print(f"坐标形状: {coords.shape}")
    print(f"时间边界形状: {time_bounds.shape}")
    print(f"基线输出形状: {output_base.shape}")
    print(f"AttnRes 输出形状: {output_attn_res.shape}")
    print(f"基线模型参数量: {sum(p.numel() for p in model_base.parameters()):,}")
    print(f"AttnRes 模型参数量: {sum(p.numel() for p in model_attn_res.parameters()):,}")
    print(f"AttnRes 额外参数: {sum(p.numel() for p in model_attn_res.parameters()) - sum(p.numel() for p in model_base.parameters()):,}")
