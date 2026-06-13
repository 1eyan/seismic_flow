import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from inspect import isfunction
from .rope import SegmentedRoPEExpCached
from .fourier_enoder import Seismic5DEncoder

def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def modulate(x, shift, scale):
    #return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    return x * (1 + scale) + shift

def _softplus_inv(target):
    # 求 softplus(x)=target 的 x（即 softplus^{-1}(target)）
    # softplus(x) = ln(1+exp(x)) -> invert: x = ln(exp(target)-1)
    return math.log(math.exp(target) - 1.0 + 1e-12)

def get_cond(rx,ry,sx,sy):
    deltaX = (rx - sx)/2
    deltaY = (ry - sy)/2
    midX   = (rx + sx) / 2
    midY   = (ry + sy) / 2
    offset = torch.sqrt(deltaX**2 + deltaY**2)
    azimuth_rad = torch.arctan2(deltaY, deltaX)
    azimuth_deg = torch.from_numpy((np.degrees(azimuth_rad.cpu().numpy()) + 360.0) % 360.0)
    return deltaX, deltaY, midX, midY, offset,azimuth_deg


def normalize(arr, amin=None, amax=None):
    # linear map to [-1,1]; if amin/amax None, use arr min/max
    if amin is None: amin = torch.min(arr)
    if amax is None: amax = torch.max(arr)
    d = amax - amin if (amax - amin) != 0 else 1.0
    arr=(arr - amin) / d 
    return arr ,amin, amax


class AdaTimeModulation(nn.Module):
    def __init__(self, hidden_dim, time_dim, eps=1e-6):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 6)
        )
        nn.init.zeros_(self.time_proj[-1].weight)
        nn.init.zeros_(self.time_proj[-1].bias)
        self.rms = nn.functional.normalize      # L2 归一化代替 mean/std

    def forward(self, x, t):
        shift_msa, scale_msa, gate_msa,\
        shift_mlp, scale_mlp, gate_mlp = self.time_proj(t).chunk(6, dim=-1)

        def _mod(inp, shift, scale, gate):
            n = self.rms(inp, dim=(2, 3), eps=1e-6)      # HW 两维一起归一化
            return inp + gate[:, :, None, None] * (n * (1 + scale[:, :, None, None]) + shift[:, :, None, None])

        x = _mod(x, shift_msa, scale_msa, gate_msa)
        x = _mod(x, shift_mlp, scale_mlp, gate_mlp)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        d_model: 词向量的维度
        max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        # 创建一个 max_len x d_model 的矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        # 偶数位置：sin，奇数位置：cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引维度

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]，为了广播匹配batch维度
        self.register_buffer('pe', pe)  # 不作为参数更新

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        返回：添加了位置编码后的张量
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=16, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):

        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = MYact()  
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)
        
    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        # 输出维度(batch_size, time_channels)
        return emb

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        QWEN3 
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.shift = nn.Parameter(torch.zeros(hidden_size))
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states + self.shift).to(input_dtype)

class MYact(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.gelu=nn.GELU()
        self.silu=nn.SiLU()
    def forward(self, x):
        return self.silu(x)

class Emb(nn.Module):
    def __init__(self, d_model: int, minLog: float, maxLog: float):
        super().__init__()
        assert d_model % 4 == 0, "d_model 必须能被 4 整除，当前为 {}".format(d_model)

        # 二维坐标用的频率向量：长度 = d_model // 4
        f_pair = torch.exp(torch.linspace(minLog, maxLog, d_model // 4))
        # 标量用的频率向量：长度 = d_model // 2
        f_scalar = torch.exp(torch.linspace(minLog, maxLog, d_model // 2)

        )
        self.register_buffer("f_pair", f_pair, persistent=False)
        self.register_buffer("f_scalar", f_scalar, persistent=False)

        self.linear = nn.Conv2d(
            d_model, d_model,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # 可选的缩放参数，先保留（下面我给出怎么用）
        self.alpha = nn.Parameter(torch.tensor(_softplus_inv(1.0)), requires_grad=True)
        self.beta  = nn.Parameter(torch.tensor(_softplus_inv(1.0)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入有两种合法模式：

        1) 标量模式（scalar mode）
           x: [B, H, W] 或 [B, 1, H, W]
           表示每个空间位置一个标量（例如时间、logτ、某个属性）

        2) 二维坐标模式（pair mode）
           x: [B, H, W, 2]
           表示每个空间位置一个二维向量（例如 (offset, azimuth)）

        输出：
           out: [B, d_model, H, W]
        """
        # ---------- 二维坐标模式：x[...,2] ----------
        if x.dim() == 4 and x.shape[-1] == 2:
            # x: [B, H, W, 2]
            B, H, W, _ = x.shape
            x1 = x[..., 0]        # [B, H, W]
            x2 = x[..., 1]        # [B, H, W]
            x1 = x1.unsqueeze(1)  # [B, 1, H, W]
            x2 = x2.unsqueeze(1)  # [B, 1, H, W]
            #频率 reshape 成 [1, F, 1, 1] 方便 broadcast
            f = self.f_pair.view(1, -1, 1, 1)  # F = d_model//4
            # Fourier features
            x1 = x1 * f * (2 * math.pi)       # [B, F, H, W]
            x2 = x2 * f * (2 * math.pi)
            x1_enc = torch.cat([torch.sin(x1), torch.cos(x1)], dim=1)  # [B, 2F, H, W]
            x2_enc = torch.cat([torch.sin(x2), torch.cos(x2)], dim=1)  # [B, 2F, H, W]
            # 2F + 2F = 4F = d_model
            x_enc = torch.cat([x1_enc, x2_enc], dim=1)                 # [B, d_model, H, W]
        # ---------- 标量模式：x 是一个场 ----------
        else:
            # 允许 x 是 [B, H, W] 或 [B, 1, H, W]
            if x.dim() == 3:
                # [B, H, W] -> [B, 1, H, W]
                x = x.unsqueeze(1)
            elif x.dim() == 4 and x.shape[1] == 1:
                # 已经是 [B,1,H,W] 直接用
                pass
            else:
                raise ValueError(
                    f"标量模式下，x 期望形状为 [B,H,W] 或 [B,1,H,W]，"
                    f"当前为 {x.shape}"
                )
            B, C, H, W = x.shape
            # f_scalar 长度 = d_model//2
            f = self.f_scalar.view(1, -1, 1, 1)  # [1, F, 1, 1]
            x = x * f * (2 * math.pi)          # [B, F, H, W]
            x_enc = torch.cat([torch.sin(x), torch.cos(x)], dim=1)  # [B, 2F, H, W] = [B,d_model,H,W]
        out = self.linear(x_enc)  # [B, d_model, H, W]
        # 如果你想用 alpha / beta 做一个可学习的残差缩放，可以这样打开：
        # a = F.softplus(self.alpha)
        # b = F.softplus(self.beta)
        # out = a * out + b * x_enc     # out 里混合了原始 Fourier feature 和线性变换
        return out
##Encoder
##conv1d for time
class Resblock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.05,
        kernel_size=(1, 7),
        norm_type: str = "instance",
    ):
        super().__init__()
        trace_pad = kernel_size[0] // 2
        time_pad = kernel_size[1] // 2

        if norm_type == "instance":
            self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.norm1 = GroupNorm(num_channels=in_channels)
            self.norm2 = GroupNorm(num_channels=out_channels)

        self.act1 = MYact()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(trace_pad, time_pad),
        )
        
        self.act2 = MYact()
        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(trace_pad, time_pad),
        )
        
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.shortcut = torch.nn.Identity()
            
        self.adaLN = AdaTimeModulation(time_dim=time_channels, hidden_dim=out_channels)
        self.time_emb = torch.nn.Linear(time_channels, out_channels)
        self.time_act = MYact()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # 主路径
        #print(x.shape)
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # 时间调制
        h = h + self.time_emb(self.time_act(t))[:, :, None, None]
        #h=self.adaLN(h,t)
        
        # 第二层处理
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        # 残差连接
        return h + self.shortcut(x)

class Downsample(nn.Module):
    """
    时间维度下采样
    """
    def __init__(self, n_channels, i,stride:int=2):
        super().__init__()
        # 只对时间维度操作
        self.conv_0 = nn.Conv2d(n_channels, n_channels, (1, stride+1), stride=(1, stride), padding=(0, stride//2),)
        self.conv_1 = nn.Conv2d(n_channels, n_channels, (1, stride+1), stride=(1, stride), padding=(0, stride//2),)
        self.conv_2 = nn.Conv2d(n_channels, n_channels, (1, stride+1), stride=(1, stride), padding=(0, stride//2),)
        self.i = i
        self.conv_list = torch.nn.ModuleList([self.conv_0,self.conv_1,self.conv_2])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv_list[self.i](x)

class AntiAliasDownsample(nn.Module):
    """
    沿时间轴 (W) 的反锯齿 1D 下采样。
    两步分离: groups=n_channels 可学习低通滤波 -> AvgPool2d 固定 stride。
    高斯初始化保证初始行为 = 经典 BlurPool。
    接受 (x, t) 签名与 Resblock 保持一致 (t 忽略)。
    """
    def __init__(self, n_channels, stride=2, aa_kernel_size=5, learnable=False):
        super().__init__()
        self.antialias = nn.Conv2d(
            n_channels, n_channels,
            kernel_size=(1, aa_kernel_size),
            padding=(0, aa_kernel_size // 2),
            groups=n_channels, bias=False
        )
        sigma = float(aa_kernel_size) / 6.0
        t_c = torch.arange(aa_kernel_size, dtype=torch.float32) - aa_kernel_size // 2
        g = torch.exp(-0.5 * (t_c / sigma) ** 2)
        g = g / g.sum()
        self.antialias.weight.data.copy_(
            g.view(1, 1, 1, -1).expand(n_channels, 1, 1, -1))
        if not learnable:
            self.antialias.weight.requires_grad_(False)
        self.down = nn.AvgPool2d((1, stride))

    def forward(self, x, t=None):
        return self.down(self.antialias(x))


class WidthUpsample1D(nn.Module):
    def __init__(self, in_channels,out_channels,width_scale):
        super().__init__()
        self.width_scale = width_scale
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * width_scale,  
            kernel_size=(5,3),
            padding=(2,1),
            stride=1,
        )
    def forward(self, x):
        # x: (B, C, H, W)
        #print(x.shape)
        x = self.conv(x)  # (B, C × r, H, W)
        B, C_r, H, W = x.shape
        C = C_r // self.width_scale
        x = x.view(B, C, self.width_scale, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # (B, C, H, W, r)
        x = x.reshape(B, C, H, W * self.width_scale)  # (B, C, H, W × r)
        return x

class WidthUpsample_Block(nn.Module):
    def __init__(self, n_channels, stride=2):
        super().__init__()
        self.upsample = WidthUpsample1D(
            in_channels=n_channels,
            out_channels=n_channels,
            width_scale=stride,
        )
    def forward(self, x):
        # (B, C, H, W) -> (B, C, H, W * stride)
        x_up = self.upsample(x)
       #residual = x_up
        #out = self.conv1(x_up)
        #out = self.act(out)
        #out = self.conv2(out)
        return x_up #+ residual

class Upsample(nn.Module):
    """
    时间维度上采样
    """
    def __init__(self, n_channels, i,stride=2):
        super().__init__()
        ##Note: 这里stride=2,即每次上采样2倍
        self.conv_0 = WidthUpsample_Block(n_channels, stride)
        #self.conv_0 = nn.ConvTranspose2d(n_channels, n_channels, (1, stride*2), stride=(1, stride), padding=(0, stride//2))
        self.conv_1 = WidthUpsample_Block(n_channels, stride)
        #self.conv_1 = nn.ConvTranspose2d(n_channels, n_channels, (1, stride*2), stride=(1, stride), padding=(0, stride//2))
        self.conv_2 = WidthUpsample_Block(n_channels, stride)
        #self.conv_2 = nn.ConvTranspose2d(n_channels, n_channels, (1, stride*2), stride=(1, stride), padding=(0, stride//2))
        self.i = i
        #self.conv_list = torch.nn.ModuleList([self.conv_0, self.conv_1, self.conv_2])
        self.conv_list = torch.nn.ModuleList([self.conv_0,self.conv_1,self.conv_2])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv_list[self.i](x)


class InterpUpsample1D(nn.Module):
    """
    沿时间轴 (W) 的上采样: 双线性插值 + 可学习 refine 卷积。
    dirac 初始化 -> 训练初期 = 纯双线性, 逐步学习高频补偿。
    接受 (x, t) 签名与 Resblock 保持一致 (t 忽略)。
    """
    def __init__(self, n_channels, stride=2):
        super().__init__()
        self.stride = stride
        self.refine = nn.Conv2d(
            n_channels, n_channels,
            kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        nn.init.dirac_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def forward(self, x, t=None):
        x = F.interpolate(x, scale_factor=(1, self.stride),
                          mode='bilinear', align_corners=False)
        return self.refine(x)


# ========== Trace-axis global attention ==========
class TraceAxisAttention2D(nn.Module):
    """
    Trace-axis global attention: 对每个时间位置 w，在 H 维度上做全局多头自注意力。

    位置编码:
    - 默认开启 RoPE（Rotary Positional Embedding），使用 `rope.py::SegmentedRoPEExpCached`
    - RoPE 作用在 q/k 上，沿 trace 维（H）旋转
    - 若 forward 不传 pos，则使用归一化的 trace 索引作为 pos（[0,1]）
    
    输入: x: [B, H, W, C]
    输出: [B, H, W, C]
    
    机制: 对每个 w，将 H 条道作为序列，做全局 MHSA。
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.1,
        qk_norm=True,
        *,
        use_rope: bool = True,
        rope_n_pos: int = 4,
        rope_min_log: float = -12,
        rope_max_log: float = 0,
        rope_mapper: str = "linear",
        rope_hidden: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_rope = bool(use_rope)
        self.rope_dim = None
        self.rope = None
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
           self.q_norm = None
           self.k_norm = None
        if self.use_rope:
            _rope_dim = self.head_dim 
            _rope_dim = (_rope_dim // 2) * 2  
            if _rope_dim < 2:
                self.use_rope = False
            else:
                self.rope_dim = _rope_dim
                self.rope = SegmentedRoPEExpCached(
                    D=self.rope_dim * self.num_heads,
                    N=self.num_heads,
                    n_pos=int(rope_n_pos),
                    min_log=float(rope_min_log),
                    max_log=float(rope_max_log),
                    mapper=str(rope_mapper),
                    hidden=int(rope_hidden),
                )

    @staticmethod
    def _default_trace_pos(B, H, device):
        """
        默认 trace 位置: 归一化到 [0,1] 的索引，shape = [B, H, 1]
        用 float32 生成，后续在 RoPE 内部按 out_dtype 再转换
        """
        if H <= 1:
            pos_1d = torch.zeros((1,), device=device, dtype=torch.float32)
        else:
            pos_1d = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=torch.float32)
        return pos_1d.view(1, H, 1).expand(B, H, 1)

    def forward(self, x, pos=None):
        """
        x: [B, H, W, C]
        返回: [B, H, W, C]

        pos(可选): [B, H] 或 [B, H, rope_n_pos]，对应每条道的“位置/坐标”特征
        """
        B, H, W, C = x.shape
        
        # 对每个时间位置 w，在 H 维度上做注意力
        # reshape: [B, H, W, C] -> [B*W, H, C]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()  # [B, W, H, C]
        x_reshaped = x_reshaped.view(B * W, H, C)  # [B*W, H, C]
        qkv = self.qkv(x_reshaped)  # [B*W, H, 3*C]
        qkv = qkv.view(B * W, H, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # [3, B*W, num_heads, H, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B*W, num_heads, H, head_dim]
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.use_rope and self.rope is not None and self.rope_dim is not None:
            if pos is None:
                pos_in = self._default_trace_pos(B, H, device=x.device)  # [B,H,1]
            else:
                if pos.dim() == 2:
                    pos_in = pos.unsqueeze(-1)
                else:
                    pos_in = pos
                if pos_in.shape[0] != B or pos_in.shape[1] != H:
                    raise RuntimeError(
                        f"TraceAxisAttention2D: pos shape must be [B,H] or [B,H,n_pos], "
                        f"got {tuple(pos.shape)} while x is {tuple(x.shape)}"
                    )
            self.rope.precompute_cos_sin(pos_in, out_dtype=q.dtype, device=q.device)
            cos = self.rope.cos
            sin = self.rope.sin

            rope_dim = self.rope_dim
            half = rope_dim // 2

            # reshape 成 [B, W, heads, H, head_dim]，避免把 W 展开到 batch 导致 cos/sin 复制
            q = q.contiguous().view(B, W, self.num_heads, H, self.head_dim)
            k = k.contiguous().view(B, W, self.num_heads, H, self.head_dim)

            q_tail = q[..., rope_dim:]
            k_tail = k[..., rope_dim:]

            q_rot = q[..., :rope_dim].contiguous().view(B, W, self.num_heads, H, half, 2)
            k_rot = k[..., :rope_dim].contiguous().view(B, W, self.num_heads, H, half, 2)

            q_even, q_odd = q_rot[..., 0], q_rot[..., 1]  # [B,W,heads,H,half]
            k_even, k_odd = k_rot[..., 0], k_rot[..., 1]

            cos_ = cos.unsqueeze(1)  # [B,1,heads,H,half]
            sin_ = sin.unsqueeze(1)  # [B,1,heads,H,half]

            q_even2 = q_even * cos_ - q_odd * sin_
            q_odd2 = q_even * sin_ + q_odd * cos_
            k_even2 = k_even * cos_ - k_odd * sin_
            k_odd2 = k_even * sin_ + k_odd * cos_

            q_rot2 = torch.stack([q_even2, q_odd2], dim=-1).view(B, W, self.num_heads, H, rope_dim)
            k_rot2 = torch.stack([k_even2, k_odd2], dim=-1).view(B, W, self.num_heads, H, rope_dim)

            q = torch.cat([q_rot2, q_tail], dim=-1).view(B * W, self.num_heads, H, self.head_dim).contiguous()
            k = torch.cat([k_rot2, k_tail], dim=-1).view(B * W, self.num_heads, H, self.head_dim).contiguous()
        
        # Scaled dot-product attention
        if hasattr(F, 'scaled_dot_product_attention') and torch.__version__ >= '2.0.0':
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False
            )  # [B*W, num_heads, H, head_dim]
        else:
            # 回退到手写实现
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*W, num_heads, H, H]
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn_output = attn @ v  # [B*W, num_heads, H, head_dim]
        
        # 恢复形状: [B*W, num_heads, H, head_dim] -> [B*W, H, C]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B*W, H, num_heads, head_dim]
        attn_output = attn_output.view(B * W, H, C)  # [B*W, H, C]
        
        # 输出投影
        out = self.proj(attn_output)  # [B*W, H, C]
        out = self.proj_drop(out)
        
        # 恢复原始形状: [B*W, H, C] -> [B, H, W, C]
        out = out.view(B, W, H, C)  # [B, W, H, C]
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, H, W, C]
        
        return out

class TraceAxisAttention2D_gate(nn.Module):
    """
    Trace-axis global attention: 对每个时间位置 w，在 H 维度上做全局多头自注意力。

    位置编码:
    - 默认开启 RoPE（Rotary Positional Embedding），使用 `rope.py::SegmentedRoPEExpCached`
    - RoPE 作用在 q/k 上，沿 trace 维（H）旋转
    - 若 forward 不传 pos，则使用归一化的 trace 索引作为 pos（[0,1]）
    
    输入: x: [B, H, W, C]
    输出: [B, H, W, C]
    
    机制: 对每个 w，将 H 条道作为序列，做全局 MHSA。
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.1,
        qk_norm=True,
        *,
        ##gate cfg
        headwise_attn_output_gate:bool = False,
        elementwise_attn_output_gate:bool = True,
        ##rope cfg
        use_rope: bool = True,
        rope_n_pos: int = 4,
        rope_min_log: float = -12,
        rope_max_log: float = 0,
        rope_mapper: str = "linear",
        rope_hidden: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        #self.qkv = nn.Linear(dim, dim * 3, bias=False)
        ##gate init
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        assert headwise_attn_output_gate or elementwise_attn_output_gate, "at least one of headwise_attn_output_gate or elementwise_attn_output_gate must be True"
        if headwise_attn_output_gate:
            self.q = nn.Linear(dim, dim+self.num_heads, bias=False)
        elif elementwise_attn_output_gate:
            self.q = nn.Linear(dim, dim * 2, bias=False)
        else:
            self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_rope = bool(use_rope)
        self.rope_dim = None
        self.rope = None
        
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
           self.q_norm = None
           self.k_norm = None
        if self.use_rope:
            _rope_dim = self.head_dim 
            _rope_dim = (_rope_dim // 2) * 2  
            if _rope_dim < 2:
                self.use_rope = False
            else:
                self.rope_dim = _rope_dim
                self.rope = SegmentedRoPEExpCached(
                    D=self.rope_dim * self.num_heads,
                    N=self.num_heads,
                    n_pos=int(rope_n_pos),
                    min_log=float(rope_min_log),
                    max_log=float(rope_max_log),
                    mapper=str(rope_mapper),
                    hidden=int(rope_hidden),
                )

    @staticmethod
    def _default_trace_pos(B, H, device):
        """
        默认 trace 位置: 归一化到 [0,1] 的索引，shape = [B, H, 1]
        用 float32 生成，后续在 RoPE 内部按 out_dtype 再转换
        """
        if H <= 1:
            pos_1d = torch.zeros((1,), device=device, dtype=torch.float32)
        else:
            pos_1d = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=torch.float32)
        return pos_1d.view(1, H, 1).expand(B, H, 1)

    def forward(self, x, pos=None,return_weights=False):
        """
        x: [B, H, W, C]
        返回: [B, H, W, C]

        pos(可选): [B, H] 或 [B, H, rope_n_pos]，对应每条道的“位置/坐标”特征
        """
        B, H, W, C = x.shape
        
        # 对每个时间位置 w，在 H 维度上做注意力
        # reshape: [B, H, W, C] -> [B*W, H, C]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()  # [B, W, H, C]
        x_reshaped = x_reshaped.view(B * W, H, C)  # [B*W, H, C]
        q = self.q(x_reshaped)
        k = self.k(x_reshaped)
        k = k.view(B * W, H,self.num_heads, self.head_dim).transpose(1, 2)  # -> [BW, heads, H, head_dim]
        v = self.v(x_reshaped)
        v = v.view(B * W, H,self.num_heads, self.head_dim).transpose(1, 2)  # -> [BW, heads, H, head_dim]
        if self.headwise_attn_output_gate:
            q, gate_score = torch.split(q, [self.head_dim * self.num_heads, self.num_heads], dim=-1)
            gate_score = gate_score.reshape(B * W, H, self.num_heads, 1)
            q = q.reshape(B * W, H, self.num_heads,self.head_dim).transpose(1, 2)
        elif self.elementwise_attn_output_gate:
            q, gate_score = torch.split(q, [self.head_dim * self.num_heads, self.head_dim * self.num_heads], dim=-1)
            gate_score = gate_score.reshape(B * W, H, self.num_heads, self.head_dim)
            q = q.reshape(B * W, H, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            q = q.reshape(B * W, H, self.num_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.use_rope and self.rope is not None and self.rope_dim is not None:
            if pos is None:
                pos_in = self._default_trace_pos(B, H, device=x.device)  # [B,H,1]
            else:
                if pos.dim() == 2:
                    pos_in = pos.unsqueeze(-1)
                else:
                    pos_in = pos
                if pos_in.shape[0] != B or pos_in.shape[1] != H:
                    raise RuntimeError(
                        f"TraceAxisAttention2D: pos shape must be [B,H] or [B,H,n_pos], "
                        f"got {tuple(pos.shape)} while x is {tuple(x.shape)}"
                    )
            self.rope.precompute_cos_sin(pos_in, out_dtype=q.dtype, device=q.device)
            cos = self.rope.cos
            sin = self.rope.sin

            rope_dim = self.rope_dim
            half = rope_dim // 2

            # reshape 成 [B, W, heads, H, head_dim]，避免把 W 展开到 batch 导致 cos/sin 复制
            q = q.contiguous().view(B, W, self.num_heads, H, self.head_dim)
            k = k.contiguous().view(B, W, self.num_heads, H, self.head_dim)

            q_tail = q[..., rope_dim:]
            k_tail = k[..., rope_dim:]

            q_rot = q[..., :rope_dim].contiguous().view(B, W, self.num_heads, H, half, 2)
            k_rot = k[..., :rope_dim].contiguous().view(B, W, self.num_heads, H, half, 2)

            q_even, q_odd = q_rot[..., 0], q_rot[..., 1]  # [B,W,heads,H,half]
            k_even, k_odd = k_rot[..., 0], k_rot[..., 1]

            cos_ = cos.unsqueeze(1)  # [B,1,heads,H,half]
            sin_ = sin.unsqueeze(1)  # [B,1,heads,H,half]

            q_even2 = q_even * cos_ - q_odd * sin_
            q_odd2 = q_even * sin_ + q_odd * cos_
            k_even2 = k_even * cos_ - k_odd * sin_
            k_odd2 = k_even * sin_ + k_odd * cos_

            q_rot2 = torch.stack([q_even2, q_odd2], dim=-1).view(B, W, self.num_heads, H, rope_dim)
            k_rot2 = torch.stack([k_even2, k_odd2], dim=-1).view(B, W, self.num_heads, H, rope_dim)

            q = torch.cat([q_rot2, q_tail], dim=-1).view(B * W, self.num_heads, H, self.head_dim).contiguous()
            k = torch.cat([k_rot2, k_tail], dim=-1).view(B * W, self.num_heads, H, self.head_dim).contiguous()
        
        # Scaled dot-product attention
        if hasattr(F, 'scaled_dot_product_attention') and torch.__version__ >= '2.0.0':
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False 
            )  # [B*W, num_heads, H, head_dim]
        else:
            # 回退到手写实现
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*W, num_heads, H, H]
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn_output = attn @ v  # [B*W, num_heads, H, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B*W, H, num_heads, head_dim]
        if self.headwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)
        elif self.elementwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)
        else:
            attn_output = attn_output
        attn_output = attn_output.view(B * W, H, C)  # [B*W, H, C]
        out = self.proj(attn_output)  # [B*W, H, C]
        out = self.proj_drop(out)
        out = out.view(B, W, H, C)  # [B, W, H, C]
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, H, W, C]
        return out


# ========== Time-axis windowed attention ==========
class TimeAxisAttention1D(nn.Module):
    """
    沿时间轴（W维）做窗口多头自注意力，每个道位置 h 独立。
    使用 RoPE 编码归一化时间位置 [0,1]。
    输入: x: [B, H, W, C]
    输出: [B, H, W, C]
    """
    def __init__(
        self,
        dim,
        window_size=64,
        shift_size=32,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.1,
        qk_norm=True,
        *,
        use_rope: bool = True,
        rope_n_pos: int = 1,
        rope_min_log: float = -12,
        rope_max_log: float = 0,
        rope_mapper: str = "linear",
        rope_hidden: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.use_rope = bool(use_rope)
        self.rope_dim = None
        self.rope = None
        if self.use_rope:
            _rope_dim = self.head_dim
            _rope_dim = (_rope_dim // 2) * 2
            if _rope_dim < 2:
                self.use_rope = False
            else:
                self.rope_dim = _rope_dim
                self.rope = SegmentedRoPEExpCached(
                    D=self.rope_dim * self.num_heads,
                    N=self.num_heads,
                    n_pos=int(rope_n_pos),
                    min_log=float(rope_min_log),
                    max_log=float(rope_max_log),
                    mapper=str(rope_mapper),
                    hidden=int(rope_hidden),
                )

    @staticmethod
    def _default_time_pos(B, W, device):
        if W <= 1:
            pos_1d = torch.zeros((1,), device=device, dtype=torch.float32)
        else:
            pos_1d = torch.linspace(0.0, 1.0, steps=W, device=device, dtype=torch.float32)
        return pos_1d.view(1, W, 1).expand(B, W, 1)

    def _create_shift_mask(self, B, H, W, win, shift, num_win, device):
        if shift == 0:
            return None
        if not (0 < shift < win):
            shift = shift % win
        img_mask = torch.zeros((W,), device=device, dtype=torch.long)
        cnt = 0
        slices = (slice(0, -win), slice(-win, -shift), slice(-shift, None))
        for s in slices:
            img_mask[s] = cnt
            cnt += 1
        mask_win = img_mask.view(num_win, win)
        mask_win = mask_win[:, None, :].expand(num_win, H, win).reshape(num_win * H, win)
        keep = (mask_win.unsqueeze(1) == mask_win.unsqueeze(2))
        keep = keep.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_win * H, win, win)
        attn_mask = (~keep).unsqueeze(1).to(torch.bool).contiguous()
        return attn_mask

    def forward(self, x, pos=None):
        B, H, W, C = x.shape
        win = self.window_size
        shift = self.shift_size
        if W < win:
            win = W
            shift = 0
        elif W % win != 0:
            found = False
            for d in range(win, 1, -1):
                if W % d == 0:
                    win = d
                    found = True
                    break
            if not found:
                win = W
                shift = 0
            shift = shift if shift < win else win // 2
        assert W % win == 0, f"W must be divisible by win: W={W}, win={win}"

        num_win = W // win
        total_win = B * num_win * H

        x_shifted = x
        pos_shifted = pos
        if shift > 0:
            x_shifted = torch.roll(x, shifts=(-shift,), dims=(2,))
            if pos is not None:
                pos_shifted = torch.roll(pos, shifts=(-shift,), dims=(2,))

        x_windows = x_shifted.view(B, H, num_win, win, C)
        x_windows = x_windows.permute(0, 2, 1, 3, 4).reshape(total_win, win, C)

        qkv = self.qkv(x_windows)
        qkv = qkv.view(total_win, win, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        if self.use_rope and self.rope is not None and self.rope_dim is not None:
            if pos_shifted is not None:
                pos_in = pos_shifted
                if pos_in.dim() == 3:
                    pos_in = pos_in.reshape(B, H, num_win, win, -1)
                    pos_in = pos_in.permute(0, 2, 1, 3, 4).reshape(total_win, win, -1)
                elif pos_in.dim() == 4:
                    pos_in = pos_in.permute(0, 2, 1, 3).reshape(total_win, win)
                    pos_in = pos_in.unsqueeze(-1)
            else:
                pos_in = self._default_time_pos(total_win, win, device=x.device)

            self.rope.precompute_cos_sin(pos_in, out_dtype=q.dtype, device=q.device)
            cos = self.rope.cos
            sin = self.rope.sin

            rope_dim = self.rope_dim
            half = rope_dim // 2

            q_tail = q[..., rope_dim:]
            k_tail = k[..., rope_dim:]

            q_rot = q[..., :rope_dim].contiguous().view(total_win, self.num_heads, win, half, 2)
            k_rot = k[..., :rope_dim].contiguous().view(total_win, self.num_heads, win, half, 2)

            q_even, q_odd = q_rot[..., 0], q_rot[..., 1]
            k_even, k_odd = k_rot[..., 0], k_rot[..., 1]

            cos_ = cos
            sin_ = sin

            q_even2 = q_even * cos_ - q_odd * sin_
            q_odd2 = q_even * sin_ + q_odd * cos_
            k_even2 = k_even * cos_ - k_odd * sin_
            k_odd2 = k_even * sin_ + k_odd * cos_

            q_rot2 = torch.stack([q_even2, q_odd2], dim=-1).view(total_win, self.num_heads, win, rope_dim)
            k_rot2 = torch.stack([k_even2, k_odd2], dim=-1).view(total_win, self.num_heads, win, rope_dim)

            q = torch.cat([q_rot2, q_tail], dim=-1).contiguous()
            k = torch.cat([k_rot2, k_tail], dim=-1).contiguous()

        attn_mask = self._create_shift_mask(B, H, W, win, shift, num_win, device=x.device)

        if hasattr(F, 'scaled_dot_product_attention') and torch.__version__ >= '2.0.0':
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn_output = attn @ v

        attn_output = attn_output.transpose(1, 2).contiguous().view(total_win, win, C)
        out = self.proj(attn_output)
        out = self.proj_drop(out)

        out = out.view(B, num_win, H, win, C)
        out = out.permute(0, 2, 1, 3, 4).reshape(B, H, W, C)

        if shift > 0:
            out = torch.roll(out, shifts=(shift,), dims=(2,))

        return out


class DiTBlockTime(nn.Module):
    """
    DiT block with time-axis windowed attention instead of trace-axis attention.
    保持 adaLN-Zero conditioning 逻辑不变。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, time_window_size=64, time_shift_size=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = TimeAxisAttention1D(
            dim=hidden_size,
            num_heads=num_heads,
            window_size=time_window_size,
            shift_size=time_shift_size,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, rope_pos=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x1 = modulate(self.norm1(x), shift_msa.unsqueeze(-2), scale_msa.unsqueeze(-2))
        x = x + gate_msa.unsqueeze(-2) * self.attn(x1, pos=None)
        x = x + gate_mlp.unsqueeze(-2) * self.mlp(modulate(self.norm2(x), shift_mlp.unsqueeze(-2), scale_mlp.unsqueeze(-2)))
        return x


class DiTBlockTrace(nn.Module):
    """
    DiT block with Trace-axis global attention instead of WindowAttention2D.
    保持 adaLN-Zero conditioning 逻辑不变。
    支持通过 rope_pos 参数传递位置信息给 RoPE。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = TraceAxisAttention2D(dim=hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x, c, rope_pos=None):
        """
        x: [B, H, W, C]
        c: [B, H, C] (fourier_emb)
        rope_pos: 可选，[B, H] 或 [B, H, n_pos]，用于 RoPE 的位置编码
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x1 = modulate(self.norm1(x), shift_msa.unsqueeze(-2), scale_msa.unsqueeze(-2))
        x = x + gate_msa.unsqueeze(-2) * self.attn(x1, pos=rope_pos)
        x = x + gate_mlp.unsqueeze(-2) * self.mlp(modulate(self.norm2(x), shift_mlp.unsqueeze(-2), scale_mlp.unsqueeze(-2)))
        return x

class SeisDiT(torch.nn.Module):
    ##adaLN-zero
    def __init__(
        self,
        image_channels,
        n_channels=64,
        channel=[1,2,2,2],
        d_model=512,
        nhead=6,
        dropout=0.1,
        num_layers=12,
        output_channels=1,
        res_blocks=2,
        strides=[2,2,2,1],
        f_dict=None,
        pe_type='transformer',
        #label_dim=5
    ):
        super(SeisDiT, self).__init__()
        # alpha = (2*num_layers)**0.25
        # beta =  (8*num_layers)**(-0.25)
        self.image_channels = image_channels
        self.n_channels= n_channels
        self.channel = channel
        n_res=len(channel)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.num_layers = num_layers

        self.tokenizer = torch.nn.Conv2d(
            image_channels, n_channels, kernel_size=(1, 3), padding=(0, 1),bias=True
        )
        self.mask_adapter_n=torch.nn.Conv2d(
            n_channels, n_channels, kernel_size=(1, 3), padding=(0, 1),bias=True
        )
        self.mask_adapter_d = torch.nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1),bias=True
        )
        self.time_emb = TimeEmbedding(d_model)
        #nn.init.zeros_(self.time_axis_mlp[-1].weight)
        #nn.init.zeros_(self.time_axis_mlp[-1].bias)
        #self.fourier_encoder=fourier_enoder.Seismic5DEncoder(coord_dim=4,max_freq=128,out_dim=d_model,num_bands=32,pe_type = pe_type)
        last_channel = n_channels*channel[-1]*channel[-2]*channel[-3]

        self.to_attn = torch.nn.Conv2d(last_channel, d_model, kernel_size=(1,3), stride=(1,1), padding=(0,1), bias=True)
        self.to_unet = torch.nn.Conv2d(d_model, last_channel, kernel_size=(1,3), stride=(1,1), padding=(0,1), bias=True)
        attenL =[]
        # embTL=[]
        # ========== 修改：使用 DiTBlockTrace 替代 DiTBlock ==========
        for i in range(num_layers):
            attenL.append(DiTBlockTrace(hidden_size=d_model,num_heads=nhead))
            # attenL.append(Attention_Block(d_model=d_model))
        self.attenL = torch.nn.ModuleList(attenL)
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        down = []  
        out_channels = in_channels = n_channels
        for i in range(n_res):
            out_channels = in_channels * channel[i]
            for _ in range(res_blocks): 
                # print(out_channels,in_channels)
                down.append(
                    Resblock(in_channels, out_channels, d_model,)
                )
                in_channels = out_channels
            if i < n_res - 1:  
                down.append(Downsample(in_channels,i,strides[i]))

        self.down = torch.nn.ModuleList(down)
        up = []
        in_channels = out_channels  
        for i in reversed(range(n_res)):
            out_channels = in_channels
            for _ in range(res_blocks):
                up.append(
                    Resblock(in_channels+out_channels, out_channels,d_model,)
                )
            out_channels = in_channels // channel[i]
            up.append(
                Resblock(in_channels+out_channels, out_channels,d_model,)
            )  
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels, i - 1,strides[i-1]))
        self.up = torch.nn.ModuleList(up)
        self.ac=MYact()
        self.norm = torch.nn.GroupNorm(16,in_channels,eps=1e-5)
        self.final = torch.nn.Conv2d(
            in_channels, output_channels, kernel_size=(1, 5), padding=(0, 2)
        )
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,condL=None,log_tau=None,time_axis=None,training=False):
        B,_,_,T = x.shape   
        x_in,x_cond = x[:,0:1],x[:,1:2] 
        mask = torch.all(x_cond == 0, dim=-1, keepdim=True).to(x_cond.dtype)  
        mask = mask.expand(-1, -1, -1, 1)
        x=self.tokenizer(x)
        x= (1-mask)*x+mask*self.mask_adapter_n(x)
        t = self.time_emb(t)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x=self.to_attn(x)
        x = (1-mask)*x+mask*self.mask_adapter_d(x) 
        B,D,H,W=x.shape
        fourier_emb = None
        if condL is not None:
            rx, ry, sx, sy = condL
            rx = rx - rx.mean(dim=-1, keepdim=True)
            ry = ry - ry.mean(dim=-1, keepdim=True)
            sx = sx - sx.mean(dim=-1, keepdim=True)
            sy = sy - sy.mean(dim=-1, keepdim=True)
            pos_emb=torch.stack([rx,ry,sx,sy], dim=-1)
            fourier_emb=self.fourier_encoder(pos_emb)
        if fourier_emb is None:
            dummy_pos_emb = torch.zeros(B, H, 4, device=x.device, dtype=x.dtype)
            fourier_emb = self.fourier_encoder(dummy_pos_emb)
        #fourier_emb =fourier_emb+t.unsqueeze(1)
        fourier_emb =t.unsqueeze(1)
        x=x.permute(0,2,3,1)                        
        #x=x.permute(0,2,3,1)
        for atten in self.attenL:
            x= atten(x,fourier_emb)
        shift, scale = self.adaLN_modulation(fourier_emb).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift.unsqueeze(-2), scale.unsqueeze(-2))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.to_unet(x)
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
        x = self.final(self.ac(self.norm(x)))
        return x


class SeisDiTRope(torch.nn.Module):
    """
    基于 SeisDiT 的网络，使用 RoPE 位置编码。
    与 SeisDiT 的区别：
    - 使用修改后的 DiTBlockTrace，支持传递 rope_pos 参数
    - 在 forward 中从 condL 提取位置信息（rx/ry）作为 RoPE 的位置输入
    """
    def __init__(
        self,
        image_channels,
        n_channels=64,
        channel=[1,2,2,2],
        d_model=512,
        nhead=8,
        dropout=0.1,
        num_layers=12,
        output_channels=1,
        res_blocks=2,
        strides=[2,2,2,1],
        f_dict=None,
        pe_type='transformer',
        missing_focus_adapter: bool = True,
    ):
        super(SeisDiTRope, self).__init__()
        self.image_channels = image_channels
        self.n_channels = n_channels
        self.channel = channel
        n_res = len(channel)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.num_layers = num_layers
        self.missing_focus_adapter = missing_focus_adapter
        #self.fourier_encoder=fourier_enoder.Seismic5DEncoder(coord_dim=4,max_freq=128,out_dim=d_model,num_bands=32,pe_type = pe_type)
        self.tokenizer = torch.nn.Conv2d(
            image_channels//2, n_channels, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.tokenizer_c = torch.nn.Conv2d(image_channels//2, n_channels, (1,3), padding=(0,1), bias=True)
        self.fuse = torch.nn.Conv2d(2*n_channels, n_channels, kernel_size=(1,1), padding=(0,0), bias=True)
        self.mask_adapter_n = torch.nn.Conv2d(
            n_channels, n_channels, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.mask_adapter_d = torch.nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.time_emb = TimeEmbedding(d_model)
        last_channel = n_channels * channel[-1] * channel[-2] * channel[-3]

        self.to_attn = torch.nn.Conv2d(
            last_channel, d_model, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.to_unet = torch.nn.Conv2d(
            d_model, last_channel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.Geomlp = nn.Sequential(
            nn.Linear(4, d_model*2),
            nn.SiLU(),
            nn.Linear(d_model*2, d_model),
        )
        nn.init.zeros_(self.Geomlp[-1].weight)
        nn.init.zeros_(self.Geomlp[-1].bias)
        self.geo_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.geo_gate.weight)
        nn.init.zeros_(self.geo_gate.bias)
        attenL = []
        for i in range(num_layers):
            attenL.append(DiTBlockTrace(hidden_size=d_model, num_heads=nhead))
        self.attenL = torch.nn.ModuleList(attenL)
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_res):
            out_channels = in_channels * channel[i]
            for _ in range(res_blocks):
                down.append(Resblock(in_channels, out_channels, d_model))
                in_channels = out_channels
            if i < n_res - 1:
                down.append(Downsample(in_channels, i, strides[i]))

        self.down = torch.nn.ModuleList(down)
        up = []
        in_channels = out_channels
        for i in reversed(range(n_res)):
            out_channels = in_channels
            for _ in range(res_blocks):
                up.append(Resblock(in_channels + out_channels, out_channels, d_model))
            out_channels = in_channels // channel[i]
            up.append(Resblock(in_channels + out_channels, out_channels, d_model))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels, i - 1, strides[i - 1]))
        self.up = torch.nn.ModuleList(up)
        self.ac = MYact()
        self.norm = torch.nn.GroupNorm(8, in_channels, eps=1e-5)
        self.final = torch.nn.Conv2d(
            in_channels, output_channels, kernel_size=(1, 5), padding=(0, 2)
        )
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condL=None, log_tau=None, time_axis=None, training=False):
        B, _, _, T = x.shape
        x_in, x_cond = x[:, 0:1], x[:, 1:2]
        mask = torch.all(x_cond == 0, dim=-1, keepdim=True).to(x_cond.dtype)
        mask = mask.expand(-1, -1, -1, 1)
        x_in = self.tokenizer(x_in)
        x_cond = self.tokenizer_c(x_cond)
        x = torch.cat([x_in, x_cond], dim=1)
        x = self.fuse(x)
        if self.missing_focus_adapter:
            x = (1 - mask) * x + mask * self.mask_adapter_n(x)
        else:
            x = x + (1 - mask) * self.mask_adapter_n(x)

        t = self.time_emb(t)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x = self.to_attn(x)
        if self.missing_focus_adapter:
            x = (1 - mask) * x + mask * self.mask_adapter_d(x)
        else:
            x = x + (1 - mask) * self.mask_adapter_d(x)
        
        fourier_emb = None
        rope_pos = None
        if condL is not None:
            rx, ry, sx, sy = condL
            rx = rx - rx.mean(dim=-1, keepdim=True)
            ry = ry - ry.mean(dim=-1, keepdim=True)
            sx = sx - sx.mean(dim=-1, keepdim=True)
            sy = sy - sy.mean(dim=-1, keepdim=True)
            pos_emb = torch.stack([rx, ry, sx, sy], dim=-1) # (B, H, 4)
            fourier_emb = self.Geomlp(pos_emb)
            rope_pos = pos_emb
            
        if fourier_emb is None:
            rope_pos = torch.zeros(B, H, 4, device=x.device, dtype=x.dtype)
            fourier_emb = self.Geomlp(rope_pos)
            
        #geo_scale = torch.tanh(self.geo_gate(fourier_emb))
        fourier_emb = t.unsqueeze(1) + self.geo_gate(fourier_emb)*fourier_emb
        
        x = x.permute(0, 2, 3, 1)

        for atten in self.attenL:
            x = atten(x, fourier_emb, rope_pos=rope_pos)
        
        shift, scale = self.adaLN_modulation(fourier_emb).chunk(2, dim=-1)
        
        x = modulate(self.norm_final(x), shift.unsqueeze(-2), scale.unsqueeze(-2))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.to_unet(x)#+h0
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
        x = self.final(self.ac(self.norm(x)))
        return x

class SeisDiTRopeV2(torch.nn.Module):
    """
    基于 SeisDiT 的网络，使用 RoPE 位置编码。
    与 SeisDiT 的区别：
    - 使用修改后的 DiTBlockTrace，支持传递 rope_pos 参数
    - 在 forward 中从 condL 提取位置信息（rx/ry）作为 RoPE 的位置输入
    """
    def __init__(
        self,
        image_channels,
        n_channels=64,
        channel=[1,2,2,2],
        d_model=512,
        nhead=8,
        dropout=0.1,
        num_layers=12,
        output_channels=1,
        res_blocks=2,
        strides=[2,2,2,1],
        mlp_ratio=2.5,
        num_bands=16,
        max_freq=128,
        missing_focus_adapter: bool = True,
    ):
        super(SeisDiTRopeV2, self).__init__()
        self.image_channels = image_channels
        self.n_channels = n_channels
        self.channel = channel
        n_res = len(channel)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.num_layers = num_layers
        self.missing_focus_adapter = missing_focus_adapter

        self.fourier_encoder = Seismic5DEncoder(
            coord_dim=4, num_bands=num_bands, max_freq=max_freq,
            include_input=True, out_dim=d_model, pe_type='log_spaced',
        )

        self.tokenizer = torch.nn.Conv2d(
            image_channels // 2, n_channels, kernel_size=(1, 5), padding=(0, 2), bias=True
        )
        self.tokenizer_c = torch.nn.Conv2d(image_channels // 2, n_channels, (1, 5), padding=(0, 2), bias=True)
        self.fuse = torch.nn.Conv2d(2 * n_channels, n_channels, kernel_size=(1, 1), padding=(0, 0), bias=True)
        self.mask_adapter_n = torch.nn.Conv2d(
            n_channels, n_channels, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.mask_adapter_d = torch.nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.time_emb = TimeEmbedding(d_model)
        last_channel = n_channels * channel[-1] * channel[-2] * channel[-3]

        self.to_attn = torch.nn.Conv2d(
            last_channel, d_model, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.to_unet = torch.nn.Conv2d(
            d_model, last_channel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )

        attenL = []
        for i in range(num_layers):
            if i % 2 == 0:
                attenL.append(DiTBlockTrace(hidden_size=d_model, num_heads=nhead, mlp_ratio=mlp_ratio))
            else:
                attenL.append(DiTBlockTime(hidden_size=d_model, num_heads=nhead, mlp_ratio=mlp_ratio))
        self.attenL = torch.nn.ModuleList(attenL)

        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_res):
            out_channels = in_channels * channel[i]
            for _ in range(res_blocks):
                down.append(Resblock(in_channels, out_channels, d_model, kernel_size=(1, 7)))
                in_channels = out_channels
            if i < n_res - 1:
                down.append(AntiAliasDownsample(in_channels, strides[i]))

        self.down = torch.nn.ModuleList(down)
        up = []
        in_channels = out_channels
        for i in reversed(range(n_res)):
            out_channels = in_channels
            for _ in range(res_blocks):
                up.append(Resblock(in_channels + out_channels, out_channels, d_model, kernel_size=(1, 7)))
            out_channels = in_channels // channel[i]
            up.append(Resblock(in_channels + out_channels, out_channels, d_model, kernel_size=(1, 7)))
            in_channels = out_channels
            if i > 0:
                up.append(InterpUpsample1D(in_channels, strides[i - 1]))
        self.up = torch.nn.ModuleList(up)
        self.ac = MYact()
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.final = torch.nn.Conv2d(
            in_channels, output_channels, kernel_size=(1, 9), padding=(0, 4)
        )
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def load_state_dict(self, state_dict, strict=True):
        """
        兼容旧版 checkpoint: 将 Geomlp.* / geo_gate.* 映射为 fourier_encoder.*
        （旧模型使用 2层MLP+gate，新模型使用 Seismic5DEncoder 多频段 Fourier 编码）
        """
        _state = dict(state_dict)
        has_old_geomlp = any(k.startswith('Geomlp.') for k in _state)
        has_new_fourier = any(k.startswith('fourier_encoder.') for k in _state)
        if has_old_geomlp and not has_new_fourier:
            import warnings
            warnings.warn(
                "Loading old-format checkpoint (Geomlp+geo_gate). "
                "fourier_encoder weights will be random-initialized. "
                "This model must be retrained or fine-tuned for best results."
            )
            for old_key in list(_state.keys()):
                if old_key.startswith('Geomlp.') or old_key.startswith('geo_gate.'):
                    del _state[old_key]
        return super().load_state_dict(_state, strict=strict)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condL=None, log_tau=None, time_axis=None, training=False):
        B, _, _, T = x.shape
        x_in, x_cond = x[:, 0:1], x[:, 1:2]
        mask = torch.all(x_cond == 0, dim=-1, keepdim=True).to(x_cond.dtype)
        mask = mask.expand(-1, -1, -1, 1)
        x_in = self.tokenizer(x_in)
        x_cond = self.tokenizer_c(x_cond)
        x = torch.cat([x_in, x_cond], dim=1)
        x = self.fuse(x)
        if self.missing_focus_adapter:
            x = (1 - mask) * x + mask * self.mask_adapter_n(x)
        else:
            x = x + (1 - mask) * self.mask_adapter_n(x)

        t = self.time_emb(t)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x = self.to_attn(x)
        if self.missing_focus_adapter:
            x = (1 - mask) * x + mask * self.mask_adapter_d(x)
        else:
            x = x + (1 - mask) * self.mask_adapter_d(x)
        if condL is not None:
            rx, ry, sx, sy = condL
            rx = rx - rx.mean(dim=-1, keepdim=True)
            ry = ry - ry.mean(dim=-1, keepdim=True)
            sx = sx - sx.mean(dim=-1, keepdim=True)
            sy = sy - sy.mean(dim=-1, keepdim=True)
            pos_emb = torch.stack([rx, ry, sx, sy], dim=-1)
            fourier_emb = self.fourier_encoder(pos_emb)
            rope_pos = pos_emb
        else:
            _B, _, _H, _ = x.shape
            dummy_pos_emb = torch.zeros(_B, _H, 4, device=x.device, dtype=x.dtype)
            fourier_emb = self.fourier_encoder(dummy_pos_emb)
            rope_pos = dummy_pos_emb

        fourier_emb = t.unsqueeze(1) + fourier_emb

        x = x.permute(0, 2, 3, 1)

        for atten in self.attenL:
            x = atten(x, fourier_emb, rope_pos=rope_pos)

        shift, scale = self.adaLN_modulation(fourier_emb).chunk(2, dim=-1)
        
        x = modulate(self.norm_final(x), shift.unsqueeze(-2), scale.unsqueeze(-2))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.to_unet(x)
        for m in self.up:
            if isinstance(m, InterpUpsample1D):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
        x = self.final(self.ac(self.norm(x)))
        return x

# ========== 测试代码 ==========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ===== 1. 构造模型实例 =====
    model = SeisDiTRopeV2(
        image_channels=2,
        n_channels=64,
        channel=[1,2,2,2],
        d_model=512,
        nhead=8,
        dropout=0.1,
        num_layers=4,
        output_channels=1,
        res_blocks=2,
        strides=[2,2,2,1],
        mlp_ratio=2.5,
        num_bands=16,
    ).to(device)
    
    model.eval()
    
    # ===== 2. 构造测试数据 =====
    B = 2
    C = 2
    H = 128        # trace dimension (must match training)
    W = 256        # time dimension (must be divisible by prod(strides) = 8)
    
    x = torch.randn(B, C, H, W, device=device)
    t = torch.randint(0, 1000, (B,), device=device).float()
    
    rx = torch.randn(B, H, device=device)
    ry = torch.randn(B, H, device=device)
    sx = torch.randn(B, H, device=device)
    sy = torch.randn(B, H, device=device)
    condL = (rx, ry, sx, sy)
    
    # ===== 3. 前向测试 =====
    print("开始前向传播测试...")
    with torch.no_grad():
        y = model(x, t, condL=condL)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    assert y.shape == (B, 1, H, W), \
        f"输出形状不匹配！期望 {(B, 1, H, W)}，实际 {y.shape}"
    assert not torch.isnan(y).any(), "输出包含 NaN 值！"
    assert not torch.isinf(y).any(), "输出包含 Inf 值！"
    
    print("✓ 测试通过！")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  - 模型参数量: {n_params / 1e6:.2f}M")
    print(f"  - 输入形状: {x.shape}")
    print(f"  - 输出形状: {y.shape}")
    print(f"  - 使用 AntiAliasDownsample + InterpUpsample1D")
