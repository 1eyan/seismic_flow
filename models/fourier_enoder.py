import math
from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


##transformer type
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
        x =x.unsqueeze(-1)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)
##gaussian type
class GaussianRFF(nn.Module):
    """
    Gaussian Random Fourier Features encoder.
    输入: coords (B, N, D)
    输出: encoded (B, N, 2 * num_bands [+ D if include_input])
    """
    def __init__(self,
                 coord_dim: int,
                 num_bands: int = 128,
                 sigma: float = 10.0,
                 include_input: bool = True,
                 out_dim: int = 256):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_bands = num_bands
        self.sigma = sigma
        self.include_input = include_input

        # 随机初始化频率矩阵 B ~ N(0, sigma^2 I)
        self.register_buffer("freq_matrix", torch.randn(num_bands, coord_dim) * sigma)

        # 输出投影 MLP
        in_dim = 2 * num_bands
        if include_input:
            in_dim += coord_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, N, D)
        return: (B, N, out_dim)
        """
        B, N, D = coords.shape
        # (B,N,D) @ (D, num_bands) -> (B,N,num_bands)
        proj = torch.matmul(coords, self.freq_matrix.T)  # 点积
        proj = 2 * math.pi * proj

        sin = torch.sin(proj)
        cos = torch.cos(proj)
        ffm = torch.cat([sin, cos], dim=-1)  # (B,N,2*num_bands)

        if self.include_input:
            ffm = torch.cat([coords, ffm], dim=-1)  # (B,N,D+2*num_bands)

        return ffm

##NERF type 最大频率需要选择
def fourier_feature_mapping(x: torch.Tensor,
                            num_bands: int = 40,
                            max_freq: float = 64.0,
                            include_input: bool = True,
                            base: float = 2.0,
                            device: Optional[torch.device] = None,
                            freq_scales: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        x: tensor with shape (..., D), values assumed normalized to [-1, 1] (float).
        num_bands: B, 每个维度的频率数量
        max_freq: 最高频率
        include_input: 是否在输出中保留原始坐标 (x)
        base: 基数，论文/NeRF 常用 base=2 表示 2^k
        device: optional device (默认用 x.device)
    Returns:
        Tensor with shape (..., D * (2 * num_bands) [+ D if include_input])
    """
    ##todo 添加可学习的调节因子
    if device is None:
        device = x.device
    x = x.to(dtype=torch.get_default_dtype(), device=device)
    orig_shape = x.shape  # (..., D)
    D = orig_shape[-1]

    # 生成 log-spaced 频率 2^{start..end}，步数为 num_bands
    if freq_scales is None:
        if max_freq <= 0:
            raise ValueError("max_freq must be > 0")
        # 计算给定 base 下的末端指数 end_exp，使 base**end_exp = max_freq
        # 保持历史行为：start_exp = 0
        end_exp = math.log(max_freq, base)
        freq_scales = torch.logspace(
            0.0, end_exp, steps=num_bands, base=base, device=device, dtype=torch.get_default_dtype()
        )  # (num_bands,)
    else:
        # 确保在正确的 device/dtype 上
        freq_scales = freq_scales.to(device=device, dtype=torch.get_default_dtype())

    xb = x.unsqueeze(-1) * freq_scales.view(*([1] * (x.dim() - 1)), -1)
    xb = 2.0 * math.pi * xb  

    sin = torch.sin(xb)
    cos = torch.cos(xb)
    # 合并 sin/cos -> (..., D, 2*num_bands)
    enc = torch.cat([sin, cos], dim=-1)
    # 展平最后两个维度 -> (..., D * 2 * num_bands)
    enc = enc.view(*orig_shape[:-1], D * 2 * num_bands)

    if include_input:
        enc = torch.cat([x, enc], dim=-1)  # (..., D + D*2*num_bands)
    return enc


class Seismic5DEncoder(nn.Module):
    """
    - 输入 coords: shape (B, D) 或 (B, N, D)  (D=5 通常)
    - 可选地对 token 维度做聚合
    - 输出 shape: (B, out_dim) or (B, N, out_dim) 取决于 aggregate 参数
    """

    def __init__(
        self,
        coord_dim: int = 9,
        num_bands: int = 8,
        num_bans_gauss:int =128,
        max_freq: float = 64.0,
        include_input: bool = True,
        mlp_hidden: Optional[list] = None,
        out_dim: int = 512,
        dropout: float = 0.0,
        activation: nn.Module = nn.SiLU(),
        norm: Optional[str] = "layernorm",
        learnable_freq: bool = False,
        base: float = 2.0,
        pe_type ='transformer',
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [512 // 2, 512 // 2]

        self.coord_dim = coord_dim
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.include_input = include_input
        self.learnable_freq = learnable_freq
        self.base = base

        # 计算 ffm 后的维度
        ffm_dim = coord_dim * (2 * num_bands)
        if include_input:
            ffm_dim += coord_dim
            
        ##选择pe
        self.pe_type = pe_type
        self.pe = PositionalEncoding(out_dim,)
        self.guassian_pe = GaussianRFF(coord_dim,num_bands=num_bans_gauss,sigma=0.1,include_input=include_input, out_dim=out_dim)

        # MLP
        layers = []
        if pe_type == 'transformer':
            in_dim = coord_dim*out_dim
        elif pe_type == 'log_spaced':
            in_dim = ffm_dim
        elif pe_type == 'gaussian':
            in_dim = coord_dim+2*num_bans_gauss
            #print('in_dim',in_dim)
        else:
            raise ValueError("pe_type must be one of 'transformer', 'log_spaced', 'gaussian'")
        for h in mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            if norm == "layernorm":
                layers.append(nn.LayerNorm(h))
            elif norm == "batchnorm":
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        # 最后一层不接激活（作为 embedding）
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, coords: torch.Tensor, aggregate: Literal["mean", "max", "none"] = "none"
    ) -> torch.Tensor:
        """
        coords: (B, D) 或 (B, N, D)
        aggregate:
            - 'mean': 先对 token 维 (N) 做 mean -> 返回 (B, out_dim)
            - 'max':  对 N 做 max -> 返回 (B, out_dim)
            - 'none': 返回每个位置的 embedding (B, N, out_dim)
        """
        if coords.dim() not in (2, 3):
            raise ValueError("coords must be shape (B, D) or (B, N, D)")

        had_token_dim = coords.dim() == 3
        if not had_token_dim:
            coords_proc = coords  # (B, D)
            ffm = fourier_feature_mapping(
                coords_proc,
                num_bands=self.num_bands,
                max_freq=self.max_freq,
                include_input=self.include_input,
                device=coords.device,
                base=self.base,
            )
            # mlp expects (..., ffm_dim) -> output (..., out_dim)
            out = self.mlp(ffm)
            self._last_output = out.detach().cpu()
            if aggregate == "none":
                # no token dim to return
                return out.unsqueeze(1)  # (B, 1, out_dim)
            return out  # (B, out_dim)
        else:
            B, N, D = coords.shape
            if self.pe_type == 'transformer':
                ffms=[]
                for i in range(coords.shape[-1]):
                    pe = self.pe(coords[...,i])
                    ffms.append(pe)
                ffm = torch.cat(ffms,dim=-1)
                ffm = ffm.view(B*N,-1)
            elif self.pe_type == 'log_spaced':
                # flatten batch+tokens for ffm
                coords_flat = coords.view(B * N, D)
                ffm = fourier_feature_mapping(
                    coords_flat,
                    num_bands=self.num_bands,
                    max_freq=self.max_freq,
                    include_input=self.include_input,
                    device=coords.device,
                    base=self.base,
                )  # (B*N, ffm_dim)
            elif self.pe_type == 'gaussian':
                ffm = self.guassian_pe(coords)
                ffm = ffm.view(B,N,-1)
                #print(ffm.shape)
            else:
                raise ValueError("pe_type must be one of 'transformer', 'log_spaced', 'gaussian'")
            out_flat = self.mlp(ffm)  # (B*N, out_dim)
            out = out_flat.view(B, N, -1)  # (B, N, out_dim)
            self._last_output = out.detach().cpu()
            if aggregate == "none":
                return out
            elif aggregate == "mean":
                return out.mean(dim=1)  # (B, out_dim)
            elif aggregate == "max":
                return out.max(dim=1).values  # (B, out_dim)
            else:
                raise ValueError("aggregate must be one of 'mean','max','none'")


class LAPE(nn.Module):
    def __init__(self, in_dim=5, num_bands=10, base=2.0, num_gaussians=16, hidden_dim=64):
        """
        Local feature-Aware Positional Encoding (LAPE)

        Args:
            in_dim: 输入维度 (比如 5D seismic: sx, sy, rx, ry, t)
            num_bands: log-spaced Fourier feature 数量
            base: 对数底数 (通常取 2)
            num_gaussians: 高斯核数量 (控制局部感知能力)
            hidden_dim: 中间维度 (MLP 映射空间)
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_bands = num_bands
        self.base = base
        self.num_gaussians = num_gaussians

        # -------- Fourier log-spaced frequencies --------
        self.freq_bands = base ** torch.linspace(0, num_bands - 1, num_bands)

        # -------- Learnable Gaussian centers & scales --------
        self.centers = nn.Parameter(torch.randn(num_gaussians, in_dim))   # μ_j
        self.scales = nn.Parameter(torch.ones(num_gaussians, in_dim))     # σ_j

        # -------- Linear projections --------
        P_dim = in_dim + 2 * in_dim * num_bands  # Fourier encoding size
        self.proj_p = nn.Linear(P_dim, hidden_dim)
        self.proj_g = nn.Linear(num_gaussians, hidden_dim)

    def fourier_features(self, x):
        """log-spaced Fourier features"""
        freq = self.freq_bands.to(x.device)  # (num_bands,)
        xb = x[..., None] * freq  # (B, N, D, num_bands)
        enc = torch.cat([torch.sin(math.pi * xb), torch.cos(math.pi * xb)], dim=-1)
        return torch.cat([x, enc.view(*x.shape[:-1], -1)], dim=-1)

    def gaussian_features(self, x):
        """Gaussian local features"""
        # x: (B, N, D), centers: (M, D), scales: (M, D)
        diff = (x.unsqueeze(-2) - self.centers) / (self.scales + 1e-6)  # (B,N,M,D)
        dist2 = (diff ** 2).sum(dim=-1)  # (B,N,M)
        return torch.exp(-dist2)  # (B,N,M)

    def forward(self, x):
        """
        Args:
            x: (B, D) or (B, N, D)
        Returns:
            LAPE embedding: (B, N, hidden_dim) or (B, hidden_dim)
        """
        if x.ndim == 2:  # (B,D)
            x = x.unsqueeze(1)  # → (B,1,D)

        P = self.fourier_features(x)     # (B,N,P_dim)
        G = self.gaussian_features(x)    # (B,N,num_gaussians)

        P_proj = self.proj_p(P)          # (B,N,H)
        G_proj = self.proj_g(G)          # (B,N,H)

        out = P_proj * G_proj            # (B,N,H)

        if out.shape[1] == 1:            # 如果输入 (B,D)
            out = out.squeeze(1)         # → (B,H)

        return out
