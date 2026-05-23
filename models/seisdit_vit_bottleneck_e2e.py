import torch

from .seisdit_vit_bottleneck import SeisDiTRope


class SeisDiTRopeE2E(SeisDiTRope):
    """
    E2E variant of SeisDiTRope:
    - keep the same backbone architecture
    - remove external diffusion/flow timestep conditioning
    - train directly from observed data to target data
    """

    def forward(self, x_obs: torch.Tensor, condL=None, training: bool = False):
        """
        Args:
            x_obs: [B, 1, H, W], observed seismic data
            condL: optional geometry tuple (rx, ry, sx, sy)
        Returns:
            [B, 1, H, W]
        """
        if x_obs.dim() != 4 or x_obs.shape[1] != 1:
            raise ValueError(f"x_obs shape must be [B,1,H,W], got {tuple(x_obs.shape)}")

        bsz = x_obs.shape[0]
        t_dummy = torch.zeros(bsz, device=x_obs.device, dtype=x_obs.dtype)

        # Keep original two-branch tokenizer path:
        # x_in branch gets zeros, x_cond branch gets observed data.
        x_in = torch.zeros_like(x_obs)
        x = torch.cat([x_in, x_obs], dim=1)

        return super().forward(
            x=x,
            t=t_dummy,
            condL=condL,
            log_tau=None,
            time_axis=None,
            training=training,
        )
