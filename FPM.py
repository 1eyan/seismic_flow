import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transport import create_transport, Sampler


def spectral_loss_1d(pred, target, alpha=1.0):
    """
    沿时间轴 (W) 的 1D 频域损失, 高频加权。

    Args:
        pred, target: [B, 1, H, W] — 速度场或重建数据
        alpha: 高频强调系数 (0=均匀, 1=Nyquist 处权重翻倍)
    Returns:
        scalar loss
    """
    pred_fft = torch.fft.rfft(pred, dim=-1, norm='ortho')
    target_fft = torch.fft.rfft(target, dim=-1, norm='ortho')
    n_freqs = pred_fft.shape[-1]
    freqs = torch.linspace(0, 1, n_freqs, device=pred.device)
    weight = 1.0 + alpha * freqs
    diff = (pred_fft - target_fft).abs()
    return (weight.view(1, 1, 1, -1) * diff).mean()


class FlowMatchingModel(nn.Module):
    """
    Flow Matching Model wrapper that maintains compatibility with DiffusionModel interface.
    
    This class wraps the transport-based flow matching implementation while maintaining
    the same interface as DiffusionModel for easy migration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        trace_num: int,
        time_steps: int,
        path_type: str = "Linear",
        prediction: str = "velocity",
        loss_weight: str = None,
        train_eps: float = None,
        sample_eps: float = None,
        sample_num: int = 5,
        device=None,
        sup_mode: str = 'all',
        use_coherence: bool = False,
        sigma_obs: float = 1e-3,
        use_bayesian: bool = True,
        # Flow matching specific parameters
        sampling_method: str = "ode",  # "ode" or "sde"
        ode_sampling_method: str = "dopri5",
        ode_num_steps: int = 50,
        ode_atol: float = 1e-6,
        ode_rtol: float = 1e-3,
        sde_sampling_method: str = "Euler",
        sde_num_steps: int = 250,
        sde_diffusion_form: str = "sigma",
        sde_diffusion_norm: float = 1.0,
        sde_last_step: str = "Mean",
        sde_last_step_size: float = 0.04,
        use_multiscale_loss: bool = False,
        multiscale_loss_weight: float = 0.1,
        # Spectral loss parameters
        use_spectral_loss: bool = False,
        spec_weight: float = 0.01,
        spec_alpha: float = 1.0,
    ) -> None:
        """
        Initialize Flow Matching Model.
        
        Args:
            model: The backbone neural network model
            trace_num: Number of traces (spatial dimension)
            time_steps: Number of time steps (temporal dimension)
            path_type: Type of flow path ("Linear", "GVP", "VP")
            prediction: Model prediction type ("velocity", "score", "noise")
            loss_weight: Loss weighting ("velocity", "likelihood", or None)
            train_eps: Small epsilon for training stability
            sample_eps: Small epsilon for sampling stability
            sample_num: Number of samples to generate
            device: Device to run on
            sup_mode: Supervision mode (kept for compatibility)
            use_coherence: Whether to use coherence (kept for compatibility)
            sigma_obs: Observation noise (kept for compatibility)
            use_bayesian: Whether to use Bayesian update (kept for compatibility)
            sampling_method: "ode" or "sde"
            ode_sampling_method: ODE solver method ("dopri5", "euler", etc.)
            ode_num_steps: Number of ODE steps
            ode_atol: ODE absolute tolerance
            ode_rtol: ODE relative tolerance
            sde_sampling_method: SDE solver method ("Euler", "Heun")
            sde_num_steps: Number of SDE steps
            sde_diffusion_form: SDE diffusion form
            sde_diffusion_norm: SDE diffusion norm
            sde_last_step: SDE last step type
            sde_last_step_size: SDE last step size
        """
        super().__init__()
        self.model = model
        self.trace_num = trace_num
        self.time_steps = time_steps
        self.sample_num = sample_num
        self.device = device
        self.sup_mode = sup_mode
        self.use_coherence = use_coherence
        self.sigma_obs = sigma_obs
        self.use_bayesian = use_bayesian
        
        # Flow matching parameters
        self.sampling_method = sampling_method
        self.ode_sampling_method = ode_sampling_method
        self.ode_num_steps = ode_num_steps
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol
        self.sde_sampling_method = sde_sampling_method
        self.sde_num_steps = sde_num_steps
        self.sde_diffusion_form = sde_diffusion_form
        self.sde_diffusion_norm = sde_diffusion_norm
        self.sde_last_step = sde_last_step
        self.sde_last_step_size = sde_last_step_size

        self.use_multiscale_loss = use_multiscale_loss
        self.multiscale_loss_weight = multiscale_loss_weight

        self.use_spectral_loss = use_spectral_loss
        self.spec_weight = spec_weight
        self.spec_alpha = spec_alpha
        self.loss_weight_type = loss_weight
        
        # Create transport object
        self.transport = create_transport(
            path_type=path_type,
            prediction=prediction,
            loss_weight=loss_weight,
            train_eps=train_eps,
            sample_eps=sample_eps,
        )
        
        # Create sampler
        self.sampler = Sampler(self.transport)
        
        # Time normalization: Flow matching uses [0, 1], but model might expect [0, timesteps]
        # We'll normalize time in the model wrapper function
        self.time_normalize = True  # Whether to normalize time to [0, timesteps] for model compatibility
        
    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalize time from [0, 1] (flow matching) to [0, timesteps] (model expectation).
        If model already expects [0, 1], set time_normalize=False.
        """
        if self.time_normalize:
            # Flow matching uses [0, 1], but model expects [0, timesteps]
            return t.float() * self.time_steps
        return t.float()
    
    def _model_wrapper(self, x: torch.Tensor, t: torch.Tensor, **model_kwargs):
        """
        Wrapper function to call the model with proper time normalization and x_cond handling.
        
        Args:
            x: Input tensor [B, C, H, W] or [B, 2*C, H, W] if x_cond is concatenated
            t: Time tensor [B] in [0, 1] range
            **model_kwargs: Additional arguments including condL, x_cond, time_axis
        """
        # Extract x_cond if present
        x_cond = model_kwargs.pop('x_cond', None)
        condL = model_kwargs.pop('condL', None)
        time_axis = model_kwargs.pop('time_axis', None)
        
        # If x_cond is provided, it should already be concatenated in the input x
        # But we need to check: if x has 2 channels and x_cond is None, split it
        # Otherwise, concatenate if needed
        if x_cond is not None:
            # Concatenate x and x_cond along channel dimension
            model_in = torch.cat([x, x_cond], dim=1)
        else:
            model_in = x
        
        # Normalize time to [0, timesteps] if model expects it
        # For now, we assume model can handle [0, 1] range
        # If not, we can multiply by time_steps here
        t_normalized = self._normalize_time(t)
        
        # Call model
        output = self.model(model_in, t_normalized, condL=condL, time_axis=time_axis)
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        condL: tuple = None,
        x_cond: torch.Tensor = None,
        time: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for training. Computes the flow matching loss.

        Args:
            x: Ground truth data [B, C, H, W]
            condL: Conditional information tuple (rx, ry, sx, sy)
            x_cond: Conditional input [B, C, H, W] (e.g., masked data)
            time: Time tensor [B] (optional, kept for compatibility, not used in flow matching)
            loss_mask: Per-trace loss mask [B, H] or None. 1=compute loss, 0=ignore.
                       When None, loss is computed over all positions (backward compatible).

        Returns:
            loss: Training loss scalar tensor
        """
        # Store x_cond and condL for use in wrapped model
        # We need to capture these in closure
        x_cond_captured = x_cond
        condL_captured = condL
        
        # Create a wrapped model that handles x_cond concatenation
        def wrapped_model(xt, t, **kwargs):
            # xt is the interpolated sample at time t [B, C, H, W]
            # We need to concatenate x_cond if provided
            if x_cond_captured is not None:
                # x_cond should have same batch size as xt
                # Handle batch size mismatch
                if x_cond_captured.shape[0] != xt.shape[0]:
                    if x_cond_captured.shape[0] == 1:
                        x_cond_batched = x_cond_captured.expand(xt.shape[0], -1, -1, -1)
                    elif x_cond_captured.shape[0] > xt.shape[0]:
                        x_cond_batched = x_cond_captured[:xt.shape[0]]
                    else:
                        # Repeat x_cond to match batch size
                        repeat_times = (xt.shape[0] + x_cond_captured.shape[0] - 1) // x_cond_captured.shape[0]
                        x_cond_batched = x_cond_captured.repeat(repeat_times, 1, 1, 1)[:xt.shape[0]]
                else:
                    x_cond_batched = x_cond_captured
                model_in = torch.cat([xt, x_cond_batched], dim=1)
            else:
                model_in = xt
            
            # Normalize time: flow matching uses [0, 1], but model might expect [0, timesteps]
            t_normalized = self._normalize_time(t)
            
            # Call original model
            return self.model(model_in, t_normalized, condL=condL_captured, time_axis=kwargs.get('time_axis', None))
        
        # Prepare model kwargs (these are passed to wrapped_model via **kwargs)
        model_kwargs = {}
        
        # Compute loss using transport (loss_mask passed separately, not via model_kwargs)
        loss_dict = self.transport.training_losses(wrapped_model, x, model_kwargs, loss_mask=loss_mask)
        loss = loss_dict['loss'].mean()

        if self.loss_weight_type == 'logitnormal':
            t_val = loss_dict['t']
            w_t = (t_val * (1.0 - t_val) + 1e-4).detach()

        if self.use_multiscale_loss:
            pred = loss_dict['pred']
            ut = loss_dict['ut']
            pred_d2 = torch.nn.functional.avg_pool2d(pred, kernel_size=2)
            ut_d2 = torch.nn.functional.avg_pool2d(ut, kernel_size=2)
            pred_d4 = torch.nn.functional.avg_pool2d(pred, kernel_size=4)
            ut_d4 = torch.nn.functional.avg_pool2d(ut, kernel_size=4)
            ms_loss = 0.5 * torch.nn.functional.mse_loss(pred_d2, ut_d2) + \
                      0.2 * torch.nn.functional.mse_loss(pred_d4, ut_d4)
            if self.loss_weight_type == 'logitnormal':
                ms_loss = ms_loss * w_t.mean()
            loss = loss + self.multiscale_loss_weight * ms_loss

        if self.use_spectral_loss:
            pred = loss_dict['pred']
            ut = loss_dict['ut']
            spec_loss = spectral_loss_1d(pred, ut, alpha=self.spec_alpha)
            if self.loss_weight_type == 'logitnormal':
                spec_loss = spec_loss * w_t.mean()
            loss = loss + self.spec_weight * spec_loss

        return loss
    
    @torch.inference_mode()
    def sample(
        self,
        condL: tuple = None,
        x_cond: torch.Tensor = None,
        x_mask: torch.Tensor = None,
        x_known: torch.Tensor = None,
        time_axis: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Sample from the flow matching model.
        
        Args:
            condL: Conditional information tuple (rx, ry, sx, sy)
            x_cond: Conditional input [B, C, H, W] (e.g., masked data)
            x_mask: Mask tensor (kept for compatibility, not used in flow matching)
            x_known: Known values tensor (kept for compatibility, not used in flow matching)
            time_axis: Time axis information (optional)
            
        Returns:
            samples: Generated samples [sample_num, C, H, W]
        """
        # Store for closure
        x_cond_captured = x_cond
        condL_captured = condL
        time_axis_captured = time_axis
        
        # Create wrapped model for sampling
        def wrapped_model(xt, t, **kwargs):
            # Concatenate x_cond if provided
            if x_cond_captured is not None:
                # Handle batch size mismatch
                if x_cond_captured.shape[0] != xt.shape[0]:
                    if x_cond_captured.shape[0] == 1:
                        x_cond_batched = x_cond_captured.expand(xt.shape[0], -1, -1, -1)
                    elif x_cond_captured.shape[0] > xt.shape[0]:
                        x_cond_batched = x_cond_captured[:xt.shape[0]]
                    else:
                        # Repeat x_cond to match batch size
                        repeat_times = (xt.shape[0] + x_cond_captured.shape[0] - 1) // x_cond_captured.shape[0]
                        x_cond_batched = x_cond_captured.repeat(repeat_times, 1, 1, 1)[:xt.shape[0]]
                else:
                    x_cond_batched = x_cond_captured
                model_in = torch.cat([xt, x_cond_batched], dim=1)
            else:
                model_in = xt
            
            # Normalize time: flow matching uses [0, 1], but model might expect [0, timesteps]
            t_normalized = self._normalize_time(t)
            
            # Call original model
            return self.model(model_in, t_normalized, condL=condL_captured, time_axis=time_axis_captured)
        
        # Prepare model kwargs (these are passed to wrapped_model via **kwargs)
        model_kwargs = {}
        
        # Initialize noise — derive shape from x_cond when available so one FPM
        # instance can handle gathers of different trace counts.
        if x_cond is not None:
            shape = x_cond.shape  # [B, 1, trace_num, time_steps]
        else:
            shape = (self.sample_num, 1, self.trace_num, self.time_steps)
        x0 = torch.randn(shape, device=self.device)
        
        # Sample using ODE or SDE
        if self.sampling_method == "ode":
            sample_fn = self.sampler.sample_ode(
                sampling_method=self.ode_sampling_method,
                num_steps=self.ode_num_steps,
                atol=self.ode_atol,
                rtol=self.ode_rtol,
                reverse=False,
            )
            # sample_fn returns a list of samples at different time steps
            samples = sample_fn(x0, wrapped_model, **model_kwargs)
            # Return the final sample (last element)
            return samples[-1]
        elif self.sampling_method == "sde":
            sample_fn = self.sampler.sample_sde(
                sampling_method=self.sde_sampling_method,
                diffusion_form=self.sde_diffusion_form,
                diffusion_norm=self.sde_diffusion_norm,
                last_step=self.sde_last_step,
                last_step_size=self.sde_last_step_size,
                num_steps=self.sde_num_steps,
            )
            # sample_fn returns a list of samples
            samples = sample_fn(x0, wrapped_model, **model_kwargs)
            # Return the final sample (last element)
            return samples[-1]
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
    
    def parameters(self):
        """Return model parameters for optimizer."""
        return self.model.parameters()
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self
