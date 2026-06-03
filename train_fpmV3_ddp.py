from datetime import datetime, timedelta
import math
import os
import pathlib
import random
from typing import BinaryIO, List, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from dataset import config as ds_config
# Flow Matching Model
from FPM import FlowMatchingModel
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.seisdit_trace_axis import SeisDiTRopeV2
from models.seisdit_vit_bottleneck import SeisDiTRope
from models.gated_seisdit_gen import create_gated_seisdit
from models.gated_seisdit_gen_encdec import create_gated_seisdit_gen_encdec
import json
import matplotlib.pyplot as plt
#import dataset.datasets as datasets
from accelerate import Accelerator
import gc
import warnings
import dataset.datasets_interp_v2 as datasets_interp
from segy_schema import get_segy_profile

warnings.filterwarnings(
    "ignore", category=UserWarning, message="The dataloader does not have many workers."
)
# GPU 由 accelerate launch 或运行脚本的 CUDA_VISIBLE_DEVICES 控制，不要在此硬编码
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Accelerate 会自动处理 NCCL 配置
def normalize_clip(data):
    threshold = np.percentile(np.abs(data), 99.5)
    data = np.clip(data, -threshold, threshold)
    data = data / threshold
    return data


RAW_DATA_PATH = (
    "/home/chengzhitong/seismic_ddpm/data/006_3a3_nucns_3a2_data_DX004_p2.sgy"
)
SIM_DATA_PATH = (
    "/home/czt/seismic_ddpm/data/data_25000_111_noS_cs_smooth5_5d_half_halfPT.h5"
)
SIM_DATA_PATH = "/NAS/data/data/Syn_seisData/Data/data5d_19_smooth5_Half_Half.h5"
SIM_DATA_PATH = (
    "/home/czt/seismic_ddpm/data/data_25000_111_noS_cs_smooth5_5d_half_halfPT.h5"
)
SIM_DATA_PATH = "/NAS/data/data/SeismicData/Marmousi/data5d_6_smooth5_Half_Half.h5"
# SIM_DATA_PATH='/NAS/data/data/SeismicData/Overthrust/data5d_6_smooth5_Half_Half.h5'
DATA_NUM = 1200
MODEL_CFG = "sam2_hiera_l.yaml"
SAM2_CKP = "/data/sam2_hiera_large.pt"


def cycle(dl: DataLoader):  # 返回一个迭代器
    while True:
        for data in dl:
            yield data


def _batch_to_xy(batch):
    """
    统一不同数据集的 batch 格式为 (data, data_mask, rx, ry, sx, sy)。
    - SegySSL/字典格式: x_gt, x_obs, gx, gy, sx, sy
    - DatasetH5_all 格式: data, masked_patch, rx_patch, ry_patch, sx_patch, sy_patch
    """
    if not isinstance(batch, dict):
        return None
    if "x_gt" in batch:
        return (
            batch["x_gt"],
            batch["x_obs"],
            batch["gx"],
            batch["gy"],
            batch["sx"],
            batch["sy"],
        )
    if "data" in batch:
        return (
            batch["data"],
            batch["masked_patch"],
            batch["rx_patch"],
            batch["ry_patch"],
            batch["sx_patch"],
            batch["sy_patch"],
        )
    return None


def save_hyperparameters(res_dir, kwargs, accelerator=None):
    """
    将超参数保存到JSON文件中

    Args:
        res_dir: 结果目录路径
        kwargs: 包含超参数的字典
        accelerator: Accelerator 实例
    """
    # 确保只在主进程（rank 0）执行保存
    if accelerator is not None and not accelerator.is_main_process:
        return
    os.makedirs(res_dir, exist_ok=True)  # 创建目录（如果不存在）
    hyperparams = kwargs.copy()
    with open(os.path.join(res_dir, "interp_settings.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)


def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    norm: bool = False,
    accelerator=None,
) -> None:
    # 确保只在主进程（rank 0）执行保存
    if accelerator is not None and not accelerator.is_main_process:
        return

    print(tensor.shape)
    assert len(tensor.shape) == 3
    tensor = tensor[0, :, :].detach().cpu()
    tensor = tensor - tensor.mean()
    # ori_tensor = ori_tensor[0, 0, :, :].detach().cpu()
    plt.figure(figsize=(6, 6))
    if norm:
        tensor /= torch.abs(torch.max(tensor, dim=0, keepdim=True)[0])
    else:
        tensor = tensor
    plt.pcolor(tensor.T, cmap="seismic", vmin=-tensor.std(), vmax=tensor.std())
    plt.ylim(plt.ylim()[::-1])
    plt.title("generate data")
    plt.xticks([])
    plt.xlabel("Trace index")
    plt.ylabel("Time(s)")
    plt.colorbar()
    # plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(fp, dpi=600)
    plt.close()


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y"}


def config():
    parser = argparse.ArgumentParser()
    # train config
    parser.add_argument("--model_name", type=str, default="Unet", help="model name")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--seed", type=int, default=515, help="random seed")
    parser.add_argument("--Type", type=str, default="nomal", help="No help")
    # model config
    parser.add_argument("--in_channels", type=int, default=1, help="input channels")
    parser.add_argument("--time_steps", type=int, default=1000, help="number of steps")
    parser.add_argument("--data_type", type=str, default="sim", help="no help")
    parser.add_argument("--model_type", type=str, default="trace_axis", help="model type")
    parser.add_argument(
        "--geom_mode",
        type=str,
        default="source",
        choices=["source", "receiver", "relative"],
        help="Geometry encoding mode for trace_axis model",
    )
    parser.add_argument("--use_missing_embedding", type=str2bool, default=False, help="use missing focus adapter")
    parser.add_argument("--use_energy_mlp", type=str2bool, default=False, help="use energy stats")
    parser.add_argument("--headwise_attn_output_gate", type=str2bool, default=True, help="use headwise attn output gate")
    parser.add_argument("--elementwise_attn_output_gate", type=str2bool, default=False, help="use elementwise attn output gate")
    parser.add_argument("--chunk_length_flow", type=int, default=256, help="chunk length for time-axis chunking in gated models")
    parser.add_argument("--chunk_overlap_flow", type=float, default=0.5, help="chunk overlap ratio for time-axis chunking in gated models")
    parser.add_argument("--use_p_scale", type=str2bool, default=False, help="apply p_scale to gated model RoPE coordinates")
    # Flow matching specific arguments
    parser.add_argument(
        "--path_type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "VP"],
        help="Flow matching path type",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
        help="Model prediction type",
    )
    parser.add_argument(
        "--loss_weight",
        type=str,
        default=None,
        choices=[None, "velocity", "likelihood"],
        help="Loss weighting type",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="ode",
        choices=["ode", "sde"],
        help="Sampling method: ode or sde",
    )
    parser.add_argument(
        "--ode_num_steps", type=int, default=50, help="Number of ODE steps for sampling"
    )
    parser.add_argument(
        "--sde_num_steps",
        type=int,
        default=250,
        help="Number of SDE steps for sampling",
    )
    # resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint path, e.g. --resume ./resultsFPM/.../checkpoints/model-5.pth",
    )
    # pretrained model initialization
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model weights for initialization (does NOT resume optimizer/scheduler), "
             "e.g. --pretrained ./resultsFPM/.../checkpoints/model-10.pth",
    )
    parser.add_argument(
        "--pretrained_strict",
        type=str2bool,
        default=True,
        help="Whether to strictly enforce matching keys when loading pretrained weights (default: True). "
             "Set to False to allow partial loading (e.g. when model head dimensions differ).",
    )
    # segy profile
    parser.add_argument(
        "--segy_profile",
        type=str,
        default="sw06",
        help="SEG-Y profile for trace sorting: sw06 | field1031 | segc3",
    )
    # others
    # --- 添加分布式训练相关参数 ---
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank. Required for multi-GPU training.",
    )
    # dataset/config.py parses data-specific flags such as --h5File via
    # parse_known_args(); keep the training parser permissive so those flags can
    # be passed in the same launch command.
    return parser.parse_known_args()[0]


class trainer:
    def __init__(
        self,
        embedding_model: torch.nn.Module,
        flow_matching_model: FlowMatchingModel,
        results_folder: str,
        dl: DataLoader,
        tgt_dl: DataLoader,
        val_dl: DataLoader,
        args: argparse.Namespace,
        accelerator: Accelerator,
        train_batch_size: int = 8,
        train_lr: float = 1e-4,
        epochs: int = 1000,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        save_and_sample_every: int = 100,
        num_samples: int = 2,
        finetune: bool = False,
    ):
        """
        初始化训练器

        Args:
            embedding_model: 嵌入模型
            flow_matching_model: Flow Matching 模型
            results_folder: 结果保存文件夹路径
            dl: 训练数据加载器
            tgt_dl: 目标数据加载器
            val_dl: 验证数据加载器
            args: 命令行参数
            accelerator: Accelerator 实例
            train_batch_size: 训练批次大小
            train_lr: 训练学习率
            epochs: 训练总轮数
            adam_betas: Adam优化器的beta参数
            save_and_sample_every: 保存和采样的间隔轮数
            num_samples: 采样数量
            finetune: 是否进行微调
        """
        # self.model = model
        self.embedding_model = embedding_model
        self.flow_matching_model = flow_matching_model
        self.channels = 1
        self.step = 0
        self.start_epoch = 0
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device

        self.results_folder = Path(results_folder)
        # 只在主进程创建目录
        if accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True)
            self.ckp_folder = self.results_folder / "checkpoints"
            self.ckp_folder.mkdir(exist_ok=True)
            self.img_folder = self.results_folder / "images"
            self.img_folder.mkdir(exist_ok=True)
            self.log_folder = self.results_folder / "logs"
            self.log_folder.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_folder))

            # 设置训练日志文件
            self.log_file_path = self.log_folder / "training_log.txt"
            self.log_file_handle = open(self.log_file_path, "a")
        else:
            self.writer = None
            self.log_file_handle = None

        self.dl = dl
        self.val_dl = val_dl
        self.tgt_dl = tgt_dl

        self.train_epochs = epochs
        # Accelerate 会自动处理 DataLoader 的长度
        self.num_steps = len(self.dl)
        self.train_num_steps = self.train_epochs * self.num_steps

        self.batch_size = train_batch_size
        self.train_lr = train_lr

        # frozen_layers_=[self.flow_matching_model.model.tokenier,self.flow_matching_model.model.down]
        frozen_layers_ = []

        if finetune:
            if len(frozen_layers_) != 0:
                for m in self.flow_matching_model.model.modules():
                    if m in frozen_layers_:
                        for params in m.parameters():
                            params.requires_grad = False
            else:
                pass

        # Accelerate 会自动处理模型包装
        model_to_optimize = self.flow_matching_model.model
        if hasattr(model_to_optimize, "module"):
            model_to_optimize = model_to_optimize.module

        self.opt = AdamW(
            [
                {
                    "params": model_to_optimize.parameters(),
                    "lr": train_lr,
                },
            ],
            lr=train_lr,
            betas=adam_betas,
            weight_decay=1e-4,
        )

        self.opt_fint = AdamW(
            [
                {
                    "params": filter(
                        lambda p: p.requires_grad, model_to_optimize.parameters()
                    ),
                    "lr": train_lr,
                }
            ],
            lr=train_lr,
            betas=adam_betas,
            weight_decay=5e-4,
        )
        # Warmup + 温和余弦退火：前 warmup_epochs 线性升 LR，之后余弦衰减到 eta_min
        self.warmup_epochs = 5
        self.base_lr = train_lr
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=max(1, self.train_epochs - self.warmup_epochs),
            eta_min=5e-5,
        )

        # 使用 Accelerate 准备模型、优化器和数据加载器
        self.flow_matching_model.model, self.opt, self.dl, self.val_dl = (
            accelerator.prepare(
                self.flow_matching_model.model, self.opt, self.dl, self.val_dl
            )
        )

        if accelerator.is_main_process:
            # 保存训练配置（在 prepare 之后，确保所有属性都已准备好）
            self._save_training_config(args)

            self._log(f"Model: {type(self.flow_matching_model.model).__name__}")
            self._log(f"Training dataset size: {len(self.dl.dataset)}")
            self._log(f"Validation dataset size: {len(self.val_dl.dataset)}")
            self._log(f"Number of training steps per epoch: {self.num_steps}")
            self._log(f"Total training steps: {self.train_num_steps}")
            self._log(f"Batch size: {self.batch_size}")
            self._log(f"Learning rate: {self.train_lr}")

    def save(self, epoch: int) -> None:
        if not self.accelerator.is_main_process:
            return

        unwrapped_model = self.accelerator.unwrap_model(self.flow_matching_model.model)
        data = {
            "model": unwrapped_model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "step": self.step,
        }

        torch.save(data, str(self.ckp_folder / f"model-epoch-{epoch}.pth"))
        self._log(
            f"Saved checkpoint (epoch {epoch}) to {self.ckp_folder / f'model-epoch-{epoch}.pth'}"
        )

    def load_checkpoint(self, ckp_path: str) -> None:
        ckp = torch.load(ckp_path, map_location=self.device)

        unwrapped_model = self.accelerator.unwrap_model(self.flow_matching_model.model)
        unwrapped_model.load_state_dict(ckp["model"])

        if "optimizer" in ckp:
            self.opt.load_state_dict(ckp["optimizer"])
        if "scheduler" in ckp:
            self.lr_scheduler.load_state_dict(ckp["scheduler"])

        saved_epoch = ckp.get("epoch", 0)
        self.start_epoch = saved_epoch
        self.step = ckp.get("step", 0)

        self._log(
            f"Resumed from checkpoint: {ckp_path} | "
            f"start_epoch={self.start_epoch} | step={self.step}"
        )

    def _log(self, message: str, print_to_console: bool = True):
        """记录日志到文件"""
        if self.accelerator.is_main_process and self.log_file_handle is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {message}\n"
            self.log_file_handle.write(log_message)
            self.log_file_handle.flush()
            if print_to_console:
                print(message)

    def _save_training_config(self, args):
        """保存完整训练配置到 JSON 文件"""
        config = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_args": {k: v for k, v in vars(args).items()},
            "model": {
                "class": type(self.flow_matching_model.model).__name__,
                "total_params": sum(p.numel() for p in self.flow_matching_model.model.parameters()),
                "trainable_params": sum(p.numel() for p in self.flow_matching_model.model.parameters() if p.requires_grad),
                "architecture": str(self.flow_matching_model.model),
            },
            "training": {
                "batch_size": self.batch_size,
                "learning_rate": self.train_lr,
                "epochs": self.train_epochs,
                "num_steps_per_epoch": self.num_steps,
                "total_steps": self.train_num_steps,
            },
            "dataset": {
                "train_length": len(self.dl.dataset),
                "val_length": len(self.val_dl.dataset),
                "class": type(self.dl.dataset).__name__,
            },
        }

        ds = self.dl.dataset
        # 数据集特有配置
        for attr in (
            "h5File_irregular", "h5File_regular", "train_idx_np",
            "survey_line_key", "gather_mode", "missing_ratio_range",
            "patch_amp_percentile", "global_amp_percentile", "time_ps", "trace_ps",
        ):
            if hasattr(ds, attr):
                config["dataset"][attr] = getattr(ds, attr)

        # 归一化统计量
        if hasattr(ds, "coord_stats"):
            config["coord_stats"] = ds.coord_stats
        if hasattr(ds, "scale"):
            config["p_scale"] = ds.scale
        if hasattr(ds, "global_amp_thres"):
            config["global_amp_thres"] = float(ds.global_amp_thres)

        config_path = self.log_folder / "training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    def _compute_val_loss(self):
        """计算验证损失"""
        self.flow_matching_model.model.eval()
        val_losses = []
        val_samples_to_eval = min(50, len(self.val_dl))  # 评估前50个样本

        with torch.no_grad():
            for idx, batch in enumerate(self.val_dl):
                if idx >= val_samples_to_eval:
                    break

                # 适配多种数据格式（SegySSL 字典 / DatasetH5_all 字典 / 元组）
                xy = _batch_to_xy(batch) if isinstance(batch, dict) else None
                if xy is not None:
                    data, data_mask, rx, ry, sx, sy = xy
                else:
                    data, data_mask, rx, ry, sx, sy, _, _ = batch

                data = data.unsqueeze(1).to(self.device)
                data_mask = data_mask.unsqueeze(1).to(self.device)
                rx, ry, sx, sy = (
                    rx.to(self.device),
                    ry.to(self.device),
                    sx.to(self.device),
                    sy.to(self.device),
                )
                condL = (rx, ry, sx, sy)

                with self.accelerator.autocast():
                    loss = self.flow_matching_model(
                        data, condL=condL, x_cond=data_mask, time=None
                    )
                val_losses.append(loss.item())

        self.flow_matching_model.model.train()
        return sum(val_losses) / len(val_losses) if val_losses else float("nan")

    def __del__(self):
        """析构函数，关闭日志文件"""
        if hasattr(self, "log_file_handle") and self.log_file_handle is not None:
            self.log_file_handle.close()

    def gen(self):
        """
        Generate samples from the model.

        This method is intended for generating samples from the trained model.
        Currently it's a placeholder and the main implementation is in train_interpolate method.
        """
        # gen 方法也需要类似修改，但 train_interpolate 是主要训练方法，这里省略 gen 的修改
        pass

    def train_interpolate_improved(self):
        accumulation_steps = 4
        if self.start_epoch > 0:
            self._log(f"Resuming training from epoch {self.start_epoch + 1}")
        with tqdm(
            initial=self.start_epoch,
            total=self.train_epochs,
            desc=f"Training",
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            batch_sample = next(iter(self.val_dl))
            xy_sample = _batch_to_xy(batch_sample) if isinstance(batch_sample, dict) else None
            if xy_sample is not None:
                data_sample, data_mask_sample, rx, ry, sx, sy = xy_sample
            else:
                data_sample, data_mask_sample, rx, ry, sx, sy, _, _ = batch_sample

            if self.accelerator.is_main_process:
                pass

            for epoch in range(self.start_epoch, self.train_epochs):
                loss_list = []
                if hasattr(self.dl.sampler, "set_epoch"):
                    self.dl.sampler.set_epoch(epoch)
                self.flow_matching_model.model.train()
                self.opt.zero_grad(set_to_none=True)

                for idx, batch in enumerate(self.dl):
                    # 适配多种数据格式（SegySSL / DatasetH5_all 字典 或 元组）
                    xy = _batch_to_xy(batch) if isinstance(batch, dict) else None
                    if xy is not None:
                        data, data_mask, rx, ry, sx, sy = xy
                    else:
                        data, data_mask, rx, ry, sx, sy, time_val, coord = batch

                    data = data.unsqueeze(1).to(self.device)
                    data_mask = data_mask.unsqueeze(1).to(self.device)
                    rx, ry, sx, sy = (
                        rx.to(self.device),
                        ry.to(self.device),
                        sx.to(self.device),
                        sy.to(self.device),
                    )
                    condL = (rx, ry, sx, sy)

                    with self.accelerator.autocast():
                        # Flow Matching 的 forward 方法返回 loss
                        loss = self.flow_matching_model(
                            data, condL=condL, x_cond=data_mask, time=None
                        )
                    loss = loss / accumulation_steps
                    self.accelerator.backward(loss)
                    do_step = ((idx + 1) % accumulation_steps == 0) or (
                        idx + 1 == len(self.dl)
                    )
                    if do_step:
                        self.accelerator.clip_grad_norm_(
                            self.flow_matching_model.parameters(), max_norm=1.0
                        )
                        self.opt.step()
                        self.opt.zero_grad(set_to_none=True)

                    # 记录原始 loss（放大回来）
                    loss_list.append(loss.item() * accumulation_steps)

                    # 只打印一次首个 loss
                    if epoch == 0 and idx == 0 and self.accelerator.is_main_process:
                        first_loss = loss.item() * accumulation_steps
                        print("the first loss is:", first_loss)
                        self._log(f"The first loss is: {first_loss:.4f}")

                # Warmup: 线性增加 LR；之后温和余弦退火
                if epoch < self.warmup_epochs:
                    warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
                    for pg in self.opt.param_groups:
                        pg["lr"] = warmup_lr
                else:
                    self.lr_scheduler.step()
                avg_loss = (
                    sum(loss_list) / len(loss_list) if loss_list else float("nan")
                )
                current_lr = self.opt.param_groups[0]["lr"]

                gc.collect()
                torch.cuda.empty_cache()

                # 计算验证损失（每10个epoch或第0个epoch）
                val_loss = None
                if epoch % 10 == 0 or epoch == 0:
                    val_loss = self._compute_val_loss()

                if self.accelerator.is_main_process:
                    pbar.set_postfix(
                        {
                            "current_epoch": epoch + 1,
                            "loss": avg_loss,
                            "val_loss": val_loss if val_loss is not None else "N/A",
                            "lr": current_lr,
                        }
                    )
                    # 记录到 TensorBoard
                    self.writer.add_scalar("Loss/train", avg_loss, epoch)
                    self.writer.add_scalar("LearningRate/train", current_lr, epoch)
                    if val_loss is not None:
                        self.writer.add_scalar("Loss/val", val_loss, epoch)
                    # 记录到日志文件
                    #self._log(
                    #    f"Epoch {epoch+1}/{self.train_epochs} - Train Loss: {avg_loss}, "
                    #    f"Val Loss: {val_loss if val_loss is not None else 'N/A':.4f}, "
                    #    f"LR: {current_lr:.6f}"
                    # )
                if (
                    (epoch + 1) % self.save_and_sample_every == 0 or epoch == self.start_epoch
                ) and self.accelerator.is_main_process:
                    unwrapped_model = self.accelerator.unwrap_model(
                        self.flow_matching_model.model
                    )
                    self.flow_matching_model.model.eval()
                    old_disable = os.environ.get("TQDM_DISABLE", "0")
                    os.environ["TQDM_DISABLE"] = "1"
                    try:
                        self.save(epoch + 1)
                        torch.cuda.empty_cache()
                    finally:
                        os.environ["TQDM_DISABLE"] = old_disable
                # Accelerate 会自动处理同步
                self.accelerator.wait_for_everyone()

                pbar.update(1)


def main():
    """
    Main function to run the training process.

    This function orchestrates the complete training pipeline including:
    - Parsing command line arguments
    - Setting up Accelerate framework
    - Loading datasets based on data type
    - Initializing the model and flow matching process
    - Creating the trainer and starting the training loop

    The function supports multiple data types (c3NA_ssl, xbfy, dongfang, etc.)
    and uses Accelerate to simplify distributed training setup.
    """
    args = config()
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dataset_args = ds_config.args

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",  # 可以改为 'fp16' 或 'bf16' 来加速训练
        kwargs_handlers=[ddp_kwargs],
    )

    # 获取设备信息
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    if rank ==0:
        print("dataset_args:", dataset_args)
        print(f"[Rank {rank}/{world_size}] Using device: {device}")
    # set seed
    # 手动设置随机种子以确保可重复性
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    interp_kwargs = dict(
        model_type=args.model_type,
        missing_ratio=dataset_args.min_r,
        max_ratio=dataset_args.max_r,
        missing_type=args.model_type,
    )

    # Flow Matching 参数配置
    loss_weight_final = None
    if args.loss_weight not in (None, "None", "none", ""):
        loss_weight_final = args.loss_weight
    fpm_kwargs = dict(
        time_num=dataset_args.time_ps,
        trace_num=dataset_args.trace_ps,
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=loss_weight_final,
        train_eps=None,  # 使用默认值
        sample_eps=None,  # 使用默认值
        sample_num=1,
        device=device,
        sup_mode="all",  # 保持兼容性
        use_coherence=False,
        sigma_obs=0.001,
        use_bayesian=False,
        sampling_method=args.sampling_method,
        ode_num_steps=args.ode_num_steps,
        sde_num_steps=args.sde_num_steps,
    )
    #dataset init
    profile = get_segy_profile(args.segy_profile)
    dataset = datasets_interp.DatasetH5_interp(
        h5File_irregular=dataset_args.h5File,
        h5File_regular=dataset_args.h5File_regular,
        train_idx_np=None,
        train=dataset_args.train,
        missing_ratio_range=(0.3, 0.5),
        use_p_scale=args.use_p_scale,
        profile=profile,
    )
    #dataset_1 = datasets.DatasetH5_all_queryctx(h5File=dataset_args.h5File, h5File_regular=dataset_args.h5File_regular, h5File_tgt=dataset_args.h5File_tgt, dataset_neighbors=dataset_args.dataset_neighbors,train=dataset_args.train)
    #dataset_1 = datasets_interp.DatasetH5_interp(h5File_irregular=dataset_args.h5File, h5File_regular=dataset_args.h5File_regular, train_idx_np=dataset_args.train_idx_np, train=dataset_args.train, survey_line_key="recv_line", missing_ratio_range=(0.5, 0.7))
    #dataset_2 = datasets_interp.DatasetH5_interp(h5File_irregular=dataset_args.h5File_2, h5File_regular=dataset_args.h5File_regular_2,train_idx_np=dataset_args.train_idx_np_2, train=dataset_args.train, survey_line_key="recv_line", missing_ratio_range=(0.5, 0.7))
    #dataset_3 = datasets_interp.DatasetH5_interp(h5File_irregular=dataset_args.h5File_3, h5File_regular=dataset_args.h5File_regular_3,train_idx_np=dataset_args.train_idx_np_3, train=dataset_args.train, survey_line_key="recv_line", missing_ratio_range=(0.5, 0.7))
    #dataset_4 = datasets_interp.DatasetH5_interp(h5File_irregular=dataset_args.h5File_4, h5File_regular=dataset_args.h5File_regular_4,train_idx_np=dataset_args.train_idx_np_4, train=dataset_args.train, survey_line_key="recv_line", missing_ratio_range=(0.5, 0.7))
    print(f'use_p_scale:{args.use_p_scale}')
    train_sampler = DistributedSampler(dataset) if world_size > 1 else None
    val_sampler = (
        DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    )

    dl_SIM = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=5,
        sampler=train_sampler,
        drop_last=True,
    )
    dl_SIM_VAL = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, sampler=val_sampler,
        drop_last=True,
    )
    if rank ==0:
        print("dataset length:", len(dataset))
        print('data_shape:',dataset[0]['data'].shape)
        print("dl_SIM length:", len(dl_SIM))
        print("dl_SIM_VAL length:", len(dl_SIM_VAL))
    
    # model init - 支持 trace_axis 和 vit_bottleneck 两种模型
    pe_type = "transformer"
    if "tp" in args.model_type:
        print("using vit_bottleneck model")
        model_unet = SeisDiTRope(
            image_channels=2,
            n_channels=32,
            f_dict=None,
            num_layers=8,
            d_model=512,
            pe_type=pe_type,
        )
    elif "gated_encdec" in args.model_type:
        print("using gated_seisdit_gen_encdec model")
        model_unet = create_gated_seisdit_gen_encdec(
            image_channels=2,
            d_model=1080,
            d_ff=2048,
            nhead=6,
            num_encoder_layers=6,
            num_bottleneck_layers=4,
            num_memory_bottleneck_layers=2,
            num_decoder_layers=6,
            chunk_length=args.chunk_length_flow,
            chunk_overlap=args.chunk_overlap_flow,
            use_energy_stats=args.use_energy_mlp,
            use_missing_embed=args.use_missing_embedding,
            headwise_attn_output_gate=args.headwise_attn_output_gate,
            elementwise_attn_output_gate=args.elementwise_attn_output_gate,
            use_qk_norm=True,
            qkv_bias=False,
            use_relative_bias=True,
            use_encoder_relative_bias=False,
            use_decoder_self_relative_bias=False,
            use_cross_relative_bias=True,
            use_time_adaln=True,
            use_cross_query_gate=True,
            num_attn_res_blocks=2,
        )
    elif "gated" in args.model_type: #drop
        print("using gated_seisdit_gen model")
        model_unet = create_gated_seisdit(
            image_channels=2,
            d_model=1080,
            d_ff=2048,
            nhead=6,
            num_encoder_layers=6,
            num_bottleneck_layers=4,
            num_decoder_layers=6,
            chunk_length=args.chunk_length_flow,
            chunk_overlap=args.chunk_overlap_flow,
            use_energy_stats=args.use_energy_mlp,
            use_missing_embed=args.use_missing_embedding,
            headwise_attn_output_gate=args.headwise_attn_output_gate,
            elementwise_attn_output_gate=args.elementwise_attn_output_gate,
        )
    elif 'trace_axis' in args.model_type:
        print("using trace_axis model")
        model_unet = SeisDiTRopeV2(
            image_channels=2,
            n_channels=32,
            f_dict=None,
            num_layers=8,
            d_model=512,
            pe_type=pe_type,
            missing_focus_adapter=args.use_missing_embedding,
            geom_mode=args.geom_mode,
        )

    print("time_steps:", args.time_steps)

    # ---- Pretrained model initialization ----
    if args.pretrained is not None:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")
        if rank == 0:
            print(f"Loading pretrained weights from: {args.pretrained} (strict={args.pretrained_strict})")
        ckp = torch.load(args.pretrained, map_location="cpu")
        state_dict = ckp.get("model", ckp)  # support both full checkpoint and raw state_dict
        # strip DDP wrapper prefix if present
        stripped = {}
        for k, v in state_dict.items():
            key = k
            if key.startswith("module."):
                key = key[7:]
            stripped[key] = v
        missing, unexpected = model_unet.load_state_dict(stripped, strict=args.pretrained_strict)
        if rank == 0:
            if missing:
                print(f"Missing keys ({len(missing)}): {missing}")
            if unexpected:
                print(f"Unexpected keys ({len(unexpected)}): {unexpected}")
        print(f"Pretrained weights loaded successfully from: {args.pretrained}")
    # ----------------------------------------------------------------

    res_dir = f"./resultsFPM/{args.model_name}_datatype_{args.data_type}_0517"

    # 将模型移动到正确的设备
    model_unet = model_unet.to(device)

    # 创建 Flow Matching Model
    fpm = FlowMatchingModel(
        model=model_unet,
        trace_num=fpm_kwargs["trace_num"],
        time_steps=fpm_kwargs["time_num"],
        path_type=fpm_kwargs["path_type"],
        prediction=fpm_kwargs["prediction"],
        loss_weight=fpm_kwargs["loss_weight"],
        train_eps=fpm_kwargs["train_eps"],
        sample_eps=fpm_kwargs["sample_eps"],
        sample_num=fpm_kwargs["sample_num"],
        device=fpm_kwargs["device"],
        sup_mode=fpm_kwargs["sup_mode"],
        use_coherence=fpm_kwargs["use_coherence"],
        sigma_obs=fpm_kwargs["sigma_obs"],
        use_bayesian=fpm_kwargs["use_bayesian"],
        sampling_method=fpm_kwargs["sampling_method"],
        ode_num_steps=fpm_kwargs["ode_num_steps"],
        sde_num_steps=fpm_kwargs["sde_num_steps"],
    )

    # 只在主进程创建结果目录
    if accelerator.is_main_process:
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

    # init trainer
    trainer_fpm = trainer(
        embedding_model=None,
        flow_matching_model=fpm,
        results_folder=res_dir,
        dl=dl_SIM,
        tgt_dl=None,
        val_dl=dl_SIM_VAL,
        args=args,
        accelerator=accelerator,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        epochs=args.epochs,
        save_and_sample_every=10,
        num_samples=args.batch_size,
        finetune=False,
    )

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        accelerator.wait_for_everyone()
        trainer_fpm.load_checkpoint(args.resume)
        accelerator.wait_for_everyone()

    # train
    trainer_fpm.train_interpolate_improved()


if __name__ == "__main__":
    main()
