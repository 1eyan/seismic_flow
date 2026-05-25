# Seismic Flow - 5D 地震数据 Flow Matching 重建

本仓库实现基于 Conditional Flow Matching 的 5D 地震缺失道插值/重建流程。代码覆盖 SEG-Y 道头检查、SEG-Y 到 H5 转换、H5 patch 数据集构建、Accelerate 多卡训练、H5 推理以及将预测结果按几何键回填到 SEG-Y。

依据代码范围：`convert_tool/`、`convert_toolV2/`、`dataset/`、`models/`、`transport/`、`FPM.py`、`train_fpmV3_ddp.py`、`gen_infer.py`、`segy_schema.py`、`check_segy_headers.py`、`run_*.sh`。

## 一、整体工作流

```text
原始/缺失/标签 SEG-Y
        |
        | 1. 检查道头、缺失道和几何字段
        v
check_segy_headers.py
        |
        | 2. 统一道序并转成 H5
        v
convert_toolV2/Segy2H5.py 或 convert_tool/Segy2H5.py
        |
        | 3. 配置 H5 路径、patch 大小、训练缺失策略
        v
dataset/config.py + dataset/datasets_interp_v2.py
        |
        | 4. Flow Matching 训练
        v
run_fm_multi_gpu.sh -> train_fpmV3_ddp.py -> FPM.py -> transport/
        |
        | 5. patch 推理、几何键聚合、SEG-Y 回填
        v
run_gen_infer_*.sh -> gen_infer.py
        |
        | 6. 输出 SEG-Y、残差、CSV/JSON 报告、可视化
        v
gen_fill_results*/
```

推荐执行顺序：

1. 用 `check_segy_headers.py` 检查输入 SEG-Y 的 trace 数、采样点数、关键道头字段、空道比例。
2. 用 `convert_toolV2/run_convert.sh` 或 `convert_tool/Segy2H5.py` 将 `irregular/mask/label` SEG-Y 转为 H5。
3. 修改 `dataset/config.py` 中的 `--h5File`、`--h5File_regular`、`--time_ps`、`--trace_ps` 等训练数据参数。
4. 修改或覆盖 `run_fm_multi_gpu.sh` 中的模型、Flow Matching 和 profile 参数，启动训练。
5. 用 `run_gen_infer_sw06.sh` 或 `run_gen_infer_field1031.sh` 指定 checkpoint、H5、mask SEG-Y，执行推理回填。
6. 检查 `summary.json`、`filled_missing.sgy`、`residual.sgy`、`*_traces.csv` 和 `vis/`。

## 二、运行环境

仓库当前没有 `requirements.txt` 或 `environment.yml`。从源码导入推断，至少需要以下 Python 包：

| 类别 | 依赖 |
| --- | --- |
| 深度学习 | `torch`, `torchvision` 可选, `accelerate`, `torchdiffeq`, `tensorboard` |
| 模型组件 | `timm`, `einops` |
| SEG-Y/H5 | `segyio`, `seisio`, `h5py`, `pandas` |
| 数值与可视化 | `numpy`, `matplotlib`, `tqdm` |

建议补充：为复现实验，建议后续增加 `requirements.txt` 或 Conda 环境文件，并记录 CUDA、PyTorch、segyio、accelerate 的版本。

## 三、数据与 H5 格式

转换脚本写出的 H5 默认取第一个 group 作为数据组，常用 group 名为 `1551`。主训练和推理数据集读取该 group 下的数据。

| H5 字段 | 说明 |
| --- | --- |
| `data` | 地震道矩阵，形状通常为 `[n_trace, n_sample]` |
| `sx`, `sy`, `rx`, `ry` | 炮点和检波点坐标，用于坐标归一化和模型条件输入 |
| `delta`, `t0` | 采样间隔和起始时间 |
| `shot_line`, `shot_no`, `recv_line`, `recv_no` | 常规道头键 |
| `shot_stake`, `recv_stake`, `cmp`, `cmp_line`, `offset` | `fixed` 模式下可写入的扩展道头键 |
| `trace_idx` | 转换前原始 SEG-Y 道序索引 |
| `mx/my/hx/hy/imx/imy/ihx/ihy/fold` | 启用 `--compute-ovt` 后写入的 OVT 字段 |

数据集处理要点：

- `DatasetH5_interp` 是当前训练和推理主类，位于 `dataset/datasets_interp_v2.py`。
- 训练模式按顺序滑窗取 `trace_ps` 道，随机整道置零，构造 self-supervised 缺失输入。
- 推理模式按 `trace_ps` 和 `overlap_ratio` 生成滑动 patch，重叠区域的同一几何键预测会求平均。
- 时间维如果长于 `time_ps`，保留深部样本；短于 `time_ps`，在浅部前置补零。
- 振幅按 patch 的 `patch_amp_percentile` 百分位裁剪并归一化，推理输出再乘回 `amp_scale`。
- 坐标使用规则 H5 的坐标范围做 min-max 归一化；启用 `use_p_scale` 时会基于网格步长调整 RoPE 坐标尺度。

## 四、SEG-Y Profile 与道头键

SEG-Y profile 定义在 `segy_schema.py`，训练、转换、推理必须使用一致的 profile，否则 H5 排序和 SEG-Y 回填键可能不一致。

| Profile | 默认道头模式 | 几何匹配键 `key_columns` | 默认排序键 `sort_keys` | 适用说明 |
| --- | --- | --- | --- | --- |
| `sw06` | `fixed` | `shot_line, shot_no, recv_line, recv_no` | `shot_line, shot_no, recv_line, recv_no` | 东方合成 sw06 风格 |
| `field1031` | `fixed` | `shot_line, shot_stake, recv_line, recv_stake` | `recv_line, recv_stake, shot_line, shot_stake` | 1031 野外数据风格 |
| `segc3` | `self_computed` | `shot_line, shot_stake, recv_line, recv_stake` | `recv_line, recv_stake, shot_line, shot_stake` | 仅依赖坐标字段并反算 line/stake |

关键字节位置：

| Profile | 字段位置 |
| --- | --- |
| `sw06` | `shot_line=221`, `shot_no=25`, `recv_line=61`, `recv_no=65`, `shot_stake=225`, `recv_stake=229`, `shot_x=73`, `shot_y=77`, `rec_x=81`, `rec_y=85` |
| `field1031` | `shot_line=17`, `shot_stake=21`, `shot_no=25`, `recv_line=61`, `recv_stake=65`, `recv_no=41`, `shot_x=73`, `shot_y=77`, `rec_x=81`, `rec_y=85` |
| `segc3` | `shot_x=73`, `shot_y=77`, `rec_x=81`, `rec_y=85`，line/stake 由坐标和 scalar 反算 |

## 五、工作流 1：SEG-Y 道头检查

入口脚本：`check_segy_headers.py`。

用途：

- 单文件查看 SEG-Y 基本信息、二进制头、文本头片段、前后若干道道头字段。
- 双文件对比 trace 数、采样点、二进制头差异、道头统计差异。
- 可选检查空道和近零道。

命令示例：

```bash
python3 check_segy_headers.py /path/to/a.sgy
python3 check_segy_headers.py /path/to/a.sgy /path/to/b.sgy
python3 check_segy_headers.py /path/to/a.sgy --fields sx,sy,gx,gy,offset,ns,dt -n 10
python3 check_segy_headers.py /path/to/a.sgy --check-empty
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `segy_a` | 必填 | 第一个 SEG-Y 文件 |
| `segy_b` | 可选 | 第二个 SEG-Y 文件；提供后进入对比模式 |
| `-n`, `--num-traces` | `5` | 展示头/尾多少道 |
| `--tail-only` | `False` | 单文件模式只显示尾部 trace |
| `--all-traces` | `False` | 展示全部 trace，道数大时谨慎使用 |
| `--check-empty` | `False` | 统计空道/近零道 |
| `--fields` | 内置常用字段 | 逗号分隔的道头字段名 |

## 六、工作流 2：SEG-Y 转 H5

### 2.1 V2 三文件转换

入口脚本：`convert_toolV2/run_convert.sh` -> `convert_toolV2/Segy2H5.py`。

适用场景：一次性将 `irregular`、`mask`、`label` 三个 SEG-Y 转为三个 H5，输出到三者公共目录下的 `h5/` 子目录。

```bash
cd convert_toolV2
IRR_SEGY=/path/irr.sgy \
MASK_SEGY=/path/mask.sgy \
LABEL_SEGY=/path/label.sgy \
SEGY_PROFILE=sw06 \
MODE=fixed \
DATASET_NAME=sw06 \
GROUP_NAME=1551 \
./run_convert.sh
```

`Segy2H5.py` 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--irr` | 无 | irregular SEG-Y 路径 |
| `--mask` | 无 | mask SEG-Y 路径 |
| `--label` | 无 | label SEG-Y 路径 |
| `--dataset-name` | `dataset` | 输出 H5 前缀，生成 `{name}_irregular.h5`、`{name}_mask.h5`、`{name}_label.h5` |
| `--mode` | `self_computed` | `fixed` 直接读 profile 字节；`self_computed` 由坐标反算 line/stake |
| `--group-name` | `1551` | H5 group 名 |
| `--compute-ovt` | `False` | 写入 midpoint/half-offset OVT 字段 |
| `--segy_profile` | 默认 `sw06` | `sw06`、`field1031` 或 `segc3` |

注意：

- V2 三文件转换会先从 `irr` 读取 headers，并复用该 headers 去组织 `mask` 和 `label`，因此三者应保持相同 trace 数和几何顺序。
- `run_convert.sh` 内置默认路径为当前机器绝对路径，实际运行前应通过环境变量覆盖或直接修改脚本。

### 2.2 V1 单文件转换

入口脚本：`convert_tool/Segy2H5.py`，配置文件：`convert_tool/dataset_config.py`。

```bash
cd convert_tool
python3 Segy2H5.py \
  --segy-profile sw06 \
  --mode fixed \
  --input-segy /path/to/data.sgy \
  --info-h5 /path/to/output.h5 \
  --group-name 1551
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--segy-profile` | `dataset_config.segy_profile` 或 `sw06` | 道头 profile |
| `--mode` | `dataset_config.segy_mode` 或 `fixed` | `fixed` 或 `self_computed` |
| `--sort-keys` | profile 默认 | 逗号分隔排序键 |
| `--info-h5` | `dataset_config.info_h5` | 输出 H5 路径 |
| `--input-segy` | `dataset_config.segyPairs` 第一项 | 输入 SEG-Y |
| `--group-name` | `segyPairs` 第一个 key | H5 group 名 |

### 2.3 V1 批量转换

入口脚本：`convert_tool/batch_segy2h5.py`。

使用方式：

1. 修改 `convert_tool/dataset_config.py` 的 `info_h5` 和 `segyPairs`。
2. 执行批量转换。

```bash
cd convert_tool
python3 batch_segy2h5.py --num-workers 4 --gzip-level 1
python3 batch_segy2h5.py --compute-ovt --mx-bin 25 --my-bin 25 --hx-bin 12.5 --hy-bin 12.5
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--num-workers` | `4` | 并行进程数 |
| `--compute-ovt` | `False` | 计算并写入 OVT 字段 |
| `--mx-bin`, `--my-bin`, `--hx-bin`, `--hy-bin` | `None` | OVT 分箱大小；为空则自动估计 |
| `--keep-temp` | `False` | 保留临时 H5 |
| `--segy-profile` | 配置文件或 `sw06` | 道头 profile |
| `--mode` | 配置文件或 `fixed` | 道头读取模式 |
| `--sort-keys` | profile 默认 | 逗号分隔排序键 |
| `--gzip-level` | `1` | H5 gzip 压缩级别，1 最快，9 最小 |
| `--chunk-ntrace` | `128` | `data` 数据集 trace 维 chunk |
| `--chunk-nsample` | `256` | `data` 数据集 sample 维 chunk |

### 2.4 V2 批量转换

入口脚本：`convert_toolV2/batch_segy2h5.py`。

该脚本结构与 V1 批量转换类似，但当前不暴露 `--segy-profile` 和 `--mode` 参数，内部使用 `convert_toolV2/Segy2H5.py` 的模块默认 profile。需要多 profile 批处理时，优先使用 `convert_tool/batch_segy2h5.py`。

## 七、工作流 3：训练数据配置

入口配置：`dataset/config.py`。

`train_fpmV3_ddp.py` 会导入 `dataset.config.args`，因此训练数据路径和 patch 大小主要由该文件解析。`dataset/config.py` 和主训练脚本都使用 `parse_known_args()`，数据集参数和训练参数可以写在同一条启动命令中。

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--h5File` | 机器绝对路径 | irregular/mask H5，用作训练输入 |
| `--h5File_regular` | 机器绝对路径 | regular/label H5，用作训练目标和坐标统计 |
| `--train_idx_np` | 机器绝对路径 | 可选训练 trace 索引；当前主训练脚本传入 `None`，不会使用该默认值 |
| `--time_ps` | `2048` | 时间采样 patch 长度 |
| `--trace_ps` | `128` | trace patch 宽度 |
| `--sample_num` | `2048` | 旧参数，主训练链路未显式使用 |
| `--train` | `True` | 数据集训练模式 |
| `--expand` | `0.1` | 旧参数，主训练链路未显式使用 |
| `--min_r`, `--max_r` | `0.4`, `0.7` | 配置中保留；当前 `train_fpmV3_ddp.py` 实际传给数据集的是硬编码 `(0.3, 0.5)` |
| `--ovt_mask_*` | 多个 | OVT 数据集相关参数，主训练链路未显式使用 |

建议修改方式：

```bash
python3 train_fpmV3_ddp.py \
  --h5File /path/sw06_mask.h5 \
  --h5File_regular /path/sw06_label.h5 \
  --time_ps 2048 \
  --trace_ps 224 \
  --model_name trace_axis \
  --model_type trace_axis
```

注意：上例中的 `--h5File` 等参数会被 `dataset/config.py` 解析，`--model_name` 等参数会继续交给 `train_fpmV3_ddp.py`。

## 八、工作流 4：模型训练

推荐入口：`run_fm_multi_gpu.sh`。

直接入口：`train_fpmV3_ddp.py`。

### 4.1 使用 shell 脚本训练

```bash
bash run_fm_multi_gpu.sh
```

`run_fm_multi_gpu.sh` 默认使用：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_NAME` | `trace_axis` | 结果目录命名用 |
| `BATCH_SIZE` | `1` | 每 GPU batch size |
| `LR` | `1e-4` | 学习率 |
| `EPOCHS` | `200` | 训练 epoch 数 |
| `MODEL_TYPE` | `trace_axis` | `trace_axis`、`gated`、`gated_encdec`、`tp/vit` |
| `DATA_TYPE` | `sw06` | 结果目录命名用 |
| `SEGY_PROFILE` | `sw06` | 道头 profile |
| `USE_P_SCALE` | `true` | 是否启用坐标 p_scale |
| `PATH_TYPE` | `Linear` | Flow Matching 路径 |
| `PREDICTION` | `velocity` | 模型预测目标 |
| `SAMPLING_METHOD` | `ode` | 采样方式 |
| `ODE_NUM_STEPS` | `50` | ODE 采样步数 |
| `SDE_NUM_STEPS` | `250` | SDE 采样步数 |
| `RESUME` | 空 | checkpoint 路径，非空时恢复训练 |

脚本里部分变量直接赋值，部分变量可通过环境变量覆盖。若要批量实验，建议直接调用 `accelerate launch` 或显式修改脚本变量。

### 4.2 直接使用 Accelerate

```bash
accelerate launch --config_file accelerate_config.yaml --main_process_port 29501 train_fpmV3_ddp.py \
  --model_name trace_axis \
  --batch_size 1 \
  --lr 1e-4 \
  --epochs 200 \
  --model_type trace_axis \
  --seed 515 \
  --data_type sw06 \
  --segy_profile sw06 \
  --use_p_scale true \
  --geom_mode source \
  --path_type Linear \
  --prediction velocity \
  --sampling_method ode \
  --ode_num_steps 50
```

`accelerate_config.yaml` 当前设置为本机多 GPU：`distributed_type=MULTI_GPU`、`gpu_ids='0,1,2,3'`、`num_processes=4`。

### 4.3 `train_fpmV3_ddp.py` 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_name` | `Unet` | 结果目录命名；脚本中结果目录为 `./resultsFPM/{model_name}_datatype_{data_type}_0517` |
| `--batch_size` | `16` | 每卡 batch size |
| `--lr` | `1e-4` | AdamW 学习率 |
| `--epochs` | `100` | 训练轮数 |
| `--device` | `cuda` | 保留参数，实际设备由 Accelerate 控制 |
| `--seed` | `515` | Python、NumPy、PyTorch 随机种子 |
| `--Type` | `nomal` | 保留参数，主链路未使用 |
| `--in_channels` | `1` | 保留参数，主链路未使用 |
| `--time_steps` | `1000` | 模型时间嵌入尺度；与 `FPM.py` 中 time normalization 相关 |
| `--data_type` | `sim` | 结果目录命名 |
| `--model_type` | `trace_axis` | 模型选择，见下表 |
| `--geom_mode` | `source` | `trace_axis` 几何编码方式：`source`、`receiver`、`relative` |
| `--use_missing_embedding` | `False` | 是否启用 missing focus adapter 或 gated missing embedding |
| `--use_energy_mlp` | `False` | gated 模型是否使用能量统计 |
| `--headwise_attn_output_gate` | `True` | gated attention 输出门控 |
| `--elementwise_attn_output_gate` | `False` | elementwise 输出门控 |
| `--chunk_length_flow` | `256` | gated 模型时间维 chunk 长度 |
| `--chunk_overlap_flow` | `0.5` | gated 模型 chunk 重叠比例 |
| `--use_p_scale` | `False` | 坐标归一化前是否应用 p_scale |
| `--path_type` | `Linear` | Flow path：`Linear`、`GVP`、`VP` |
| `--prediction` | `velocity` | `velocity`、`score`、`noise` |
| `--loss_weight` | `None` | `None`、`velocity`、`likelihood` |
| `--sampling_method` | `ode` | 训练时保存采样配置：`ode` 或 `sde` |
| `--ode_num_steps` | `50` | ODE 采样步数 |
| `--sde_num_steps` | `250` | SDE 采样步数 |
| `--resume` | `None` | 恢复训练 checkpoint |
| `--segy_profile` | `sw06` | 训练数据排序和几何键 profile |
| `--local_rank` | `-1` | 分布式保留参数 |

训练内部固定行为：

- 使用 `Accelerator(mixed_precision="fp16")`。
- 使用 `DistributedDataParallelKwargs(find_unused_parameters=True)`。
- 优化器为 AdamW，主优化器 `weight_decay=1e-4`。
- 内部梯度累积步数为 `4`，CLI 未暴露。
- 学习率前 `5` 个 epoch 线性 warmup，之后 `CosineAnnealingLR`，`eta_min=5e-5`。
- 每 `10` 个 epoch 保存一次 checkpoint；起始 epoch 也会保存一次。
- 每 `10` 个 epoch 或第 0 个 epoch 计算验证 loss，最多评估 50 个 batch。
- 训练日志写入 `resultsFPM/.../logs/training_log.txt` 和 TensorBoard。
- 完整配置写入 `resultsFPM/.../logs/training_config.json`。

### 4.4 模型类型

| `--model_type` | 代码路径 | 说明 |
| --- | --- | --- |
| 包含 `trace_axis` | `models/seisdit_trace_axis.py::SeisDiTRopeV2` | 当前主模型，trace-axis attention + RoPE |
| 包含 `gated_encdec` | `models/gated_seisdit_gen_encdec.py` | gated 生成式 encoder-decoder |
| 包含 `gated` | `models/gated_seisdit_gen.py` | gated SeisDiT 生成器 |
| 包含 `tp` 或推理中包含 `vit` | `models/seisdit_vit_bottleneck.py::SeisDiTRope` | ViT bottleneck 变体 |

训练脚本中 `trace_axis` 的主结构固定为 `n_channels=32`、`num_layers=8`、`d_model=512`。gated 训练结构固定为 `d_model=1080`、`d_ff=2048`、`nhead=6`，推理时对应超参数必须与 checkpoint 匹配。

### 4.5 Flow Matching 参数

`FPM.py::FlowMatchingModel` 封装 backbone、`transport.create_transport()` 和 sampler。

| 参数 | 说明 |
| --- | --- |
| `path_type=Linear/GVP/VP` | 选择 interpolant path，对应 `transport/path.py` |
| `prediction=velocity/score/noise` | 模型预测速度、score 或 noise |
| `loss_weight=None/velocity/likelihood` | 训练损失加权方式 |
| `sampling_method=ode/sde` | 推理采样走 ODE 或 SDE |
| `ode_sampling_method` | 默认 `dopri5`，推理脚本可设置 |
| `ode_num_steps`, `ode_atol`, `ode_rtol` | ODE 采样步数和容差 |
| `sde_sampling_method`, `sde_num_steps` | SDE 采样方法和步数 |

## 九、工作流 5：推理与 SEG-Y 回填

推荐入口：

- `run_gen_infer_sw06.sh`
- `run_gen_infer_field1031.sh`

直接入口：`gen_infer.py`。

### 5.1 shell 脚本示例

```bash
# sw06，默认 4 进程 torchrun
CHECKPOINT=/path/model-epoch-140.pth \
H5_REGULAR=/path/sw06_label.h5 \
H5_MASK=/path/sw06_mask.h5 \
MASK_SEGY=/path/mask.sgy \
LABEL_SEGY=/path/label.sgy \
NPROC_PER_NODE=4 \
bash run_gen_infer_sw06.sh

# field1031，默认 8 进程 torchrun
CHECKPOINT=/path/model-epoch-190.pth \
H5_REGULAR=/path/field1031_label.h5 \
H5_MASK=/path/field1031_mask.h5 \
MASK_SEGY=/path/mask_from_label.sgy \
LABEL_SEGY=/path/reg5dbin_label1031.sgy \
NPROC_PER_NODE=8 \
bash run_gen_infer_field1031.sh
```

两个脚本都会把 stdout 保存到 `${OUTPUT_DIR}/run_gen_infer.stdout.log`。

### 5.2 `gen_infer.py` 直接调用

```bash
python3 gen_infer.py \
  --checkpoint /path/model.pth \
  --h5_regular /path/label.h5 \
  --h5_mask /path/mask.h5 \
  --mask_path /path/mask.sgy \
  --label_segy /path/label.sgy \
  --output_dir ./gen_fill_results \
  --segy_profile sw06 \
  --header_mode fixed \
  --model_type trace_axis \
  --time_ps 2048 \
  --trace_ps 224 \
  --overlap_ratio 0.5 \
  --use_p_scale true \
  --sampling_method ode \
  --ode_num_steps 50 \
  --visualize true
```

### 5.3 推理输入输出

输入：

| 输入 | 说明 |
| --- | --- |
| `--checkpoint` | 训练得到的 `.pth` 文件 |
| `--h5_regular` | regular/label H5，用于 target、坐标统计和可视化对比 |
| `--h5_mask` | 含缺失道的 H5，用于推理输入 |
| `--mask_path` | 含缺失道的 SEG-Y 模板，脚本只回填全零缺失道 |
| `--label_segy` | 可选，计算残差 SEG-Y 和残差统计 |

输出：

| 输出 | 说明 |
| --- | --- |
| `filled_missing.sgy` | 回填后的 SEG-Y |
| `filled_missing_sorted.sgy` | `--sort_output true` 时输出，道头和道数据按 profile sort keys 同步排序 |
| `residual.sgy` | 提供 `--label_segy` 时输出，只在填充道上写预测与标签差 |
| `summary.json` | trace 数、缺失数、填充数、未填充数、耗时等 |
| `gen_infer_fill_segy.log` | 推理日志 |
| `filled_missing_traces.csv` | 已填充 trace 索引 |
| `unfilled_missing_traces.csv` | 仍未填充的缺失 trace 索引 |
| `still_missing_after_write_traces.csv` | 写回后仍接近全零的 trace |
| `observed_changed_traces.csv` | 非缺失道被改变的异常索引 |
| `unmatched_prediction_traces.csv` | 几何键匹配异常相关索引 |
| `vis/batch_*.png` | 开启可视化时输出 masked/pred/gt/residual/FK/SNR 图 |

### 5.4 `gen_infer.py` 参数

数据与输出：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--checkpoint` | 必填 | 模型 checkpoint |
| `--h5_regular` | 必填 | regular/label H5 |
| `--h5_mask` | 必填 | mask H5 |
| `--mask_path` | 必填 | mask SEG-Y 模板 |
| `--label_segy` | `None` | 可选标签 SEG-Y，用于残差 |
| `--output_dir` | `gen_fill_results` | 输出目录 |
| `--output_segy` | `{output_dir}/filled_missing.sgy` | 回填输出 SEG-Y |
| `--output_residual_segy` | `{output_dir}/residual.sgy` | 残差 SEG-Y |
| `--segy_profile` | `sw06` | 道头 profile |
| `--header_mode` | profile 默认 | `fixed` 或 `self_computed` |

推理运行：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `cuda:0` | 单 GPU 模式设备；torchrun 下由 `LOCAL_RANK` 控制 |
| `--num_workers` | `4` | DataLoader worker 数 |
| `--inference_batch_size` | `1` | 每次 forward 的 patch 数，显存不足时调小 |
| `--time_ps` | `None` | 时间 patch 长度；为空时来自 `dataset/config.py` |
| `--trace_ps` | `None` | trace patch 宽度；必须与训练匹配 |
| `--overlap_ratio` | `0.5` | 滑窗重叠比例 |
| `--missing_eps` | `1e-10` | SEG-Y 全零缺失道判定阈值 |
| `--h5_missing_eps` | `missing_eps` | H5 缺失道判定阈值 |
| `--non_strict_load` | `False` | 关闭 checkpoint strict load |
| `--strict_fill` | `False` | 若未填充或误改已知道则报错 |
| `--no_progress` | `False` | 关闭 tqdm |
| `--sort_output` | `False` | 是否输出排序版 SEG-Y |
| `--backfill_interval` | `0` | 每 N 个 batch 增量写一次 SEG-Y，0 表示结束后统一写 |

模型结构：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_type` | `gated` | 推理模型类型；shell 脚本覆盖为 `trace_axis` |
| `--gated_d_model` | `1080` | gated 模型 d_model |
| `--gated_d_ff` | `2048` | gated FFN 宽度 |
| `--gated_nhead` | `6` | gated attention 头数 |
| `--gated_encoder_layers` | `4` | 推理默认 encoder 层数 |
| `--gated_bottleneck_layers` | `4` | bottleneck 层数 |
| `--gated_decoder_layers` | `4` | decoder 层数 |
| `--gated_encdec_mem_bottleneck_layers` | `2` | gated_encdec memory bottleneck 层数 |
| `--chunk_length_flow` | `256` | gated 时间 chunk 长度 |
| `--chunk_overlap_flow` | `0.5` | gated 时间 chunk 重叠比例 |
| `--use_energy_mlp` | `False` | gated 能量统计 |
| `--use_missing_embedding` | `True` | missing embedding |
| `--headwise_attn_output_gate` | `True` | headwise gate |
| `--elementwise_attn_output_gate` | `False` | elementwise gate |
| `--geom_mode` | `source` | `trace_axis` 几何编码方式 |
| `--use_p_scale` | `False` | 是否启用 p_scale |

Flow Matching 采样：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--path_type` | `Linear` | `Linear`、`GVP`、`VP` |
| `--prediction` | `velocity` | `velocity`、`score`、`noise` |
| `--loss_weight` | `None` | 与训练一致 |
| `--sampling_method` | `ode` | `ode` 或 `sde` |
| `--ode_sampling_method` | `dopri5` | ODE solver |
| `--ode_num_steps` | `50` | ODE 步数 |
| `--ode_atol`, `--ode_rtol` | `1e-6`, `1e-3` | ODE 容差 |
| `--sde_sampling_method` | `Euler` | SDE solver |
| `--sde_num_steps` | `250` | SDE 步数 |

可视化：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--visualize` | `False` | 是否保存 batch 图 |
| `--vis_batches` | `0` | 最多保存多少个 batch，0 表示全部 |

### 5.5 推理回填逻辑

1. 读取 `mask_path`，用 `missing_eps` 找出全零缺失道。
2. 按 `segy_profile.key_columns` 解析 SEG-Y 道头，构建 `{几何键: trace_idx}` lookup。
3. 读取 `h5_mask` 和 `h5_regular`，按 profile sort keys 排序并生成滑动 patch。
4. 对每个 patch 调用 `FlowMatchingModel.sample()`。
5. 对同一几何键的多次 patch 预测求平均。
6. 只写回 `mask_path` 中判定为缺失的 trace，不改动已观测道。
7. 输出 CSV/JSON 报告；若 `strict_fill=true`，发现未填充或误改已观测道会抛错。

## 十、工作流 6：结果核查

建议至少执行以下检查：

```bash
python3 check_segy_headers.py /path/mask.sgy ./gen_fill_results/filled_missing.sgy --check-empty
python3 check_segy_headers.py ./gen_fill_results/filled_missing.sgy ./gen_fill_results/residual.sgy -n 5
```

重点查看：

- `summary.json` 中 `missing_total`、`written`、`unfilled`、`still_missing_after_write` 是否符合预期。
- `observed_changed` 是否为 0。
- `unmatched_geometry_keys` 是否为 0；若不为 0，优先检查 `segy_profile`、`header_mode`、H5 与 SEG-Y 的 trace 排序和几何键字段。
- `vis/` 中预测道与标签、残差和 FK 谱是否有明显异常。

## 十一、目录与脚本说明

| 路径 | 角色 |
| --- | --- |
| `README.md` | 当前工作流说明 |
| `segy_schema.py` | 统一定义 SEG-Y profile、道头字节、排序键和 H5 fallback |
| `check_segy_headers.py` | SEG-Y 单文件检查和双文件对比 |
| `convert_tool/` | V1 单文件/批量 SEG-Y -> H5 工具，profile CLI 较完整 |
| `convert_toolV2/` | V2 三文件转换工具，适合一次转换 irr/mask/label |
| `dataset/config.py` | 训练数据路径和 patch 尺寸配置 |
| `dataset/datasets_interp_v2.py` | 当前主数据集，训练和推理均使用 |
| `dataset/datasets_interp.py` | 早期按测线/炮集组织的数据集 |
| `dataset/datasets.py` | 遗留数据集，包含更多 masking 变体 |
| `dataset/datasets_ovt.py` | OVT 域数据集 |
| `FPM.py` | Flow Matching 模型包装，封装训练 loss 与采样 |
| `transport/` | Flow Matching path、loss、ODE/SDE sampler |
| `models/seisdit_trace_axis.py` | 当前 trace-axis 主模型 |
| `models/gated_seisdit_gen.py` | gated 生成器 |
| `models/gated_seisdit_gen_encdec.py` | gated encoder-decoder 生成器 |
| `models/seisdit_vit_bottleneck.py` | ViT bottleneck 变体 |
| `train_fpmV3_ddp.py` | Accelerate 多卡训练入口 |
| `gen_infer.py` | 当前主推理和 SEG-Y 回填入口 |
| `gen_infer_bak.py` | 旧版/备份推理实现，保留了自定义 H5FillDataset 逻辑 |
| `run_fm_multi_gpu.sh` | 训练启动脚本 |
| `run_gen_infer_sw06.sh` | sw06 推理启动脚本 |
| `run_gen_infer_field1031.sh` | field1031 推理启动脚本 |
| `accelerate_config.yaml` | Accelerate 多卡配置 |

## 十二、常见问题与建议补充

| 问题 | 排查建议 |
| --- | --- |
| H5 与 SEG-Y 回填匹配不上 | 确认转换、训练、推理都使用同一个 `segy_profile` 和 `header_mode`；检查 `key_columns` 对应字段是否存在且无异常 |
| 推理显存不足 | 减小 `--inference_batch_size`、`--trace_ps`，或提高 `NPROC_PER_NODE` 分摊 patch |
| checkpoint load 报 missing/unexpected keys | 确认推理的 `model_type`、gated 结构参数、`use_missing_embedding`、`geom_mode`、`use_p_scale` 与训练一致；必要时临时加 `--non_strict_load` 定位差异 |
| 训练数据路径不生效 | `--h5File` 和 `--h5File_regular` 由 `dataset/config.py` 解析；确认命令参数在 Python 启动时传入，或直接修改配置文件 |
| `field1031` 或 `segc3` 排序异常 | 检查 profile 中的 `sort_keys`、`key_columns`、`byte_pos` 是否符合实际 SEG-Y 道头 |
| `run_*.sh` 默认路径不存在 | 脚本里有大量机器绝对路径，运行前必须用环境变量覆盖或编辑脚本 |
| 依赖安装不明确 | 当前仓库未提供环境文件，建议补充 `requirements.txt` |

建议补充：

- 增加环境文件和最小可运行样例数据说明。
- 将 `dataset/config.py` 中的训练缺失率和 `train_fpmV3_ddp.py` 的硬编码 `(0.3, 0.5)` 统一成 CLI 参数。
- 将 `resultsFPM/..._0517` 中的日期后缀改成可配置参数。
- 为 `convert_toolV2/batch_segy2h5.py` 补充 `--segy-profile` 和 `--mode` 参数，使其与 V1 批处理一致。
