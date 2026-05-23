# Seismic Flow — 5D 地震数据 Flow Matching 重建

基于 **Conditional Flow Matching (CFM)** 的 5D 地震数据缺失道插值/重建工具。支持分布式训练、多种骨干网络和后处理可视化。

## 整体流程

```
SEG-Y 原始数据
    │
    ▼
convert_tool / convert_toolV2 (Segy2H5.py)
    │  SEG-Y → H5 格式转换 + 道序重排 + OVT 计算
    ▼
H5 格式数据集 (train / test)
    │
    ├── train_fpmV3_ddp.py  (训练)
    │      Flow Matching 训练 (Accelerate 多卡)
    │
    └── gen_infer.py        (推理)
           H5 → 模型推理 → 写回 SEG-Y
```

## 目录结构

| 路径 | 说明 |
|------|------|
| `FPM.py` | `FlowMatchingModel` 包装类，封装训练/采样接口 |
| `train_fpmV3_ddp.py` | 主训练脚本（HuggingFace Accelerate） |
| `gen_infer.py` | 推理脚本：缺失道重建 + 写回 SEG-Y + 可视化 |
| `gen_infer_bak.py` | 推理脚本备份 |
| `segy_schema.py` | SEG-Y 道头配置：sw06 / field1031 / segc3 三种 Profile |
| `check_segy_headers.py` | SEG-Y 道头检查与比对工具 |
| `accelerate_config.yaml` | Accelerate 分布式训练配置 |
| `run_fm_multi_gpu.sh` | 训练启动脚本 |
| `run_gen_infer_field1031.sh` | field1031 推理启动脚本 |
| `run_gen_infer_sw06.sh` | sw06 推理启动脚本 |

### convert_tool (V1) / convert_toolV2 (V2)

SEG-Y → H5 格式转换工具。

- **Segy2H5.py**: 单文件转换，支持 `fixed` / `self_computed` 两种道头模式，可选 OVT 域计算
- **batch_segy2h5.py**: 多进程并行批量转换，分两阶段（并行写入临时 H5 → 合并）
- **dataset_config.py**: 配置输入 SEG-Y 文件列表、输出路径、SEG-Y Profile 等
- **run_convert.sh** (V2): 一键转换脚本，支持 `irr + mask + label` 三文件同时转换

道头模式:
- `fixed`: 从 SEG-Y 道头固定字节位置读取测线/炮桩等字段
- `self_computed`: 仅读取炮点/检波点坐标，通过缩放因子反算 line/stake

### dataset/

地震数据加载模块。

- **config.py**: 数据集参数（patch 大小、缺失率范围、数据路径等）
- **datasets_interp_v2.py**: 核心数据集类
  - `DatasetH5_interp`: 训练（随机整道缺失 self-supervised）+ 测试（patch 滑动窗口）
  - `DatasetH5_interp_v2`: 统一归一化版本（全局振幅阈值）
- **datasets.py**: 遗留数据集（含 block/mixed masking 模式）
- **datasets_interp.py**: 早期插值数据集
- **datasets_ovt.py**: OVT 域数据集（按 midpoint/half-offset 分箱）

### transport/

Flow Matching 核心数学引擎（移植自 [SiT](https://github.com/willisma/SiT)）。

- **transport.py**: `Transport` 类（loss 计算、drift/score 函数）、`Sampler` 类
- **path.py**: 三种耦合路径: `ICPlan` (Linear)、`VPCPlan` (VP)、`GVPCPlan` (GVP)
- **integrators.py**: ODE 求解器（torchdiffeq，支持 dopri5/euler/heun）和 SDE 求解器（Euler-Maruyama / Heun）
- **utils.py**: 工具函数

### models/

骨干网络架构。

- **seisdit_trace_axis.py** (80KB): 主模型，基于 trace-axis transformer，带 RoPE 位置编码
- **seisdit_vit_bottleneck.py**: ViT bottleneck 模型
- **gated_seisdit_gen.py**: 门控 SeisDiT 生成器
- **gated_seisdit_gen_encdec.py**: 门控 SeisDiT 编码器-解码器
- **gated_seisdit_gen_bak.py**: 生成器备份
- **gated_transformer_v5.py / v9.py**: 门控 Transformer 变体
- **gated_transformer_v9_encdec.py / gated_transformer_encdec.py**: 编码器-解码器变体
- **fourier_enoder.py**: Fourier 特征编码
- **rope.py**: Rotary Position Embedding (RoPE)

## 训练

```bash
# 使用 accelerate 多卡训练
bash run_fm_multi_gpu.sh

# 或直接调用
accelerate launch --config_file accelerate_config.yaml train_fpmV3_ddp.py \
    --model_name trace_axis \
    --batch_size 1 \
    --lr 1e-4 \
    --epochs 200 \
    --model_type trace_axis \
    --data_type sw06 \
    --segy_profile sw06 \
    --use_p_scale true
```

### 模型类型

- `trace_axis`: 默认，trace-axis transformer
- `gated`: 门控 SeisDiT 生成器
- `gated_encdec`: 门控 SeisDiT 编码器-解码器
- `tp` / `vit`: ViT bottleneck

### Flow Matching 参数

| 参数 | 选项 | 说明 |
|------|------|------|
| `--path_type` | Linear / GVP / VP | 耦合路径类型 |
| `--prediction` | velocity / score / noise | 模型预测目标 |
| `--sampling_method` | ode / sde | 采样方法 |
| `--ode_num_steps` | int | ODE 步数（默认 50） |
| `--sde_num_steps` | int | SDE 步数（默认 250） |

## 推理

```bash
# sw06 数据
bash run_gen_infer_sw06.sh

# field1031 数据
bash run_gen_infer_field1031.sh

# 自定义参数
python3 gen_infer.py \
    --checkpoint /path/to/model.pth \
    --h5_regular /path/to/regular.h5 \
    --h5_mask /path/to/mask.h5 \
    --mask_path /path/to/mask.sgy \
    --output_dir ./results \
    --segy_profile sw06 \
    --model_type trace_axis
```

### 推理输出

- `filled_missing.sgy`: 填充后的 SEG-Y
- `residual.sgy`: 残差 SEG-Y（需提供 `--label_segy`）
- `summary.json`: 详细统计
- `vis/`: 可视化（masked input / prediction / ground truth / FK 谱 / SNR）
- `*_traces.csv`: 填充/未填充/异常道索引

## SEG-Y Profile

三种预定义的道头配置：

| Profile | 适用数据 | 道头模式 | key_columns |
|---------|---------|----------|-------------|
| `sw06` | 东方合成数据 | fixed | shot_line, shot_no, recv_line, recv_no |
| `field1031` | 野外 1031 数据 | fixed | shot_line, shot_stake, recv_line, recv_stake |
| `segc3` | 自计算坐标 | self_computed | 基于坐标缩放反算 |

## 数据转换

```bash
# V2 单次转换（irr + mask + label 三文件）
cd convert_toolV2
./run_convert.sh

# V1 批量转换
cd convert_tool
python batch_segy2h5.py

# 单文件转换
python Segy2H5.py --input-segy /path/to/data.sgy --mode fixed
```

## SEG-Y 检查

```bash
# 查看单文件道头
python check_segy_headers.py data.sgy

# 比对两个文件
python check_segy_headers.py a.sgy b.sgy
```
