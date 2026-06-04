#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3

# 训练参数
# 全开（默认）                                                                                                                                                
  #bash run_ddpm_multi_gpu.sh --model_type gated ...                                                                                                               
                                                                                                                                                                  
  # 消融能量感知                                                                                                                                                  
  #bash run_ddpm_multi_gpu.sh --model_type gated --gated_use_energy_stats false --gated_use_structural_mask true --gated_use_missing_embed true                    
                                                                                                                                                                  
  # 消融结构性注意力                                                                                                                                              
  #bash run_ddpm_multi_gpu.sh --model_type gated --gated_use_energy_stats true --gated_use_structural_mask false --gated_use_missing_embed true
                                                                                                                                                                  
  # 消融缺失嵌入                                                                                                                                                
  #bash run_ddpm_multi_gpu.sh --model_type gated --gated_use_energy_stats true --gated_use_structural_mask true --gated_use_missing_embed false
                                                                                                                                                                  
  # 全关（基线）
  #bash run_ddpm_multi_gpu.sh --model_type gated --gated_use_energy_stats false --gated_use_structural_mask false --gated_use_missing_embed false                  
                                                                                                                                                       


MODEL_NAME="trace_axis"
BATCH_SIZE=8
LR=1e-4
EPOCHS=200
MODEL_TYPE="trace_axis"  # trace_axis | gated | tp
SEED=515
DATA_TYPE="field1031"
SEGY_PROFILE="${SEGY_PROFILE:-field1031}"  # sw06 | field1031 | segc3
USE_P_SCALE="${USE_P_SCALE:-true}"
USE_MISSING_EMBEDDING="${USE_MISSING_EMBEDDING:-false}"
USE_ENERGY_MLP="${USE_ENERGY_MLP:-false}"
GEOM_MODE="${GEOM_MODE:-source}"  # source | receiver | relative
HEADWISE_ATTN_OUTPUT_GATE="${HEADWISE_ATTN_OUTPUT_GATE:-true}"
ELEMENTWISE_ATTN_OUTPUT_GATE="${ELEMENTWISE_ATTN_OUTPUT_GATE:-false}"
PATH_TYPE="${PATH_TYPE:-Linear}"  # Linear | GVP | VP
PREDICTION="${PREDICTION:-velocity}"  # velocity | score | noise
LOSS_WEIGHT="${LOSS_WEIGHT:-logitnormal}"  # None | velocity | likelihood | logitnormal
USE_MULTISCALE_LOSS="${USE_MULTISCALE_LOSS:-true}"
MULTISCALE_LOSS_WEIGHT="${MULTISCALE_LOSS_WEIGHT:-0.1}"
SAMPLING_METHOD="${SAMPLING_METHOD:-ode}"  # ode | sde
ODE_NUM_STEPS="${ODE_NUM_STEPS:-50}"
SDE_NUM_STEPS="${SDE_NUM_STEPS:-250}"
RESUME="${RESUME:-}"  # set to checkpoint path to resume
PRETRAINED="${PRETRAINED:-}"  # set to pretrained checkpoint path for weight initialization
PRETRAINED_STRICT="${PRETRAINED_STRICT:-true}"
echo "======================================"
echo "开始使用 Accelerate 进行分布式训练"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCHS"
echo "Model Type: $MODEL_TYPE"
echo "Data Type: $DATA_TYPE"
echo "segy_profile: $SEGY_PROFILE"
echo "geom_mode: $GEOM_MODE"
echo "use_p_scale: $USE_P_SCALE"
echo "loss_weight: $LOSS_WEIGHT"
echo "use_multiscale_loss: $USE_MULTISCALE_LOSS"
echo "pretrained: ${PRETRAINED:-none}"
echo "pretrained_strict: $PRETRAINED_STRICT"
echo "======================================"

# 方法1: 使用配置文件（推荐，端口自动选择）
accelerate launch --config_file accelerate_config.yaml --main_process_port 29501 train_fpmV3_ddp.py \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --model_type $MODEL_TYPE \
    --seed $SEED \
    --data_type $DATA_TYPE \
    --segy_profile $SEGY_PROFILE \
    --use_p_scale $USE_P_SCALE \
    --use_missing_embedding $USE_MISSING_EMBEDDING \
    --use_energy_mlp $USE_ENERGY_MLP \
    --geom_mode $GEOM_MODE \
    --headwise_attn_output_gate $HEADWISE_ATTN_OUTPUT_GATE \
    --elementwise_attn_output_gate $ELEMENTWISE_ATTN_OUTPUT_GATE \
    --path_type $PATH_TYPE \
    --prediction $PREDICTION \
    --loss_weight $LOSS_WEIGHT \
    --use_multiscale_loss $USE_MULTISCALE_LOSS \
    --multiscale_loss_weight $MULTISCALE_LOSS_WEIGHT \
    --sampling_method $SAMPLING_METHOD \
    --ode_num_steps $ODE_NUM_STEPS \
    --sde_num_steps $SDE_NUM_STEPS \
    $([ -n "${RESUME}" ] && echo "--resume ${RESUME}") \
    $([ -n "${PRETRAINED}" ] && echo "--pretrained ${PRETRAINED} --pretrained_strict ${PRETRAINED_STRICT}")

# 方法2: 直接指定参数（无需配置文件，自动选择端口）
# accelerate launch --num_processes=2 --num_machines=1 --mixed_precision=no --main_process_port=0 train_ddpmV3_ddp.py \
#     --model_name $MODEL_NAME \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --epochs $EPOCHS \
#     --data_type $DATA_TYPE \
#     --seed $SEED

# 方法3: 指定特定端口（如果端口冲突）
# accelerate launch --num_processes=2 --num_machines=1 --mixed_precision=no --main_process_port=29501 train_ddpmV3_ddp.py \
#     --model_name $MODEL_NAME \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --epochs $EPOCHS \
#     --data_type $DATA_TYPE \
#     --seed $SEED

echo "训练完成！"