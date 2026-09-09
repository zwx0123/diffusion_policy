#!/bin/bash
# LightDP训练脚本
# 用法: bash scripts/train_lightdp.sh [目标层数]
# 示例: bash scripts/train_lightdp.sh 6
# 训练阶段: warmup(30) -> pruning(80) -> finetune(150) = 260 epochs

set -e

TARGET_LAYERS=${1:-6}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

echo "============================================"
echo "LightDP Training Script"
echo "  Target layers: $TARGET_LAYERS"
echo "  Project dir:  $PROJECT_DIR"
echo "  Epochs:        260 (warmup 30 + pruning 80 + finetune 150)"
echo "============================================"

python train.py \
  --config-name=train_diffusion_transformer_lightdp_pusht \
  name="lightdp_pusht_${TARGET_LAYERS}layer" \
  lightdp.target_layers=${TARGET_LAYERS}

echo "============================================"
echo "Training completed for ${TARGET_LAYERS}-layer pruning."
echo "============================================"