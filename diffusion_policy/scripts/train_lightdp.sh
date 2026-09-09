#!/bin/bash

# LightDP训练脚本
export CUDA_VISIBLE_DEVICES=0

python train.py \
  --config-name=train_diffusion_transformer_lightdp_pusht \
  name=train_lightdp_pusht_d6 \
  policy.target_layers=6 \
  lightdp.target_layers=6 \
  lightdp.pruning_epochs=20 \
  training.num_epochs=30