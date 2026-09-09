"""
DiffusionTransformerLightDPPolicy: 简化的 LightDP 策略

特性:
1. 仅保留训练循环逻辑
2. 剪枝逻辑全部委托给 LightDPTransformerForDiffusion 模型
3. 代码简洁，易于维护
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.transformer_for_diffusion_lightdp import LightDPTransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DiffusionTransformerLightDPPolicy(BaseLowdimPolicy):
    """
    简化版 LightDP 策略
    
    训练流程:
    1. Warmup阶段 (前 warmup_epochs): 训练基础模型，不应用剪枝
    2. 剪枝学习阶段: 学习每层的重要性门控
    3. 硬剪枝: 物理移除低重要性层
    4. Fine-tune阶段: 在剪枝后模型上继续训练
    
    所有剪枝逻辑集成在 model (LightDPTransformerForDiffusion) 中
    """
    
    def __init__(
        self,
        model: LightDPTransformerForDiffusion,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 100,
        obs_as_cond: bool = False,
        pred_action_steps_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs
        
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        self.normalizer = LinearNormalizer()
        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_cond else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.target_layers = model.target_layers
        self.num_layers = model.n_layer
        self.warmup_epochs = model.warmup_epochs
        self.svd_rank = model.svd_rank
        
        logger.info(
            f"DiffusionTransformerLightDPPolicy: num_layers={self.num_layers}, "
            f"target_layers={self.target_layers}, warmup_epochs={self.warmup_epochs}"
        )
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测动作"""
        assert 'obs' in obs_dict
        
        was_training = self.training
        self.eval()
        
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim
        
        device = self.device
        dtype = self.dtype
        
        cond = None
        cond_data = None
        cond_mask = None
        
        if self.obs_as_cond:
            cond = nobs[:, :To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
        else:
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True
        
        trajectory = torch.randn(size=shape, dtype=dtype, device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            if not self.obs_as_cond and cond_data is not None:
                trajectory[cond_mask] = cond_data[cond_mask]
            
            model_output = self.model(
                sample=trajectory,
                timestep=t,
                cond=cond,
                use_gating=False
            )
            
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory
            ).prev_sample
        
        if not self.obs_as_cond and cond_data is not None:
            trajectory[cond_mask] = cond_data[cond_mask]
        
        naction_pred = trajectory[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        
        if was_training:
            self.train()
        
        del trajectory, cond, cond_data, cond_mask
        del naction_pred
        
        return {'action': action}
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        hard_prune: bool = False,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        assert 'valid_mask' not in batch
        
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        
        cond = None
        trajectory = action
        
        if self.obs_as_cond:
            cond = obs[:, :self.n_obs_steps, :]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)
        
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )
        
        if self.obs_as_cond:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)
        
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        is_warmup = not self.model._warmup_done
        is_finetune = self.model._hard_prune_done
        
        # 确定是否使用门控
        use_gating = not is_finetune
        
        # 前向传播
        model_output = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            cond=cond,
            use_gating=use_gating
        )
        
        # 计算扩散损失
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        diffusion_loss = F.mse_loss(model_output, target, reduction='none')
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = diffusion_loss.mean()
        
        # 计算门控损失
        if is_finetune or is_warmup:
            total_loss = diffusion_loss
            l1_loss = torch.tensor(0.0, device=trajectory.device)
            binarization_loss = torch.tensor(0.0, device=trajectory.device)
        else:
            gate_soft = self.model.get_gate_scores().to(device=trajectory.device)
            
            binarization_loss = (gate_soft * (1.0 - gate_soft)).mean()
            target_k = float(self.target_layers)
            count_loss = (gate_soft.sum() - target_k) ** 2
            
            importance_loss = torch.tensor(0.0, device=trajectory.device)
            if self.model.layer_importance is not None:
                imp = torch.tensor(
                    self.model.layer_importance,
                    device=gate_soft.device,
                    dtype=gate_soft.dtype
                )
                importance_loss = ((1.0 - gate_soft) * imp).sum()
                del imp
            
            l1_loss = gate_soft.mean()
            
            total_loss = (
                diffusion_loss
                + 1.0 * binarization_loss
                + 0.1 * count_loss
                + 0.5 * importance_loss
                + 0.01 * l1_loss
            )
        
        # 获取门控统计
        gate_np = self.model.get_gate_scores().detach().cpu().numpy()
        retained = int((gate_np > 0.5).sum())
        
        result = {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss.detach(),
            'pruning_loss': l1_loss.detach(),
            'binarization_loss': binarization_loss.detach(),
            'gate_scores': float(np.mean(gate_np)),
            'gate_min': float(np.min(gate_np)),
            'gate_max': float(np.max(gate_np)),
            'gates': gate_np.tolist(),
            'retained_layers': retained
        }
        
        # 清理中间变量
        del noise, timesteps, trajectory, noisy_trajectory, condition_mask, loss_mask
        del model_output, target, obs, action, cond
        del diffusion_loss, l1_loss, binarization_loss
        del gate_np
        if 'gate_soft' in locals():
            del gate_soft
        
        return result
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def get_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        lr = kwargs.get('learning_rate', 1e-4)
        weight_decay = kwargs.get('weight_decay', 1e-3)
        pruning_lr = kwargs.get('pruning_lr', lr * 10)
        
        optim_groups = self.model.get_optim_groups(weight_decay=weight_decay)
        
        if not self.model._hard_prune_done:
            pruning_params = [self.model.gate_logits]
            optim_groups.append({
                'params': pruning_params,
                'lr': pruning_lr,
                'weight_decay': 0.0
            })
        
        optimizer = torch.optim.AdamW(optim_groups, lr=lr)
        return optimizer
    
    def hard_prune(self) -> int:
        """执行硬剪枝（委托给模型）"""
        return self.model.hard_prune()
    
    def compute_layer_importance(self, svd_rank: int = None):
        """计算层重要性（委托给模型）"""
        self.model.compute_layer_importance(svd_rank=svd_rank)
    
    def get_active_gates(self) -> torch.Tensor:
        """获取硬门控决策"""
        scores = self.model.get_gate_scores()
        hard = (scores > 0.5).float()
        return hard
    
    def get_pruning_stats(self) -> Dict:
        """获取剪枝统计"""
        return self.model.get_model_info()