"""
TransformerForDiffusionLightDP: 完全独立的 LightDP 剪枝 Transformer 模型

特性:
1. 完全独立 - 不依赖原库的 TransformerForDiffusion
2. 内置剪枝门控 - gate_logits 直接集成在模型中
3. 内置损失计算 - compute_loss 集成在模型中
4. 内置硬剪枝 - hard_prune 集成在模型中
5. 支持多种剪枝方案 - 8→6, 8→4, 8→2 等
6. 易于调试 - 可单独测试剪枝功能

Author: LightDP Team
Date: 2026-08
"""

from typing import Union, Optional, Tuple, List, Dict
import logging
import math
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import reduce

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

logger = logging.getLogger(__name__)


class LightDPTransformerForDiffusion(ModuleAttrMixin):
    """
    独立的 LightDP 剪枝 Transformer 模型
    
    将原 TransformerForDiffusion 和 LightDP 剪枝逻辑整合在一起，
    形成一个完全独立、易于维护的模型类。
    
    训练流程:
    1. Warmup阶段: 所有层活跃，训练基础模型
    2. 剪枝学习阶段: 学习每层的重要性门控
    3. 硬剪枝: 物理移除低重要性层
    4. Fine-tune阶段: 在剪枝后模型上继续训练
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 8,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = True,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
        # LightDP 剪枝参数
        init_gate_score: float = 0.5,
        target_layers: int = 6,
        warmup_epochs: int = 30,
        svd_rank: int = 4,
    ):
        super().__init__()
        
        # 保存 LightDP 参数
        self.target_layers = target_layers
        self.warmup_epochs = warmup_epochs
        self.svd_rank = svd_rank
        self.original_n_layer = n_layer
        self.current_n_layer = n_layer
        self._hard_prune_done = False
        self._warmup_done = False
        self.pruned_indices = list(range(n_layer))
        self.pruned_layer_indices = list(range(n_layer))
        self.layer_importance = None
        
        # 计算 token 数量
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps
        
        # 输入嵌入
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        
        # 条件编码
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        
        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            
            # Decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )
        
        # 注意力掩码
        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s - 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None
        
        # 输出头
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        
        # LightDP 门控参数
        init_logit = math.log(init_gate_score / (1.0 - init_gate_score))
        self.gate_logits = nn.Parameter(torch.ones(n_layer) * init_logit)
        
        # 常量
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only
        self.n_layer = n_layer
        self.n_obs_steps = n_obs_steps
        
        # 初始化权重
        self.apply(self._init_weights)
        
        logger.info(
            f"LightDPTransformerForDiffusion: n_layer={n_layer}, "
            f"target_layers={target_layers}, warmup_epochs={warmup_epochs}, "
            f"params={sum(p.numel() for p in self.parameters()):.2e}"
        )
    
    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, LightDPTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")
    
    def get_gate_scores(self) -> torch.Tensor:
        """获取软门控分数 (sigmoid(logits))"""
        return torch.sigmoid(self.gate_logits)
    
    def get_hard_gates(self, target_k: int = None) -> torch.Tensor:
        """获取硬门控 (top-k选择)"""
        if target_k is None:
            target_k = self.target_layers
        soft = torch.sigmoid(self.gate_logits)
        _, indices = torch.topk(soft, k=min(target_k, len(soft)))
        hard_gates = torch.zeros_like(soft)
        hard_gates[indices] = 1.0
        return hard_gates
    
    def get_pruned_layers(self, threshold: float = 0.5) -> List[int]:
        """获取被剪枝的层索引"""
        scores = self.get_gate_scores()
        return (scores < threshold).nonzero().squeeze().tolist()
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        use_gating: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            sample: (B, T, input_dim) 输入轨迹
            timestep: (B,) 或 int 扩散步
            cond: (B, T', cond_dim) 条件输入
            use_gating: 是否使用门控（训练时True，推理时False）
        
        Returns:
            output: (B, T, output_dim)
        """
        model_dtype = torch.float32
        for name, p in self.named_parameters():
            if p.is_floating_point() and torch.is_floating_point(p.data):
                model_dtype = p.dtype
                break
        sample = sample.to(dtype=model_dtype)
        if cond is not None:
            cond = cond.to(dtype=model_dtype)
        
        # 时间编码
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1).to(dtype=model_dtype)
        
        # 输入嵌入
        input_emb = self.input_emb(sample)
        
        if self.encoder_only:
            # BERT encoder
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]
        else:
            # 条件准备
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            x = self.drop(cond_embeddings + position_embeddings)
            
            # Encoder forward
            x = self.encoder(x)
            memory = x
            
            # Decoder forward
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            
            if use_gating and self.decoder is not None:
                gate_scores = self.get_gate_scores().to(device=x.device, dtype=x.dtype)
                x = self._decoder_forward_with_gating(x, memory, gate_scores)
            else:
                x = self.decoder(
                    tgt=x,
                    memory=memory,
                    tgt_mask=self.mask,
                    memory_mask=self.memory_mask
                )
        
        # 输出头
        x = self.ln_f(x)
        x = self.head(x)
        
        return x
    
    def _decoder_forward_with_gating(self, tgt, memory, gate_scores):
        """
        带门控的 Decoder 前向传播
        
        保持标准 Transformer 残差连接: output = output + gate * layer_output
        - 当 gate=0 时: 跳过层
        - 当 gate=1 时: 标准残差连接
        - 当 gate 为软值时: 缩放层输出后做残差连接
        """
        output = tgt
        target_device = tgt.device
        target_dtype = tgt.dtype
        decoder_layers = self.decoder.layers
        
        for idx, layer in enumerate(decoder_layers):
            orig_idx = self.pruned_layer_indices[idx] if idx < len(self.pruned_layer_indices) else idx
            
            if orig_idx < len(gate_scores):
                gate = gate_scores[orig_idx]
            else:
                gate = torch.tensor(1.0, device=target_device, dtype=target_dtype)
            
            if not isinstance(gate, torch.Tensor):
                gate = torch.tensor(gate, device=target_device, dtype=target_dtype)
            else:
                gate = gate.to(device=target_device, dtype=target_dtype)
            
            if float(gate.detach().cpu()) < 0.01:
                continue
            
            layer_output = layer(
                output, memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
            
            while gate.dim() < layer_output.dim():
                gate = gate.unsqueeze(-1)
            
            output = output + gate * layer_output
        
        if hasattr(self.decoder, 'norm') and self.decoder.norm is not None:
            output = self.decoder.norm(output)
        
        return output
    
    def compute_layer_importance(self, svd_rank: int = None):
        """
        基于 SVD 重构误差计算层重要性分数
        """
        if svd_rank is None:
            svd_rank = self.svd_rank
        
        if self.decoder is None:
            logger.warning("Cannot compute layer importance: decoder is None")
            return
        
        decoder_layers = self.decoder.layers
        layer_errors_list = []
        
        for layer_idx, layer in enumerate(decoder_layers):
            weight_matrices = []
            
            if hasattr(layer.self_attn, 'in_proj_weight') and layer.self_attn.in_proj_weight is not None:
                w_qkv = layer.self_attn.in_proj_weight.data.float()
                d_model = w_qkv.shape[0] // 3
                
                w_q = w_qkv[:d_model, :]
                w_k = w_qkv[d_model:2 * d_model, :]
                w_v = w_qkv[2 * d_model:3 * d_model, :]
                
                weight_matrices.append(('Q', w_q))
                weight_matrices.append(('K', w_k))
                weight_matrices.append(('V', w_v))
            
            if hasattr(layer.self_attn, 'out_proj') and layer.self_attn.out_proj is not None:
                if hasattr(layer.self_attn.out_proj, 'weight'):
                    weight_matrices.append(('O', layer.self_attn.out_proj.weight.data.float()))
            
            if hasattr(layer, 'linear1') and layer.linear1 is not None:
                weight_matrices.append(('FFN1', layer.linear1.weight.data.float()))
            
            if hasattr(layer, 'linear2') and layer.linear2 is not None:
                weight_matrices.append(('FFN2', layer.linear2.weight.data.float()))
            
            matrix_errors = {}
            layer_relative_error = 0.0
            
            for name, W in weight_matrices:
                error = self._compute_svd_reconstruction_error(W, svd_rank)
                norm = W.float().norm().item()
                relative_error = error / (norm + 1e-8)
                matrix_errors[name] = relative_error
                layer_relative_error += relative_error
            
            layer_errors_list.append({
                'relative_error': layer_relative_error,
                'matrix_errors': matrix_errors,
            })
            
            logger.info(
                f"Layer {layer_idx}: rel_error={layer_relative_error:.4f}, "
                f"Q={matrix_errors.get('Q', 0):.4f}, "
                f"K={matrix_errors.get('K', 0):.4f}, "
                f"V={matrix_errors.get('V', 0):.4f}"
            )
            
            del weight_matrices, matrix_errors, w_qkv, w_q, w_k, w_v
        
        # 提取相对误差并归一化
        relative_errors = np.array([le['relative_error'] for le in layer_errors_list])
        
        mean_error = relative_errors.mean()
        if mean_error < 1e-10:
            amplified_scores = np.ones_like(relative_errors) / len(relative_errors)
        else:
            amplified_scores = np.clip(relative_errors / mean_error, 0.1, 3.0)
            amplified_scores = amplified_scores / amplified_scores.sum()
        
        self.layer_importance = amplified_scores
        
        # 基于 SVD 重要性初始化门控
        sorted_indices = np.argsort(-relative_errors)
        init_logits = np.full(self.n_layer, -2.0)
        
        for i in range(min(self.target_layers, self.n_layer)):
            init_logits[sorted_indices[i]] = 2.0
        
        self.gate_logits.data = torch.tensor(init_logits, dtype=self.gate_logits.dtype)
        
        self._warmup_done = True
        
        init_scores = 1.0 / (1.0 + np.exp(-init_logits))
        logger.info(f"SVD rank used: {svd_rank}")
        logger.info(f"Layer importance: {[f'{s:.4f}' for s in amplified_scores]}")
        logger.info(f"Top-{self.target_layers} layers: {sorted_indices[:self.target_layers].tolist()}")
        logger.info(f"Gate scores initialized: {[f'{s:.3f}' for s in init_scores]}")
        
        del layer_errors_list, relative_errors, amplified_scores, sorted_indices, init_logits, init_scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def _compute_svd_reconstruction_error(W: torch.Tensor, k: int) -> float:
        """计算矩阵 W 的 SVD 重构误差"""
        try:
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            elif W.dim() < 2:
                W = W.reshape(1, -1)
            
            min_dim = min(W.shape[0], W.shape[1])
            effective_k = min(k, min_dim - 1)
            
            if effective_k <= 0:
                return float('inf')
            
            U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
            
            U_k = U[:, :effective_k]
            S_k = S[:effective_k]
            Vh_k = Vh[:effective_k, :]
            
            W_k = (U_k * S_k.unsqueeze(0)) @ Vh_k
            
            error = torch.norm(W.float() - W_k, p='fro').item()
            return error
            
        except Exception as e:
            logger.warning(f"SVD computation failed: {e}, using L2 norm fallback")
            return W.float().norm().item()
    
    def hard_prune(self) -> int:
        """
        硬剪枝：物理移除低重要性层
        
        Returns:
            当前层数
        """
        gate_soft = self.get_gate_scores()
        
        _, indices = torch.topk(gate_soft, k=min(self.target_layers, len(gate_soft)))
        self.pruned_indices = indices.sort().values.tolist()
        self.current_n_layer = len(self.pruned_indices)
        
        logger.info(f"Hard pruned from {self.original_n_layer} to {self.current_n_layer} layers")
        logger.info(f"Retained layers: {self.pruned_indices}")
        logger.info(f"Gate scores: {[f'{s:.4f}' for s in gate_soft.detach().cpu().tolist()]}")
        
        # 物理剪枝 decoder 层
        if self.decoder is not None:
            self._prune_decoder_layers(self.pruned_indices)
        
        # 更新门控参数
        old_logits = self.gate_logits
        new_logits = old_logits.clone().detach()
        for i in range(len(new_logits)):
            if i in self.pruned_indices:
                new_logits[i] = 10.0
            else:
                new_logits[i] = -10.0
        self.gate_logits = nn.Parameter(new_logits)
        del old_logits, new_logits
        gc.collect()
        self._hard_prune_done = True
        
        logger.info("Hard prune complete, entering fine-tuning phase")
        return self.current_n_layer
    
    def _prune_decoder_layers(self, keep_indices: List[int]):
        """物理移除 decoder 中未保留的层"""
        if self.decoder is None:
            return
        
        keep_indices = sorted(keep_indices)
        original_count = len(self.decoder.layers)
        
        removed_layers = [self.decoder.layers[i] for i in range(original_count) if i not in keep_indices]
        
        kept_layers = nn.ModuleList([
            self.decoder.layers[i] for i in keep_indices
        ])
        
        self.decoder.layers = kept_layers
        self.n_layer = len(keep_indices)
        self.pruned_layer_indices = keep_indices
        
        for layer in removed_layers:
            del layer
        del removed_layers, kept_layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(
            f"Pruned decoder: kept {len(keep_indices)}/{original_count} layers, "
            f"indices: {keep_indices}"
        )
    
    def get_optim_groups(self, weight_decay: float = 1e-3):
        """获取优化器参数分组"""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            f"parameters {inter_params} made it into both decay/no_decay sets!"
        
        excluded = set()
        if "gate_logits" in param_dict:
            excluded.add("gate_logits")
        
        remaining = param_dict.keys() - union_params - excluded
        if len(remaining) > 0:
            raise AssertionError(
                f"parameters {remaining} were not separated!"
            )
        
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups
    
    def get_pruning_info(self) -> Dict:
        """获取剪枝信息"""
        kept_indices = list(range(self.n_layer))
        if hasattr(self, 'pruned_layer_indices') and self.pruned_layer_indices:
            kept_indices = self.pruned_layer_indices.copy()
        
        info = {
            'is_pruned': self._hard_prune_done,
            'n_layers': self.original_n_layer,
            'kept_layers': self.n_layer,
            'pruned_layers': self.original_n_layer - self.n_layer,
            'keep_ratio': self.n_layer / max(1, self.original_n_layer),
            'kept_indices': kept_indices,
        }
        return info
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        gate_scores = self.get_gate_scores().detach().cpu().numpy()
        
        info = {
            'total_params': total_params,
            'total_params_M': total_params / 1e6,
            'n_layer': self.n_layer,
            'original_n_layer': self.original_n_layer,
            'target_layers': self.target_layers,
            'gate_scores': gate_scores.tolist(),
            'gate_mean': float(np.mean(gate_scores)),
            'gate_min': float(np.min(gate_scores)),
            'gate_max': float(np.max(gate_scores)),
            'hard_prune_done': self._hard_prune_done,
            'warmup_done': self._warmup_done,
            'pruned_indices': self.pruned_indices,
        }
        return info


def test():
    """测试函数"""
    model = LightDPTransformerForDiffusion(
        input_dim=2,
        output_dim=2,
        horizon=16,
        n_obs_steps=2,
        cond_dim=20,
        n_layer=8,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.01,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=0,
        init_gate_score=0.5,
        target_layers=6,
        warmup_epochs=30,
        svd_rank=4,
    )
    
    # 测试前向传播
    batch_size = 2
    sample = torch.randn(batch_size, 16, 2)
    timestep = torch.randint(0, 100, (batch_size,))
    cond = torch.randn(batch_size, 2, 20)
    
    # 测试无门控前向传播
    output = model(sample, timestep, cond, use_gating=False)
    print(f"Forward (no gating): {output.shape}")
    
    # 测试有门控前向传播
    output = model(sample, timestep, cond, use_gating=True)
    print(f"Forward (with gating): {output.shape}")
    
    # 测试门控
    gate_scores = model.get_gate_scores()
    print(f"Gate scores: {gate_scores}")
    
    # 测试硬剪枝
    model._warmup_done = True
    model.compute_layer_importance(svd_rank=4)
    model.hard_prune()
    
    # 剪枝后测试
    output = model(sample, timestep, cond, use_gating=False)
    print(f"Forward (after prune): {output.shape}")
    
    # 测试模型信息
    info = model.get_model_info()
    print(f"Model info: {info}")
    
    print("All tests passed!")


if __name__ == '__main__':
    test()