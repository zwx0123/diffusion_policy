"""
LightDP - 可学习剪枝的 Diffusion Transformer Policy

基于 TransformerForDiffusion 实现，添加：
1. 可学习的门控参数（gate score）
2. Gumbel-Sigmoid 软剪枝
3. 硬剪枝和微调支持
4. SVD 初始化支持
"""

from typing import Union, Optional, Tuple, Dict, List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class GumbelSigmoidGate(nn.Module):
    """
    可学习的门控层，使用 Gumbel-Sigmoid 进行软剪枝
    
    Args:
        init_score: 初始门控得分
        temperature: Gumbel-Softmax 温度
        temperature_min: 最小温度（用于退火）
        temperature_decay: 温度衰减率
    """
    
    def __init__(
        self,
        init_score: float = 0.5,
        temperature: float = 0.5,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.999
    ):
        super().__init__()
        # 可学习的门控参数
        self.gate_score = nn.Parameter(torch.tensor(init_score))
        
        # 温度参数
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        
        # 记录统计信息
        self.register_buffer('mask_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))
        
    def get_temperature(self):
        """获取当前温度"""
        return max(self.temperature, self.temperature_min)
    
    def update_temperature(self):
        """温度退火"""
        self.temperature = max(
            self.temperature * self.temperature_decay,
            self.temperature_min
        )
    
    def forward(self, training: bool = True, hard: bool = False) -> torch.Tensor:
        """
        获取门控掩码
        
        Args:
            training: 是否训练模式
            hard: 是否使用硬掩码（推理时使用）
        
        Returns:
            mask: 门控掩码，范围 [0, 1]
        """
        if hard:
            # 硬掩码：根据 gate_score 决定保留或丢弃
            return (self.gate_score > 0).float()
        
        if not training:
            # 推理时使用软掩码（sigmoid）
            return torch.sigmoid(self.gate_score / self.get_temperature())
        
        # 训练时使用 Gumbel-Sigmoid
        eps = 1e-20
        
        # 采样两个独立的 Gumbel 噪声
        uniform1 = torch.rand_like(self.gate_score).clamp(eps, 1 - eps)
        uniform2 = torch.rand_like(self.gate_score).clamp(eps, 1 - eps)
        
        gumbel1 = -torch.log(-torch.log(uniform1))
        gumbel2 = -torch.log(-torch.log(uniform2))
        
        # Gumbel-Softmax for binary
        logits = torch.stack([
            self.gate_score + gumbel1,  # 保留
            gumbel2  # 丢弃
        ], dim=0)
        
        soft_mask = F.softmax(logits / self.get_temperature(), dim=0)
        
        # 返回保留概率
        mask = soft_mask[0]
        
        # 记录历史
        with torch.no_grad():
            idx = self.history_idx.item() % 100
            self.mask_history[idx] = mask.item()
            self.history_idx += 1
        
        return mask


class PrunableTransformerEncoder(nn.Module):
    """
    可剪枝的 Transformer Encoder
    
    包装原始 encoder 的每一层，添加门控机制
    """
    
    def __init__(
        self,
        encoder: nn.TransformerEncoder,
        init_gate_score: float = 0.5,
        gumbel_temperature: float = 0.5,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.999
    ):
        super().__init__()
        
        # 保存原始 encoder
        self.encoder = encoder
        self.num_layers = len(encoder.layers)
        
        # 为每一层创建门控
        self.gates = nn.ModuleList([
            GumbelSigmoidGate(
                init_score=init_gate_score,
                temperature=gumbel_temperature,
                temperature_min=temperature_min,
                temperature_decay=temperature_decay
            )
            for _ in range(self.num_layers)
        ])
        
        # 记录每层是否被剪枝
        self.register_buffer('pruned_mask', torch.zeros(self.num_layers, dtype=torch.bool))
        
        # 记录每层的 mask 值（用于监控）
        self.last_masks = [0.0] * self.num_layers
        
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，逐层应用门控
        
        Args:
            src: 输入序列 [B, T, C]
            mask: 注意力掩码
        
        Returns:
            输出序列 [B, T, C]
        """
        output = src
        
        for idx, (layer, gate) in enumerate(zip(self.encoder.layers, self.gates)):
            if self.pruned_mask[idx]:
                # 已剪枝的层执行恒等映射
                continue
            
            # 计算该层输出
            layer_output = layer(output, src_mask=mask)
            
            # 获取门控掩码
            gate_mask = gate(training=self.training, hard=False)
            
            # 记录 mask 值
            self.last_masks[idx] = gate_mask.item() if gate_mask.dim() == 0 else gate_mask.mean().item()
            
            # 应用门控：mask * layer_out + (1 - mask) * output
            # 扩展 mask 维度
            if gate_mask.dim() == 0:
                gate_mask = gate_mask.unsqueeze(0)
            while gate_mask.dim() < output.dim():
                gate_mask = gate_mask.unsqueeze(-1)
            
            output = gate_mask * layer_output + (1 - gate_mask) * output
        
        return output
    
    def prune_layers(self, keep_indices: List[int]):
        """硬剪枝：只保留指定索引的层"""
        for idx in range(self.num_layers):
            if idx not in keep_indices:
                self.pruned_mask[idx] = True
                logger.info(f"Pruned encoder layer {idx}")
        
        logger.info(f"Encoder: kept {len(keep_indices)}/{self.num_layers} layers")
    
    def get_gate_scores(self) -> List[float]:
        """获取所有层的门控得分"""
        return [gate.gate_score.item() for gate in self.gates]
    
    def get_active_layers(self) -> int:
        """获取活跃层数"""
        return sum(1 for i in range(self.num_layers) if not self.pruned_mask[i])
    
    def update_temperatures(self):
        """更新所有门控的温度"""
        for gate in self.gates:
            gate.update_temperature()


class PrunableTransformerDecoder(nn.Module):
    """
    可剪枝的 Transformer Decoder
    """
    
    def __init__(
        self,
        decoder: nn.TransformerDecoder,
        init_gate_score: float = 0.5,
        gumbel_temperature: float = 0.5,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.999
    ):
        super().__init__()
        
        self.decoder = decoder
        self.num_layers = len(decoder.layers)
        
        self.gates = nn.ModuleList([
            GumbelSigmoidGate(
                init_score=init_gate_score,
                temperature=gumbel_temperature,
                temperature_min=temperature_min,
                temperature_decay=temperature_decay
            )
            for _ in range(self.num_layers)
        ])
        
        self.register_buffer('pruned_mask', torch.zeros(self.num_layers, dtype=torch.bool))
        self.last_masks = [0.0] * self.num_layers
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播，逐层应用门控
        """
        output = tgt
        
        for idx, (layer, gate) in enumerate(zip(self.decoder.layers, self.gates)):
            if self.pruned_mask[idx]:
                continue
            
            layer_output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )
            
            gate_mask = gate(training=self.training, hard=False)
            self.last_masks[idx] = gate_mask.item() if gate_mask.dim() == 0 else gate_mask.mean().item()
            
            if gate_mask.dim() == 0:
                gate_mask = gate_mask.unsqueeze(0)
            while gate_mask.dim() < output.dim():
                gate_mask = gate_mask.unsqueeze(-1)
            
            output = gate_mask * layer_output + (1 - gate_mask) * output
        
        return output
    
    def prune_layers(self, keep_indices: List[int]):
        """硬剪枝"""
        for idx in range(self.num_layers):
            if idx not in keep_indices:
                self.pruned_mask[idx] = True
                logger.info(f"Pruned decoder layer {idx}")
        
        logger.info(f"Decoder: kept {len(keep_indices)}/{self.num_layers} layers")
    
    def get_gate_scores(self) -> List[float]:
        return [gate.gate_score.item() for gate in self.gates]
    
    def get_active_layers(self) -> int:
        return sum(1 for i in range(self.num_layers) if not self.pruned_mask[i])
    
    def update_temperatures(self):
        for gate in self.gates:
            gate.update_temperature()


class TransformerForDiffusionLightDP(ModuleAttrMixin):
    """
    LightDP 版本的 TransformerForDiffusion
    
    在原始模型基础上添加可学习剪枝机制
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
        # LightDP 参数
        init_gate_score: float = 0.5,
        gumbel_temperature: float = 0.5,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.999,
        use_svd_init: bool = False,
        svd_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        
        # 保存 LightDP 参数
        self.init_gate_score = init_gate_score
        self.gumbel_temperature = gumbel_temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.use_svd_init = use_svd_init
        self.svd_init_scale = svd_init_scale
        
        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps
        
        # timing attributes
        self.enable_timing = False
        self.timing_stats = defaultdict(list)
        
        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        
        # cond encoder
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
                # 使用可剪枝的 encoder
                self.encoder = PrunableTransformerEncoder(
                    encoder=nn.TransformerEncoder(
                        encoder_layer=encoder_layer,
                        num_layers=n_cond_layers
                    ),
                    init_gate_score=init_gate_score,
                    gumbel_temperature=gumbel_temperature,
                    temperature_min=temperature_min,
                    temperature_decay=temperature_decay
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            
            # decoder - 使用可剪枝的 decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.decoder = PrunableTransformerDecoder(
                decoder=nn.TransformerDecoder(
                    decoder_layer=decoder_layer,
                    num_layers=n_layer
                ),
                init_gate_score=init_gate_score,
                gumbel_temperature=gumbel_temperature,
                temperature_min=temperature_min,
                temperature_decay=temperature_decay
            )
        else:
            # encoder only BERT
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
            # 使用可剪枝的 encoder
            self.encoder = PrunableTransformerEncoder(
                encoder=nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_layer
                ),
                init_gate_score=init_gate_score,
                gumbel_temperature=gumbel_temperature,
                temperature_min=temperature_min,
                temperature_decay=temperature_decay
            )
        
        # attention mask
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
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only
        
        # init
        self.apply(self._init_weights)
        
        # 如果使用 SVD 初始化
        if use_svd_init:
            self._svd_init_gate_scores()
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        logger.info(
            "prunable layers: encoder=%d, decoder=%d",
            len(self.encoder.gates) if hasattr(self.encoder, 'gates') else 0,
            len(self.decoder.gates) if hasattr(self.decoder, 'gates') else 0
        )
    
    def enable_cuda_timing(self):
        self.enable_timing = True
        self.timing_stats.clear()
    
    def disable_cuda_timing(self):
        self.enable_timing = False
    
    def get_timing_stats(self):
        if not self.timing_stats:
            return {}
        stats = {}
        for block_name, times in self.timing_stats.items():
            if times:
                times_array = np.array(times)
                stats[block_name] = {
                    'mean_ms': float(np.mean(times_array)),
                    'std_ms': float(np.std(times_array)),
                    'min_ms': float(np.min(times_array)),
                    'max_ms': float(np.max(times_array)),
                    'count': len(times_array)
                }
        return stats
    
    def _record_time(self, block_name, start_event, end_event):
        if self.enable_timing and start_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            self.timing_stats[block_name].append(elapsed_time)
    
    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential,
                        GumbelSigmoidGate,
                        PrunableTransformerEncoder,
                        PrunableTransformerDecoder)
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
        elif isinstance(module, TransformerForDiffusionLightDP):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def _svd_init_gate_scores(self):
        """
        使用 SVD 重构误差初始化门控得分
        误差越大说明该层越重要，初始得分越高
        """
        logger.info("Initializing gate scores using SVD...")
        
        with torch.no_grad():
            # 对 encoder 的每一层
            if hasattr(self.encoder, 'gates'):
                for idx, (layer, gate) in enumerate(zip(self.encoder.encoder.layers, self.encoder.gates)):
                    # 获取该层的主要权重矩阵
                    # 对于 TransformerEncoderLayer，主要是 self_attn 和 linear 层
                    importance = self._compute_layer_importance(layer)
                    # 归一化并设置初始得分
                    gate.gate_score.data = torch.tensor(
                        self.svd_init_scale * importance,
                        device=gate.gate_score.device
                    )
            
            # 对 decoder 的每一层
            if hasattr(self.decoder, 'gates'):
                for idx, (layer, gate) in enumerate(zip(self.decoder.decoder.layers, self.decoder.gates)):
                    importance = self._compute_layer_importance(layer)
                    gate.gate_score.data = torch.tensor(
                        self.svd_init_scale * importance,
                        device=gate.gate_score.device
                    )
    
    def _compute_layer_importance(self, layer) -> float:
        """
        计算层的重要性（基于权重矩阵的 SVD 奇异值）
        """
        importance = 0.0
        num_weights = 0
        
        for name, param in layer.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # 计算奇异值
                try:
                    singular_values = torch.linalg.svdvals(param.data)
                    # 使用最大奇异值作为重要性指标
                    importance += singular_values[0].item()
                    num_weights += 1
                except:
                    pass
        
        if num_weights > 0:
            importance /= num_weights
        
        return importance
    
    def get_optim_groups(self, weight_decay: float = 1e-3):
        """获取优化器参数组，门控参数使用不同的学习率"""
        decay = set()
        no_decay = set()
        gate_params = set()
        
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                
                # 门控参数特殊处理
                if 'gate' in fpn.lower() and 'score' in fpn.lower():
                    gate_params.add(fpn)
                elif pn.endswith("bias"):
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
        
        # 验证参数分类
        inter_params = decay & no_decay
        union_params = decay | no_decay | gate_params
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated!" % (str(param_dict.keys() - union_params),)
        
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
                "lr_scale": 1.0
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
                "lr_scale": 1.0
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(gate_params))],
                "weight_decay": 0.0,
                "lr_scale": 10.0  # 门控参数使用更大的学习率
            },
        ]
        
        return optim_groups
    
    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-3,
                             betas: Tuple[float, float] = (0.9, 0.95),
                             gate_lr_scale: float = 10.0):
        """配置优化器"""
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        
        # 调整门控参数的学习率
        for group in optim_groups:
            if 'lr_scale' in group:
                group['lr'] = learning_rate * group['lr_scale']
                del group['lr_scale']
            else:
                group['lr'] = learning_rate
        
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        
        return optimizer
    
    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                cond: Optional[torch.Tensor] = None,
                **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        model_dtype = self.input_emb.weight.dtype
        sample = sample.to(dtype=model_dtype)
        if cond is not None:
            cond = cond.to(dtype=model_dtype)
        
        # Block 1: time encoding
        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start1.record()
        
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1).to(dtype=model_dtype)
        
        self._record_time('1_Time_Encoding', start1, end1)
        
        # Block 2: input embedding
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start2.record()
        
        input_emb = self.input_emb(sample)
        
        self._record_time('2_Input_Embedding', start2, end2)
        
        if self.encoder_only:
            # Block 3: BERT encoder (prunable)
            start3 = torch.cuda.Event(enable_timing=True)
            end3 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start3.record()
            
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]
            
            self._record_time('3_Encoder_Only', start3, end3)
        else:
            # Block 3: condition preparation
            start3 = torch.cuda.Event(enable_timing=True)
            end3 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start3.record()
            
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            x = self.drop(cond_embeddings + position_embeddings)
            
            self._record_time('3_Condition_Prep', start3, end3)
            
            # Block 4: encoder forward (prunable)
            start4 = torch.cuda.Event(enable_timing=True)
            end4 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start4.record()
            
            x = self.encoder(x)
            memory = x
            
            self._record_time('4_Encoder_Forward', start4, end4)
            
            # Block 5: decoder forward (prunable)
            start5 = torch.cuda.Event(enable_timing=True)
            end5 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start5.record()
            
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
            
            self._record_time('5_Decoder_Forward', start5, end5)
        
        # Block 6: output head
        start6 = torch.cuda.Event(enable_timing=True)
        end6 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start6.record()
        
        x = self.ln_f(x)
        x = self.head(x)
        
        self._record_time('6_Output_Head', start6, end6)
        
        return x
    
    # ===== LightDP 特定方法 =====
    
    def get_all_gate_scores(self) -> Dict[str, List[float]]:
        """获取所有层的门控得分"""
        scores = {}
        
        if hasattr(self.encoder, 'get_gate_scores'):
            scores['encoder'] = self.encoder.get_gate_scores()
        
        if hasattr(self.decoder, 'get_gate_scores'):
            scores['decoder'] = self.decoder.get_gate_scores()
        
        return scores
    
    def get_all_last_masks(self) -> Dict[str, List[float]]:
        """获取最近一次前向传播的掩码值"""
        masks = {}
        
        if hasattr(self.encoder, 'last_masks'):
            masks['encoder'] = self.encoder.last_masks
        
        if hasattr(self.decoder, 'last_masks'):
            masks['decoder'] = self.decoder.last_masks
        
        return masks
    
    def prune_layers(self, target_layers: int, strategy: str = 'global'):
        """
        硬剪枝
        
        Args:
            target_layers: 目标保留层数
            strategy: 剪枝策略
                - 'global': 全局排序，保留得分最高的层
                - 'per_module': 每个模块独立剪枝
        """
        logger.info(f"Pruning to {target_layers} layers using {strategy} strategy")
        
        if strategy == 'global':
            # 收集所有层的得分
            all_scores = []
            layer_info = []
            
            if hasattr(self.encoder, 'get_gate_scores'):
                for idx, score in enumerate(self.encoder.get_gate_scores()):
                    all_scores.append(score)
                    layer_info.append(('encoder', idx))
            
            if hasattr(self.decoder, 'get_gate_scores'):
                for idx, score in enumerate(self.decoder.get_gate_scores()):
                    all_scores.append(score)
                    layer_info.append(('decoder', idx))
            
            # 选择得分最高的 target_layers 层
            keep_indices = np.argsort(all_scores)[-target_layers:]
            
            # 分别记录 encoder 和 decoder 要保留的层
            encoder_keep = []
            decoder_keep = []
            
            for idx in keep_indices:
                module_type, layer_idx = layer_info[idx]
                if module_type == 'encoder':
                    encoder_keep.append(layer_idx)
                else:
                    decoder_keep.append(layer_idx)
            
            # 执行剪枝
            if hasattr(self.encoder, 'prune_layers'):
                self.encoder.prune_layers(encoder_keep)
            
            if hasattr(self.decoder, 'prune_layers'):
                self.decoder.prune_layers(decoder_keep)
        
        elif strategy == 'per_module':
            # 每个模块独立保留 target_layers 层
            if hasattr(self.encoder, 'get_gate_scores'):
                encoder_scores = self.encoder.get_gate_scores()
                encoder_keep = np.argsort(encoder_scores)[-target_layers:].tolist()
                self.encoder.prune_layers(encoder_keep)
            
            if hasattr(self.decoder, 'get_gate_scores'):
                decoder_scores = self.decoder.get_gate_scores()
                decoder_keep = np.argsort(decoder_scores)[-target_layers:].tolist()
                self.decoder.prune_layers(decoder_keep)
        
        # 记录剪枝后的状态
        active_layers = self.get_active_layers()
        logger.info(f"Pruning complete. Active layers: {active_layers}")
    
    def get_active_layers(self) -> Dict[str, int]:
        """获取当前活跃层数"""
        active = {}
        
        if hasattr(self.encoder, 'get_active_layers'):
            active['encoder'] = self.encoder.get_active_layers()
        
        if hasattr(self.decoder, 'get_active_layers'):
            active['decoder'] = self.decoder.get_active_layers()
        
        return active
    
    def update_temperatures(self):
        """更新所有门控的温度（退火）"""
        if hasattr(self.encoder, 'update_temperatures'):
            self.encoder.update_temperatures()
        
        if hasattr(self.decoder, 'update_temperatures'):
            self.decoder.update_temperatures()
    
    def get_pruning_stats(self) -> Dict:
        """获取剪枝统计信息"""
        stats = {
            'gate_scores': self.get_all_gate_scores(),
            'last_masks': self.get_all_last_masks(),
            'active_layers': self.get_active_layers(),
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        
        # 计算被剪枝的参数量
        pruned_params = 0
        if hasattr(self.encoder, 'pruned_mask'):
            for idx, is_pruned in enumerate(self.encoder.pruned_mask):
                if is_pruned and idx < len(self.encoder.encoder.layers):
                    layer = self.encoder.encoder.layers[idx]
                    pruned_params += sum(p.numel() for p in layer.parameters())
        
        if hasattr(self.decoder, 'pruned_mask'):
            for idx, is_pruned in enumerate(self.decoder.pruned_mask):
                if is_pruned and idx < len(self.decoder.decoder.layers):
                    layer = self.decoder.decoder.layers[idx]
                    pruned_params += sum(p.numel() for p in layer.parameters())
        
        stats['pruned_params'] = pruned_params
        stats['compression_ratio'] = 1.0 - pruned_params / stats['total_params']
        
        return stats


def test():
    """测试 LightDP 模型"""
    # 测试 encoder-only 模式
    transformer = TransformerForDiffusionLightDP(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        causal_attn=True,
        n_layer=8,
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    out = transformer(sample, timestep)
    print(f"Encoder-only output shape: {out.shape}")
    
    # 测试门控得分
    scores = transformer.get_all_gate_scores()
    print(f"Gate scores: {scores}")
    
    # 测试剪枝
    transformer.prune_layers(target_layers=4, strategy='global')
    print(f"Active layers after pruning: {transformer.get_active_layers()}")
    
    # 测试剪枝后的前向传播
    out = transformer(sample, timestep)
    print(f"Output shape after pruning: {out.shape}")
    
    # 测试 decoder 模式
    transformer2 = TransformerForDiffusionLightDP(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        n_layer=8,
    )
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    cond = torch.zeros((4, 4, 10))
    out = transformer2(sample, timestep, cond)
    print(f"Decoder mode output shape: {out.shape}")
    
    # 测试剪枝统计
    stats = transformer2.get_pruning_stats()
    print(f"Pruning stats: {stats}")


if __name__ == "__main__":
    test()