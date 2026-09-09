from typing import Union, Optional, Tuple, List
import logging
import math
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class GumbelSoftmaxLayerGate(nn.Module):
    """
    Gumbel-Softmax gate for each transformer layer.
    Learns to select which layers to keep.
    During training, uses straight-through estimation.
    """
    def __init__(self, n_layers: int, temperature: float = 1.0):
        super().__init__()
        self.n_layers = n_layers
        self.temperature = temperature
        self.logits = nn.Parameter(torch.zeros(n_layers))
        self.active = False

    def forward(self, training: bool = True):
        if training and self.temperature > 0:
            gumbel = -torch.log(-torch.log(
                torch.rand(self.n_layers, device=self.logits.device).clamp(1e-6, 1.0)
            ).clamp(1e-6, 1.0))
            y_soft = torch.sigmoid((self.logits + gumbel) / self.temperature)
            y_hard = (y_soft > 0.5).float()
            return y_hard - y_soft.detach() + y_soft
        else:
            return (torch.sigmoid(self.logits) > 0.5).float()

    def get_prob(self):
        return torch.sigmoid(self.logits).detach().cpu().numpy()

    def set_temperature(self, temp: float):
        self.temperature = temp


class PruningScheduler:
    """
    Anneals Gumbel-Softmax temperature over training.
    """
    def __init__(self, init_temp: float = 1.0, min_temp: float = 0.1,
                 anneal_steps: int = 1000):
        self.init_temp = init_temp
        self.min_temp = min_temp
        self.anneal_steps = anneal_steps

    def get_temperature(self, step: int) -> float:
        ratio = min(1.0, step / max(1, self.anneal_steps))
        return self.init_temp * ((self.min_temp / self.init_temp) ** ratio)


def init_gate_with_svd(weight_tensor: torch.Tensor, n_kept: int):
    """
    Initialize gate logits based on SVD energy of layer weights.
    Layers with higher SVD energy get higher logits (more likely to be kept).
    """
    if weight_tensor is None:
        return torch.zeros(weight_tensor.shape[0]) if weight_tensor is not None else None
    if weight_tensor.dim() < 2:
        return torch.zeros(weight_tensor.shape[0])

    energies = []
    for i in range(weight_tensor.shape[0]):
        w = weight_tensor[i].float()
        try:
            _, s, _ = torch.linalg.svd(w.reshape(w.shape[0], -1), full_matrices=False)
            energy = (s ** 2).sum().item()
        except Exception:
            energy = w.norm().item() ** 2
        energies.append(energy)

    energies = np.array(energies)
    energies = energies / (energies.sum() + 1e-10)
    topk = np.argsort(-energies)[:n_kept]
    logits = np.zeros(len(energies))
    logits[topk] = 5.0
    return torch.from_numpy(logits).float()


class PrunableTransformerDecoder(nn.Module):
    """
    A TransformerDecoder that supports layer gating.
    Each decoder layer is multiplied by the corresponding gate value.
    """
    def __init__(self, decoder_layer: nn.TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=decoder_layer.d_model,
                nhead=decoder_layer.self_attn.num_heads,
                dim_feedforward=decoder_layer.linear1.out_features,
                dropout=decoder_layer.dropout,
                activation=decoder_layer.activation,
                batch_first=decoder_layer.batch_first,
                norm_first=decoder_layer.norm_first,
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = None
        if decoder_layer.norm_first:
            self.norm = nn.LayerNorm(decoder_layer.d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                gate_values=None):
        output = tgt
        for i, mod in enumerate(self.layers):
            if gate_values is not None:
                g = gate_values[i] if i < len(gate_values) else 1.0
                if not isinstance(g, torch.Tensor):
                    g = torch.tensor(g, device=output.device, dtype=output.dtype)
                if float(g.detach().cpu()) < 0.5:
                    continue
                output = mod(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
                output = output * g
            else:
                output = mod(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class LayerPruner:
    """
    Post-training layer pruning utilities for TransformerForDiffusion.
    Supports L1-norm and SVD-energy based importance scoring.
    """

    @staticmethod
    def get_layer_importance(model, target: str = 'decoder',
                             method: str = 'l1_norm') -> List[float]:
        """
        Compute importance score for each layer.
        """
        if target == 'decoder':
            layers = model.decoder.layers
        elif target == 'encoder':
            layers = model.encoder.layers if hasattr(model.encoder, 'layers') else []
        else:
            raise ValueError(f"Unknown target: {target}")

        scores = []
        for layer in layers:
            score = LayerPruner._score_layer(layer, method)
            scores.append(score)
        return scores

    @staticmethod
    def _score_layer(layer, method: str) -> float:
        """Score a single transformer layer."""
        if isinstance(layer, nn.TransformerDecoderLayer):
            attn = layer.self_attn.in_proj_weight
            ff1 = layer.linear1.weight
            ff2 = layer.linear2.weight
        elif isinstance(layer, nn.TransformerEncoderLayer):
            attn = layer.self_attn.in_proj_weight
            ff1 = layer.linear1.weight
            ff2 = layer.linear2.weight
        else:
            return 0.0

        if method == 'l1_norm':
            score = (attn.norm(p=1) + ff1.norm(p=1) + ff2.norm(p=1)).item()
        elif method == 'l2_norm':
            score = (attn.norm(p=2) + ff1.norm(p=2) + ff2.norm(p=2)).item()
        elif method == 'svd_energy':
            def svd_energy(w):
                try:
                    _, s, _ = torch.linalg.svd(
                        w.float().reshape(w.shape[0], -1), full_matrices=False
                    )
                    return (s ** 2).sum().item()
                except Exception:
                    return w.norm().item() ** 2
            score = svd_energy(attn) + svd_energy(ff1) + svd_energy(ff2)
        else:
            raise ValueError(f"Unknown method: {method}")
        return float(score)

    @staticmethod
    def prune_model(model, keep_ratio: float = 0.5, target: str = 'decoder',
                    method: str = 'l1_norm', strategy: str = 'topk'):
        """
        Physically prune layers from the model.
        Returns: (pruned_model, kept_indices, removed_indices)
        """
        scores = LayerPruner.get_layer_importance(model, target, method)
        n_layers = len(scores)
        n_keep = max(1, int(round(n_layers * keep_ratio)))

        if strategy == 'topk':
            kept_idx = sorted(np.argsort(scores)[-n_keep:].tolist())
        elif strategy == 'uniform':
            step = n_layers / n_keep
            kept_idx = sorted([int(i * step) for i in range(n_keep)])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        removed_idx = sorted(set(range(n_layers)) - set(kept_idx))

        if target == 'decoder':
            LayerPruner._prune_decoder(model, kept_idx)
        elif target == 'encoder':
            LayerPruner._prune_encoder(model, kept_idx)

        return model, kept_idx, removed_idx

    @staticmethod
    def _prune_decoder(model, kept_idx: List[int]):
        """Rebuild model.decoder with only the kept layers."""
        old_decoder = model.decoder
        if not hasattr(old_decoder, 'layers') or len(old_decoder.layers) == 0:
            return

        decoder_layer = old_decoder.layers[0]
        new_decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_layer.d_model,
            nhead=decoder_layer.self_attn.num_heads,
            dim_feedforward=decoder_layer.linear1.out_features,
            dropout=decoder_layer.dropout,
            activation=decoder_layer.activation,
            batch_first=decoder_layer.batch_first,
            norm_first=decoder_layer.norm_first,
        )
        new_decoder = nn.TransformerDecoder(new_decoder_layer, num_layers=len(kept_idx))

        with torch.no_grad():
            for new_i, old_i in enumerate(kept_idx):
                old_state = old_decoder.layers[old_i].state_dict()
                new_state = new_decoder.layers[new_i].state_dict()
                for k in new_state:
                    if k in old_state and new_state[k].shape == old_state[k].shape:
                        new_state[k].copy_(old_state[k])
                new_decoder.layers[new_i].load_state_dict(new_state)

            if old_decoder.norm is not None and new_decoder.norm is not None:
                new_decoder.norm.load_state_dict(old_decoder.norm.state_dict())

        model.decoder = new_decoder
        model.n_layer = len(kept_idx)

    @staticmethod
    def _prune_encoder(model, kept_idx: List[int]):
        """Rebuild model.encoder with only the kept layers (for encoder-only models)."""
        old_encoder = model.encoder
        if not hasattr(old_encoder, 'layers') or len(old_encoder.layers) == 0:
            return

        encoder_layer = old_encoder.layers[0]
        new_encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_layer.d_model,
            nhead=encoder_layer.self_attn.num_heads,
            dim_feedforward=encoder_layer.linear1.out_features,
            dropout=encoder_layer.dropout,
            activation=encoder_layer.activation,
            batch_first=encoder_layer.batch_first,
            norm_first=encoder_layer.norm_first,
        )
        new_encoder = nn.TransformerEncoder(new_encoder_layer, num_layers=len(kept_idx))

        with torch.no_grad():
            for new_i, old_i in enumerate(kept_idx):
                old_state = old_encoder.layers[old_i].state_dict()
                new_state = new_encoder.layers[new_i].state_dict()
                for k in new_state:
                    if k in old_state and new_state[k].shape == old_state[k].shape:
                        new_state[k].copy_(old_state[k])
                new_encoder.layers[new_i].load_state_dict(new_state)

            if old_encoder.norm is not None and new_encoder.norm is not None:
                new_encoder.norm.load_state_dict(old_encoder.norm.state_dict())

        model.encoder = new_encoder

    @staticmethod
    def measure_model_size(model) -> dict:
        """Return parameter count info."""
        total = sum(p.numel() for p in model.parameters())
        decoder_params = 0
        if hasattr(model, 'decoder') and model.decoder is not None:
            decoder_params = sum(p.numel() for p in model.decoder.parameters())
        encoder_params = 0
        if hasattr(model, 'encoder') and model.encoder is not None:
            encoder_params = sum(p.numel() for p in model.encoder.parameters())
        return {
            'total_params': total,
            'total_params_M': total / 1e6,
            'decoder_params': decoder_params,
            'encoder_params': encoder_params,
        }


class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
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
            causal_attn: bool=False,
            time_as_cond: bool=True,
            obs_as_cond: bool=False,
            n_cond_layers: int = 0,
            enable_layer_pruning: bool = False,
            pruning_block_size: int = 4,
            pruning_keep_ratio: float = 0.5,
            pruning_temperature: float = 0.5,
        ) -> None:
        super().__init__()

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
                    dim_feedforward=4*n_emb,
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
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
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
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
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
                mask = t >= (s-1)
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
        self.n_layer = n_layer

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_pruning_info(self):
        """Return pruning-related info (used by eval.py)."""
        kept_indices = list(range(self.n_layer))
        if hasattr(self, 'pruned_layer_indices') and self.pruned_layer_indices:
            kept_indices = self.pruned_layer_indices.copy()
        
        info = {
            'is_pruned': False,
            'n_layers': self.n_layer,
            'kept_layers': self.n_layer,
            'pruned_layers': 0,
            'keep_ratio': 1.0,
            'kept_indices': kept_indices,
        }
        if hasattr(self, 'decoder') and self.decoder is not None:
            if hasattr(self.decoder, 'layers'):
                current_layers = len(self.decoder.layers)
                if current_layers < self.n_layer:
                    info['is_pruned'] = True
                    info['kept_layers'] = current_layers
                    info['pruned_layers'] = self.n_layer - current_layers
                    info['keep_ratio'] = current_layers / self.n_layer
                    info['kept_indices'] = kept_indices[:current_layers] if len(kept_indices) >= current_layers else kept_indices
        elif hasattr(self, 'encoder') and self.encoder is not None:
            if hasattr(self.encoder, 'layers'):
                current_layers = len(self.encoder.layers)
                if current_layers < self.n_layer:
                    info['is_pruned'] = True
                    info['kept_layers'] = current_layers
                    info['pruned_layers'] = self.n_layer - current_layers
                    info['keep_ratio'] = current_layers / self.n_layer
        return info

    def enable_cuda_timing(self):
        self.enable_timing = True
        self.timing_stats.clear()
        
    def disable_cuda_timing(self):
        self.enable_timing = False
        self.timing_stats.clear()
        
    def clear_timing_stats(self):
        self.timing_stats.clear()
        
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
            nn.Sequential)
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
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
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
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
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

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None,
        layer_masks: Optional[torch.Tensor]=None,
        **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        layer_masks: (n_layer,) gate values for each decoder layer, None=all active
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
            # Block 3: BERT encoder
            start3 = torch.cuda.Event(enable_timing=True)
            end3 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start3.record()
            
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:,1:,:]
            
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
            
            # Block 4: encoder forward
            start4 = torch.cuda.Event(enable_timing=True)
            end4 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start4.record()
            
            x = self.encoder(x)
            memory = x
            
            self._record_time('4_Encoder_Forward', start4, end4)
            
            # Block 5: decoder forward (with optional layer gating)
            start5 = torch.cuda.Event(enable_timing=True)
            end5 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start5.record()
            
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            
            if layer_masks is not None and self.decoder is not None:
                x = self._decoder_forward_with_gating(
                    x, memory, layer_masks
                )
            else:
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

    def _decoder_forward_with_gating(self, tgt, memory, layer_masks):
        """
        Decoder forward with per-layer gating.
        保持标准 Transformer 残差连接: output = output + gate * layer_output
        - 当gate=0时: 跳过层 (output不变)
        - 当gate=1时: 标准残差连接 (output = output + layer_output)
        - 当gate为软值时: 缩放层输出后做残差连接
        
        物理剪枝后, self.pruned_layer_indices 存储保留层的原始索引
        decoder_layers[i] 对应原始层索引 pruned_layer_indices[i]
        """
        output = tgt
        target_device = tgt.device
        target_dtype = tgt.dtype
        decoder_layers = self.decoder.layers
        
        for idx, layer in enumerate(decoder_layers):
            if hasattr(self, 'pruned_layer_indices') and self.pruned_layer_indices:
                orig_idx = self.pruned_layer_indices[idx] if idx < len(self.pruned_layer_indices) else idx
            else:
                orig_idx = idx
            
            if orig_idx < len(layer_masks):
                gate = layer_masks[orig_idx]
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

    def get_pruning_stats(self):
        """Get pruning statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        
        decoder_layers = self.decoder.layers if self.decoder is not None else []
        layer_info = []
        for idx, layer in enumerate(decoder_layers):
            layer_params = sum(p.numel() for p in layer.parameters())
            layer_info.append({
                'layer_idx': idx,
                'params': layer_params,
            })
        
        return {
            'total_params': total_params,
            'decoder_layers': len(decoder_layers),
            'layer_info': layer_info,
        }

    def prune_decoder_layers(self, keep_indices):
        """Physically remove unpruned decoder layers"""
        if self.decoder is None:
            return
        
        keep_indices = sorted(keep_indices)
        original_count = len(self.decoder.layers)
        old_layers = self.decoder.layers
        
        kept_layers = nn.ModuleList([
            self.decoder.layers[i] for i in keep_indices
        ])
        
        self.decoder.layers = kept_layers
        self.n_layer = len(keep_indices)
        self.pruned_layer_indices = keep_indices
        
        del old_layers
        gc.collect()
        
        logger.info(
            f"Pruned decoder: kept {len(keep_indices)}/{original_count} layers, "
            f"indices: {keep_indices}"
        )


def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        causal_attn=True,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        time_as_cond=False,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)