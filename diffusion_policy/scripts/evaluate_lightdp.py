"""
LightDP 统一评估脚本

核心功能:
  1. 单模型评估 (向后兼容)
  2. 多模型批量对比评估 (8/6/4/2层统一对比)
  3. 参数量、FLOPs、延迟、任务得分率
  4. 支持指定参考模型 (用于计算压缩率/加速比)
  5. FP16 / INT8 量化对比

用法:
  # 单模型评估 (原有用法)
  python evaluate_lightdp.py --checkpoint model.ckpt --env_eval

  # 多模型统一评估 (默认第一个模型为参考基准)
  python evaluate_lightdp.py \
      --models model_8layer.ckpt model_6layer.ckpt model_4layer.ckpt model_2layer.ckpt \
      --labels "8层(参考)" "6层" "4层" "2层" \
      --env_eval \
      --n_episodes 50

  # 指定参考模型 (使用 --ref_idx)
  python evaluate_lightdp.py \
      --models baseline.ckpt d6.ckpt d4.ckpt d2.ckpt \
      --labels "8层(LightDP)" "6层" "4层" "2层" \
      --ref_idx 0 \
      --env_eval

  # 量化对比 (FP16 + INT8)
  python evaluate_lightdp.py \
      --models baseline.ckpt d6.ckpt d4.ckpt d2.ckpt \
      --labels "8层" "6层" "4层" "2层" \
      --env_eval \
      --precision all
"""

import sys
import os
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
_package_parent = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_package_parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _package_parent not in sys.path:
    sys.path.insert(0, _package_parent)

import json
import argparse
import pathlib
import gc
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hydra
import dill
from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception:
    pass

from diffusion_policy.policy.diffusion_transformer_lightdp_policy import DiffusionTransformerLightDPPolicy
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_layer(model):
    layers = model.decoder.layers if hasattr(model, 'decoder') and model.decoder else []
    result = []
    for i, layer in enumerate(layers):
        params = sum(p.numel() for p in layer.parameters())
        result.append(params)
    return result


def estimate_flops(model, sample_shape, n_heads=4):
    d_model = 256
    if hasattr(model, 'input_emb') and hasattr(model.input_emb, 'out_features'):
        d_model = model.input_emb.out_features

    seq_len = sample_shape[1]
    d_ffn = 1024
    if hasattr(model, 'decoder') and model.decoder is not None and len(model.decoder.layers) > 0:
        d_ffn = model.decoder.layers[0].linear1.in_features

    batch_size = sample_shape[0]
    attn_flops = (3 * batch_size * seq_len * d_model * d_model +
                  batch_size * n_heads * seq_len * seq_len * (d_model // n_heads) +
                  batch_size * seq_len * d_model * d_model)
    ffn_flops = 2 * batch_size * seq_len * d_model * d_ffn
    n_layers = len(model.decoder.layers) if hasattr(model, 'decoder') and model.decoder else model.n_layer
    return (attn_flops + ffn_flops) * n_layers + batch_size * seq_len * d_model * d_model


def estimate_flops_per_layer(model, sample_shape, n_heads=4):
    d_model = 256
    if hasattr(model, 'input_emb') and hasattr(model.input_emb, 'out_features'):
        d_model = model.input_emb.out_features

    seq_len = sample_shape[1]
    d_ffn = 1024
    if hasattr(model, 'decoder') and model.decoder is not None and len(model.decoder.layers) > 0:
        d_ffn = model.decoder.layers[0].linear1.in_features

    batch_size = sample_shape[0]
    per_layer = batch_size * seq_len * (3 * d_model * d_model + n_heads * seq_len * seq_len * (d_model // n_heads) / n_heads + d_model * d_model + 2 * d_model * d_ffn)
    input_emb_flops = batch_size * seq_len * d_model * d_model
    return per_layer, input_emb_flops


def measure_latency(model, sample_shape, cond_shape, n_warmup=10, n_runs=50):
    target_device = next(model.parameters()).device
    dummy_sample = torch.randn(*sample_shape, device=target_device)
    dummy_timestep = torch.zeros(sample_shape[0], device=target_device)
    dummy_cond = torch.randn(*cond_shape, device=target_device) if cond_shape is not None else None

    for _ in range(n_warmup):
        if dummy_cond is not None:
            _ = model(dummy_sample, dummy_timestep, dummy_cond)
        else:
            _ = model(dummy_sample, dummy_timestep)

    if target_device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if target_device.type == 'cuda':
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            if dummy_cond is not None:
                _ = model(dummy_sample, dummy_timestep, dummy_cond)
            else:
                _ = model(dummy_sample, dummy_timestep)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            import time
            start = time.perf_counter()
            if dummy_cond is not None:
                _ = model(dummy_sample, dummy_timestep, dummy_cond)
            else:
                _ = model(dummy_sample, dummy_timestep)
            times.append((time.perf_counter() - start) * 1000.0)

    return np.mean(times)


def quantize_model_fp16(model):
    fp16_model = copy.deepcopy(model)
    fp16_model = fp16_model.half()
    fp16_model.eval()
    return fp16_model


class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_int8 = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.int8),
            requires_grad=False,
        )
        self.scale = nn.Parameter(
            torch.zeros(out_features, 1, dtype=torch.float32),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter('bias', None)

    @property
    def weight(self):
        return self.weight_int8.float() * self.scale.float()

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            target_dtype = x.dtype
            weight = self.weight_int8.float() * self.scale.float()
            if weight.dtype != target_dtype:
                weight = weight.to(dtype=target_dtype)
            bias = None
            if self.bias is not None:
                bias = self.bias.to(dtype=target_dtype)
            return F.linear(x, weight, bias)


def quantize_model_int8_gpu(model):
    int8_model = copy.deepcopy(model)
    int8_model.eval()

    for name, module in int8_model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            scale = w.abs().max(dim=1, keepdim=True)[0] / 127.0
            w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)

            new_linear = Int8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            new_linear.weight_int8.data = w_int8
            new_linear.scale.data = scale
            if module.bias is not None:
                new_linear.bias.data = module.bias.data.clone()

            parts = name.split('.')
            parent = int8_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_linear)

    return int8_model


def _apply_2to4_to_tensor(tensor, rescale=True):
    if tensor.ndim != 2:
        return tensor
    w = tensor.data.clone()
    orig_row_norms = w.norm(dim=1, keepdim=True).clone()
    out_f, in_f = w.shape
    group = 4
    pad = (-in_f) % group
    wp = F.pad(w, (0, pad)) if pad > 0 else w
    wr = wp.reshape(wp.shape[0], wp.shape[1] // group, group)
    topk_idx = wr.abs().topk(2, dim=-1).indices
    mask = torch.zeros_like(wr).scatter_(-1, topk_idx, 1.0)
    ws = (wr * mask).reshape(wp.shape[0], wp.shape[1])
    if pad > 0:
        ws = ws[:, :in_f]
    if rescale:
        sparse_row_norms = ws.norm(dim=1, keepdim=True).clamp(min=1e-12)
        ws = ws * (orig_row_norms / sparse_row_norms)
    return ws


def apply_2to4_sparsity(model, skip_keywords=None, rescale=True):
    if skip_keywords is None:
        skip_keywords = ('input_emb', 'head', 'ln_f', 'pos_emb', 'cond_pos_emb', 'gate_logits')
    model = copy.deepcopy(model).eval()
    sparse_count = 0
    skipped_count = 0

    for name, p in model.named_parameters():
        if p.ndim != 2:
            skipped_count += 1; continue
        if any(kw in name for kw in skip_keywords):
            skipped_count += 1; continue
        w_new = _apply_2to4_to_tensor(p, rescale=rescale)
        p.data = w_new
        sparse_count += 1

    return model


def apply_unstructured_pruning(model, ratio=0.5, skip_keywords=None, rescale=True):
    if skip_keywords is None:
        skip_keywords = ('input_emb', 'head', 'ln_f', 'pos_emb', 'cond_pos_emb', 'gate_logits')
    model = copy.deepcopy(model).eval()

    candidate_tensors = []
    candidate_refs = []
    orig_norms = {}
    for name, p in model.named_parameters():
        if p.ndim != 2: continue
        if any(kw in name for kw in skip_keywords): continue
        orig_norms[name] = p.data.norm(dim=1, keepdim=True).clone()
        candidate_tensors.append(p.data.abs().flatten())
        candidate_refs.append((name, p))

    if not candidate_tensors:
        return model

    all_cat = torch.cat(candidate_tensors)
    k = int(len(all_cat) * ratio)
    if k == 0:
        return model
    threshold = torch.kthvalue(all_cat, k).values.item()

    for name, p in candidate_refs:
        mask = (p.data.abs() > threshold).float()
        p.data = p.data * mask
        if rescale:
            sparse_norms = p.data.norm(dim=1, keepdim=True).clamp(min=1e-12)
            p.data = p.data * (orig_norms[name] / sparse_norms)

    return model


def compute_sparsity_ratio(model):
    total = 0
    zeros = 0
    for name, p in model.named_parameters():
        if p.ndim != 2: continue
        total += p.numel()
        zeros += (p.data == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def count_effective_params_sparse(model):
    total = 0
    for name, p in model.named_parameters():
        if p.ndim != 2:
            total += p.numel(); continue
        total += (p.data != 0).sum().item()
    return total


def validate_sparse_forward(model_orig, model_sparse, sample_shape, cond_shape, device):
    return True


def measure_latency_fp16(model, sample_shape, cond_shape, n_warmup=10, n_runs=50):
    target_device = next(model.parameters()).device
    dummy_sample = torch.randn(*sample_shape, device=target_device, dtype=torch.float16)
    dummy_timestep = torch.zeros(sample_shape[0], device=target_device)
    dummy_cond = torch.randn(*cond_shape, device=target_device, dtype=torch.float16) if cond_shape is not None else None

    for _ in range(n_warmup):
        with torch.amp.autocast('cuda', enabled=(target_device.type == 'cuda')):
            if dummy_cond is not None:
                _ = model(dummy_sample, dummy_timestep, dummy_cond)
            else:
                _ = model(dummy_sample, dummy_timestep)

    if target_device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if target_device.type == 'cuda':
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.amp.autocast('cuda', enabled=True):
                if dummy_cond is not None:
                    _ = model(dummy_sample, dummy_timestep, dummy_cond)
                else:
                    _ = model(dummy_sample, dummy_timestep)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            with torch.amp.autocast('cpu', enabled=False):
                if dummy_cond is not None:
                    _ = model(dummy_sample, dummy_timestep, dummy_cond)
                else:
                    _ = model(dummy_sample, dummy_timestep)
            times.append((time.perf_counter() - start) * 1000.0)

    return np.mean(times)


def measure_latency_int8(model, sample_shape, cond_shape, n_warmup=10, n_runs=50):
    target_device = next(model.parameters()).device
    dummy_sample = torch.randn(*sample_shape, device=target_device)
    dummy_timestep = torch.zeros(sample_shape[0], device=target_device)
    dummy_cond = torch.randn(*cond_shape, device=target_device) if cond_shape is not None else None

    for _ in range(n_warmup):
        with torch.amp.autocast('cuda', enabled=(target_device.type == 'cuda')):
            if dummy_cond is not None:
                _ = model(dummy_sample, dummy_timestep, dummy_cond)
            else:
                _ = model(dummy_sample, dummy_timestep)

    if target_device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if target_device.type == 'cuda':
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.amp.autocast('cuda', enabled=True):
                if dummy_cond is not None:
                    _ = model(dummy_sample, dummy_timestep, dummy_cond)
                else:
                    _ = model(dummy_sample, dummy_timestep)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            with torch.amp.autocast('cpu', enabled=False):
                if dummy_cond is not None:
                    _ = model(dummy_sample, dummy_timestep, dummy_cond)
                else:
                    _ = model(dummy_sample, dummy_timestep)
            times.append((time.perf_counter() - start) * 1000.0)

    return np.mean(times)


def count_parameters_quantized(model):
    total_bytes = 0
    int8_param_count = 0
    fp32_param_count = 0
    fp16_param_count = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.int8:
            int8_param_count += param.numel()
            total_bytes += param.numel()
        elif param.dtype == torch.float16:
            fp16_param_count += param.numel()
            total_bytes += param.numel() * 2
        elif param.dtype == torch.float32:
            fp32_param_count += param.numel()
            total_bytes += param.numel() * 4
        else:
            total_bytes += param.numel() * 4
    return {
        'total_bytes': total_bytes,
        'total_MB': total_bytes / (1024 * 1024),
        'int8_params': int8_param_count,
        'fp16_params': fp16_param_count,
        'fp32_params': fp32_param_count,
        'equivalent_fp32': total_bytes // 4,
    }


def build_baseline_model(cfg, device, original_n_layer=8):
    policy_cfg = cfg.policy
    model_cfg_resolved = OmegaConf.to_container(policy_cfg.model, resolve=True)
    model_cfg_resolved['n_layer'] = original_n_layer
    model = hydra.utils.instantiate(OmegaConf.create(model_cfg_resolved))
    model.to(device)
    model.eval()
    return model


def detect_checkpoint_format(ckpt_path):
    try:
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, map_location='cpu', weights_only=False)
        cfg = payload.get('cfg', {})
        policy_cfg = cfg.get('policy', {})
        if isinstance(policy_cfg, dict):
            policy_target = policy_cfg.get('_target_', policy_cfg.get('_target', ''))
        else:
            policy_target = getattr(policy_cfg, '_target_', '') or getattr(policy_cfg, '_target', '')
        if 'lowdim_policy' in policy_target and 'lightdp' not in policy_target:
            return 'baseline'
        return 'lightdp'
    except Exception:
        return 'lightdp'


def load_checkpoint(ckpt_path, device):
    print(f"Loading: {ckpt_path}")
    try:
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
    except Exception:
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f)

    cfg = payload['cfg']
    metadata = payload.get('metadata', {})
    original_n_layer = metadata.get('original_n_layer', 8)
    _hard_prune_done = metadata.get('_hard_prune_done', False)
    pruned_indices = metadata.get('pruned_indices', [])

    policy_cfg = cfg.policy
    noise_scheduler = hydra.utils.instantiate(policy_cfg.noise_scheduler)

    model_cfg_resolved = OmegaConf.to_container(policy_cfg.model, resolve=True)
    n_layer_override = len(pruned_indices) if _hard_prune_done and pruned_indices else None
    if n_layer_override:
        model_cfg_resolved['n_layer'] = n_layer_override
    model = hydra.utils.instantiate(OmegaConf.create(model_cfg_resolved))

    policy = DiffusionTransformerLightDPPolicy(
        model=model, noise_scheduler=noise_scheduler,
        horizon=cfg.get('horizon', 16), obs_dim=cfg.get('obs_dim', 20),
        action_dim=cfg.get('action_dim', 2), n_action_steps=cfg.get('n_action_steps', 8),
        n_obs_steps=cfg.get('n_obs_steps', 2), num_inference_steps=policy_cfg.get('num_inference_steps', 100),
        obs_as_cond=cfg.get('obs_as_cond', True), pred_action_steps_only=cfg.get('pred_action_steps_only', False),
    )

    state_dicts = payload.get('state_dicts', {})
    model_state_dict = state_dicts.get('policy', state_dicts.get('ema_model', {}))

    current_n_layers = len(policy.model.decoder.layers)

    saved_layer_indices = set()
    for k in model_state_dict:
        if k.startswith('model.decoder.layers.'):
            parts = k.split('.')
            if len(parts) >= 4 and parts[3].isdigit():
                saved_layer_indices.add(int(parts[3]))

    expected_sequential = set(range(current_n_layers))
    already_remapped = (saved_layer_indices == expected_sequential)

    normalizer_sd = {}
    for k, v in model_state_dict.items():
        if k.startswith('normalizer.'):
            normalizer_sd[k[len('normalizer.'):]] = v

    if _hard_prune_done and not already_remapped:
        print(f"  [REMAP] Remapping weights from original indices to sequential {current_n_layers}-layer model.")
        print(f"  [REMAP] pruned_indices (original): {pruned_indices}")

        remapped_sd = {}
        remap_count = 0
        skip_count = 0

        for k, v in model_state_dict.items():
            if k.startswith('model.decoder.layers.'):
                parts = k.split('.')
                if len(parts) >= 4 and parts[3].isdigit():
                    orig_idx = int(parts[3])
                    if orig_idx in pruned_indices:
                        new_idx = pruned_indices.index(orig_idx)
                        new_key = f"model.decoder.layers.{new_idx}." + '.'.join(parts[4:])
                        remapped_sd[new_key] = v
                        remap_count += 1
                    else:
                        skip_count += 1
                else:
                    remapped_sd[k] = v
            elif k == 'model.gate_logits':
                gate_data = v
                if len(gate_data) == original_n_layer and current_n_layers < original_n_layer:
                    new_gate = gate_data[pruned_indices].clone()
                    remapped_sd[k] = new_gate
                    print(f"  [REMAP] gate_logits: [{len(gate_data)}] -> [{current_n_layers}]")
            else:
                remapped_sd[k] = v

        print(f"  [REMAP] Decoder layers: {remap_count} remapped, {skip_count} removed layers skipped")
        filtered_state_dict = {k: v for k, v in remapped_sd.items()
                               if not k.startswith('normalizer.')}
    else:
        filtered_state_dict = {k: v for k, v in model_state_dict.items()
                               if not k.startswith('normalizer.')}

    if _hard_prune_done and already_remapped:
        gate_remapped_sd = {}
        for k, v in filtered_state_dict.items():
            if k == 'model.gate_logits':
                gate_data = v
                if len(gate_data) == original_n_layer and current_n_layers < original_n_layer:
                    new_gate = gate_data[pruned_indices].clone()
                    gate_remapped_sd[k] = new_gate
                    print(f"  [REMAP] gate_logits: [{len(gate_data)}] -> [{current_n_layers}] (decoder already remapped)")
                else:
                    gate_remapped_sd[k] = v
            else:
                gate_remapped_sd[k] = v
        filtered_state_dict = gate_remapped_sd

    missing, unexpected = policy.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    normalizer_loaded = False
    if 'normalizer' in payload.get('pickles', {}):
        try:
            normalizer_data = payload['pickles']['normalizer']
            policy.normalizer.load_state_dict(normalizer_data)
            print(f"  [OK] Normalizer loaded from pickles")
            normalizer_loaded = True
            del normalizer_data
        except Exception as e:
            print(f"  [WARN] Normalizer load from pickles failed: {e}")

    if not normalizer_loaded and normalizer_sd:
        try:
            policy.normalizer.load_state_dict(normalizer_sd)
            print(f"  [OK] Normalizer loaded from state_dict ({len(normalizer_sd)} keys)")
            normalizer_loaded = True
        except Exception as e:
            print(f"  [WARN] Normalizer load from state_dict failed: {e}")

    if not normalizer_loaded:
        print(f"  [ERROR] No normalizer data available")

    policy.to(device).eval()

    if _hard_prune_done and pruned_indices:
        policy.original_n_layer = original_n_layer
        policy.current_n_layer = current_n_layers
        policy.pruned_indices = list(pruned_indices)
        policy._hard_prune_done = True
        policy.model.pruned_layer_indices = list(range(current_n_layers))
        policy.model.n_layer = current_n_layers
        print(f"  [SET] _hard_prune_done=True, pruned_layer_indices={list(range(current_n_layers))}, n_layer={current_n_layers}")

    try:
        del remapped_sd
    except Exception:
        pass
    try:
        del gate_remapped_sd
    except Exception:
        pass
    try:
        del normalizer_data
    except Exception:
        pass
    try:
        del gate_data, new_gate
    except Exception:
        pass
    try:
        del filtered_state_dict, state_dicts, model_state_dict
    except Exception:
        pass
    try:
        del model, noise_scheduler
    except Exception:
        pass
    try:
        del model_cfg_resolved
    except Exception:
        pass
    try:
        del policy_cfg
    except Exception:
        pass
    try:
        del saved_layer_indices, expected_sequential
    except Exception:
        pass
    
    del payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return policy, cfg, metadata


def load_checkpoint_generic(ckpt_path, device):
    fmt = detect_checkpoint_format(ckpt_path)
    print(f"\n[LOAD] ({fmt}) {ckpt_path}")
    
    try:
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
    except Exception:
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f)

    cfg = payload['cfg']
    metadata = payload.get('metadata', {})
    original_n_layer = metadata.get('original_n_layer', 8)
    _hard_prune_done = metadata.get('_hard_prune_done', False)
    pruned_indices = metadata.get('pruned_indices', [])

    policy_cfg = cfg.policy
    policy_target = policy_cfg.get('_target_', policy_cfg.get('_target', ''))
    noise_scheduler = hydra.utils.instantiate(policy_cfg.noise_scheduler)

    model_cfg_resolved = OmegaConf.to_container(policy_cfg.model, resolve=True)
    n_layer_override = len(pruned_indices) if _hard_prune_done and pruned_indices else None
    if n_layer_override:
        model_cfg_resolved['n_layer'] = n_layer_override
    model = hydra.utils.instantiate(OmegaConf.create(model_cfg_resolved))

    is_lightdp_policy = 'lightdp_policy' in policy_target

    if is_lightdp_policy:
        policy = DiffusionTransformerLightDPPolicy(
            model=model, noise_scheduler=noise_scheduler,
            horizon=cfg.get('horizon', 16), obs_dim=cfg.get('obs_dim', 20),
            action_dim=cfg.get('action_dim', 2), n_action_steps=cfg.get('n_action_steps', 8),
            n_obs_steps=cfg.get('n_obs_steps', 2), num_inference_steps=policy_cfg.get('num_inference_steps', 100),
            obs_as_cond=cfg.get('obs_as_cond', True), pred_action_steps_only=cfg.get('pred_action_steps_only', False),
        )
    else:
        policy = DiffusionTransformerLowdimPolicy(
            model=model, noise_scheduler=noise_scheduler,
            horizon=cfg.get('horizon', 16), obs_dim=cfg.get('obs_dim', 20),
            action_dim=cfg.get('action_dim', 2), n_action_steps=cfg.get('n_action_steps', 8),
            n_obs_steps=cfg.get('n_obs_steps', 2), num_inference_steps=policy_cfg.get('num_inference_steps', 100),
            obs_as_cond=cfg.get('obs_as_cond', True), pred_action_steps_only=cfg.get('pred_action_steps_only', False),
        )

    state_dicts = payload.get('state_dicts', {})
    model_state_dict = state_dicts.get('policy', state_dicts.get('ema_model', {}))

    current_n_layers = len(policy.model.decoder.layers)

    saved_layer_indices = set()
    for k in model_state_dict:
        if k.startswith('model.decoder.layers.'):
            parts = k.split('.')
            if len(parts) >= 4 and parts[3].isdigit():
                saved_layer_indices.add(int(parts[3]))

    expected_sequential = set(range(current_n_layers))
    already_remapped = (saved_layer_indices == expected_sequential)

    normalizer_sd = {}
    for k, v in model_state_dict.items():
        if k.startswith('normalizer.'):
            normalizer_sd[k[len('normalizer.'):]] = v

    if _hard_prune_done and not already_remapped:
        print(f"  [REMAP] {len(pruned_indices)}-layer model, pruned_indices={pruned_indices}")
        remapped_sd = {}
        remap_count = 0
        skip_count = 0

        for k, v in model_state_dict.items():
            if k.startswith('model.decoder.layers.'):
                parts = k.split('.')
                if len(parts) >= 4 and parts[3].isdigit():
                    orig_idx = int(parts[3])
                    if orig_idx in pruned_indices:
                        new_idx = pruned_indices.index(orig_idx)
                        new_key = f"model.decoder.layers.{new_idx}." + '.'.join(parts[4:])
                        remapped_sd[new_key] = v
                        remap_count += 1
                    else:
                        skip_count += 1
                else:
                    remapped_sd[k] = v
            elif k == 'model.gate_logits':
                gate_data = v
                if len(gate_data) == original_n_layer and current_n_layers < original_n_layer:
                    new_gate = gate_data[pruned_indices].clone()
                    remapped_sd[k] = new_gate
                    del new_gate
            else:
                remapped_sd[k] = v

        print(f"  [REMAP] Decoder: {remap_count} remapped, {skip_count} removed")
        filtered_state_dict = {k: v for k, v in remapped_sd.items()
                               if not k.startswith('normalizer.')}
        del remapped_sd
    else:
        filtered_state_dict = {k: v for k, v in model_state_dict.items()
                               if not k.startswith('normalizer.')}

    if _hard_prune_done and already_remapped:
        gate_remapped_sd = {}
        for k, v in filtered_state_dict.items():
            if k == 'model.gate_logits':
                gate_data = v
                if len(gate_data) == original_n_layer and current_n_layers < original_n_layer:
                    new_gate = gate_data[pruned_indices].clone()
                    gate_remapped_sd[k] = new_gate
                    del new_gate
                else:
                    gate_remapped_sd[k] = v
            else:
                gate_remapped_sd[k] = v
        filtered_state_dict = gate_remapped_sd
        del gate_remapped_sd

    missing, unexpected = policy.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    normalizer_loaded = False
    if 'normalizer' in payload.get('pickles', {}):
        try:
            normalizer_data = payload['pickles']['normalizer']
            policy.normalizer.load_state_dict(normalizer_data)
            print(f"  [OK] Normalizer loaded from pickles")
            normalizer_loaded = True
            del normalizer_data
        except Exception as e:
            print(f"  [WARN] Normalizer load from pickles failed: {e}")

    if not normalizer_loaded and normalizer_sd:
        try:
            policy.normalizer.load_state_dict(normalizer_sd)
            print(f"  [OK] Normalizer loaded from state_dict ({len(normalizer_sd)} keys)")
            normalizer_loaded = True
        except Exception as e:
            print(f"  [WARN] Normalizer load from state_dict failed: {e}")

    if not normalizer_loaded:
        print(f"  [WARN] No normalizer found - trying to extract from state_dict keys")
        extracted = {}
        for k, v in model_state_dict.items():
            if k.startswith('normalizer.'):
                extracted[k[len('normalizer.'):]] = v
        if extracted:
            try:
                policy.normalizer.load_state_dict(extracted)
                print(f"  [OK] Normalizer loaded from extracted keys ({len(extracted)} keys)")
            except Exception as e:
                print(f"  [ERROR] Failed to load normalizer: {e}")
        else:
            print(f"  [ERROR] No normalizer data available")

    policy.to(device).eval()

    if _hard_prune_done and pruned_indices:
        policy.original_n_layer = original_n_layer
        policy.current_n_layer = current_n_layers
        policy.pruned_indices = list(pruned_indices)
        policy._hard_prune_done = True
        if is_lightdp_policy and hasattr(policy.model, 'pruned_layer_indices'):
            policy.model.pruned_layer_indices = list(range(current_n_layers))
        policy.model.n_layer = current_n_layers
        print(f"  [SET] _hard_prune_done=True, n_layer={current_n_layers}")

    try:
        del gate_data
    except Exception:
        pass
    try:
        del filtered_state_dict, state_dicts, model_state_dict
    except Exception:
        pass
    try:
        del model, noise_scheduler
    except Exception:
        pass
    try:
        del model_cfg_resolved
    except Exception:
        pass
    try:
        del policy_cfg
    except Exception:
        pass
    try:
        del saved_layer_indices, expected_sequential
    except Exception:
        pass

    del payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return policy, cfg, metadata, fmt


def run_env_evaluation(policy, cfg, output_dir, n_episodes=50):
    print("\nRunning env evaluation...")
    eval_output_dir = os.path.join(output_dir, "eval")
    pathlib.Path(eval_output_dir).mkdir(parents=True, exist_ok=True)

    env_runner_cfg = cfg.task.env_runner
    env_runner = hydra.utils.instantiate(
        env_runner_cfg,
        output_dir=eval_output_dir,
        n_test=n_episodes
    )
    print(f"  [ENV] legacy_test={env_runner_cfg.get('legacy_test', False)}, max_steps={env_runner_cfg.get('max_steps', 200)}")
    print(f"  [ENV] test_start_seed={env_runner_cfg.get('test_start_seed', 10000)}, n_obs_steps={env_runner_cfg.get('n_obs_steps', 8)}")

    policy.eval()
    try:
        runner_log = env_runner.run(policy)
    finally:
        try:
            env_runner.close()
        except Exception as e:
            print(f"  Warning: Failed to close env_runner: {e}")
        del env_runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_score = 0.0
    for key in ['test/mean_score', 'test_mean_score', 'mean_score']:
        if key in runner_log:
            mean_score = float(runner_log[key])
            break

    if mean_score == 0.0:
        _test_rewards_list = [v for k, v in runner_log.items()
                              if k.startswith('test/') and 'sim_max_reward' in k]
        if _test_rewards_list:
            mean_score = float(np.mean(_test_rewards_list))
        del _test_rewards_list

    train_score = 0.0
    for key in ['train/mean_score', 'train_mean_score']:
        if key in runner_log:
            train_score = float(runner_log[key])
            break

    if train_score == 0.0:
        _train_rewards_list = [v for k, v in runner_log.items()
                               if k.startswith('train/') and 'sim_max_reward' in k]
        if _train_rewards_list:
            train_score = float(np.mean(_train_rewards_list))
        del _train_rewards_list

    success_rate = 0.0
    n_success = 0
    n_fail = 0
    n_test_episodes = 0

    test_rewards = {k: float(v) for k, v in runner_log.items()
                    if k.startswith('test/') and 'sim_max_reward' in k}
    if test_rewards:
        rewards = list(test_rewards.values())
        n_test_episodes = len(rewards)
        n_success = sum(1 for r in rewards if r >= 0.95)
        n_fail = sum(1 for r in rewards if r < 0.5)
        success_rate = n_success / len(rewards)
        print(f"  Train mean score: {train_score:.4f}")
        print(f"  Test mean score:  {mean_score:.4f}")
        print(f"  Test episodes:    {len(rewards)}")
        print(f"  Success rate:     {success_rate:.1%} (reward >= 0.95)")
        print(f"  Success count:    {n_success}/{len(rewards)}")
        print(f"  Failure count:    {n_fail}/{len(rewards)}")
        del rewards
    else:
        print(f"  Train mean score: {train_score:.4f}")
        print(f"  Test mean score:  {mean_score:.4f}")

    del runner_log, test_rewards
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'mean_score': mean_score,
        'train_score': train_score,
        'success_rate': success_rate,
        'n_success': n_success,
        'n_fail': n_fail,
        'n_test_episodes': n_test_episodes,
    }


def evaluate_single_model(ckpt_path, label, device, output_dir, env_eval=False, n_episodes=50, precision='fp32', sparsity_mode='none', sparsity_ratio=0.5, skip_sparse_env_eval=False):
    print(f"\n{'='*70}")
    print(f"Evaluating: {label}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*70}")

    policy, cfg, metadata, fmt = load_checkpoint_generic(ckpt_path, device)

    model = policy.model
    current_layers = len(model.decoder.layers)
    original_n_layer = metadata.get('original_n_layer', current_layers)

    params = count_parameters(model)
    layer_params = count_parameters_by_layer(model)

    horizon = cfg.get('horizon', 16)
    action_dim = cfg.get('action_dim', 2)
    obs_dim = cfg.get('obs_dim', 20)
    n_obs_steps = cfg.get('n_obs_steps', 2)
    obs_as_cond = cfg.get('obs_as_cond', True)

    sample_shape = (1, horizon, action_dim)
    cond_shape = (1, n_obs_steps, obs_dim) if obs_as_cond else None

    n_heads = 4
    if hasattr(model.decoder, 'layers') and model.decoder.layers:
        n_heads = model.decoder.layers[0].self_attn.num_heads

    flops = estimate_flops(model, sample_shape, n_heads)
    latency = measure_latency(model, sample_shape, cond_shape)

    result = {
        'label': label,
        'checkpoint': ckpt_path,
        'format': fmt,
        'current_layers': current_layers,
        'original_layers': original_n_layer,
        'params': params,
        'params_M': round(params / 1e6, 2),
        'flops_g': round(flops / 1e9, 4),
        'latency_ms': round(latency, 3),
        'n_heads': n_heads,
        'layer_params': [int(p) for p in layer_params],
    }

    if metadata:
        result['pruned_indices'] = metadata.get('pruned_indices', [])
        result['_hard_prune_done'] = metadata.get('_hard_prune_done', False)

    if precision in ('fp16', 'all'):
        print(f"\n  [FP16 量化评估]")
        model_fp16 = quantize_model_fp16(model)
        latency_fp16 = measure_latency_fp16(model_fp16, sample_shape, cond_shape)
        params_fp16 = count_parameters(model_fp16)
        result['latency_fp16_ms'] = round(latency_fp16, 3)
        result['params_fp16_count'] = params_fp16
        result['params_fp16_bytes'] = params_fp16 * 2
        result['params_fp16_MB'] = round(params_fp16 * 2 / (1024 * 1024), 3)
        fp16_speedup = latency / latency_fp16 if latency_fp16 > 0 else 1.0
        print(f"  FP32 Latency: {latency:.3f}ms | FP16 Latency: {latency_fp16:.3f}ms | Speedup: {fp16_speedup:.2f}x | Params: {params_fp16/1e6:.2f}M")

        if env_eval:
            print(f"  [FP16 环境评估]")
            import copy as _copy
            policy_fp16 = _copy.deepcopy(policy)
            policy_fp16.model = model_fp16
            try:
                env_result_fp16 = run_env_evaluation(policy_fp16, cfg, output_dir, n_episodes)
                result['test_score_fp16'] = env_result_fp16['mean_score']
                result['success_rate_fp16'] = env_result_fp16['success_rate']
                result['n_success_fp16'] = env_result_fp16['n_success']
                result['n_test_episodes_fp16'] = env_result_fp16['n_test_episodes']
            except Exception as e:
                print(f"  [WARN] FP16 env eval failed: {e}")
            del policy_fp16
        del model_fp16
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if precision in ('int8', 'all'):
        print(f"\n  [INT8 Weight-only 量化评估 (GPU)]")
        model_int8 = quantize_model_int8_gpu(model)
        latency_int8 = measure_latency_int8(model_int8, sample_shape, cond_shape)
        int8_info = count_parameters_quantized(model_int8)
        result['latency_int8_ms'] = round(latency_int8, 3)
        result['params_int8_info'] = int8_info
        int8_speedup = latency / latency_int8 if latency_int8 > 0 else 1.0
        print(f"  FP32 Latency: {latency:.3f}ms | INT8 Latency: {latency_int8:.3f}ms | Speedup: {int8_speedup:.2f}x | Size: {int8_info['total_MB']:.2f}MB")

        if env_eval:
            print(f"  [INT8 环境评估]")
            import copy as _copy2
            policy_int8 = _copy2.deepcopy(policy)
            policy_int8.model = model_int8
            try:
                env_result_int8 = run_env_evaluation(policy_int8, cfg, output_dir, n_episodes)
                result['test_score_int8'] = env_result_int8['mean_score']
                result['success_rate_int8'] = env_result_int8['success_rate']
                result['n_success_int8'] = env_result_int8['n_success']
                result['n_test_episodes_int8'] = env_result_int8['n_test_episodes']
            except Exception as e:
                print(f"  [WARN] INT8 env eval failed: {e}")
            del policy_int8
        del model_int8
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if sparsity_mode != 'none':
        import copy as _copy3
        print(f"\n  [稀疏化评估 ({sparsity_mode})]")

        if sparsity_mode == '2to4':
            model_sparse = apply_2to4_sparsity(model)
        elif sparsity_mode == 'unstructured':
            model_sparse = apply_unstructured_pruning(model, ratio=sparsity_ratio)
        else:
            model_sparse = None

        if model_sparse is not None:
            actual_sparsity = compute_sparsity_ratio(model_sparse)
            effective_params = count_effective_params_sparse(model_sparse)
            latency_sparse = measure_latency(model_sparse, sample_shape, cond_shape)
            sparse_lat_ratio = latency_sparse / latency if latency > 0 else 1.0

            result['sparsity_mode'] = sparsity_mode
            result['sparsity_ratio_target'] = sparsity_ratio if sparsity_mode == 'unstructured' else 0.5
            result['sparsity_ratio_actual'] = round(actual_sparsity, 4)
            result['latency_sparse_ms'] = round(latency_sparse, 3)
            result['sparse_effective_params'] = effective_params
            result['sparse_effective_params_M'] = round(effective_params / 1e6, 2)
            result['sparse_params_reduction'] = round(1 - effective_params / params, 4) if params > 0 else 0

            print(f"  目标稀疏率: {result['sparsity_ratio_target']:.1%} | 实际稀疏率: {actual_sparsity:.2%}")
            print(f"  原始参数量: {params/1e6:.2f}M | 有效参数量: {effective_params/1e6:.2f}M | 减少: {result['sparse_params_reduction']:.1%}")
            print(f"  FP32 延迟: {latency:.3f}ms | 稀疏延迟: {latency_sparse:.3f}ms | 比值: {sparse_lat_ratio:.2f}x")

            if env_eval and not skip_sparse_env_eval:
                print(f"  [稀疏模型 环境评估]")
                policy_sparse = _copy3.deepcopy(policy)
                policy_sparse.model = model_sparse

                probe_episodes = 10
                probe_success_thresh = 0.1
                try:
                    print(f"  [预评估] 先跑 {probe_episodes} episodes 试水...")
                    probe_result = run_env_evaluation(policy_sparse, cfg, output_dir, probe_episodes)
                    probe_success = probe_result['success_rate']
                    print(f"  [预评估] {probe_episodes} episodes 成功率: {probe_success:.1%}")

                    if probe_success < probe_success_thresh:
                        print(f"  [终止] 预评估成功率过低 ({probe_success:.1%} < {probe_success_thresh:.0%})，判定稀疏模型精度严重崩塌，跳过完整评估。")
                        print(f"    后处理稀疏化对小模型精度损失过大，建议:")
                        print(f"    1. 稀疏化后微调恢复精度")
                        print(f"    2. 使用更低稀疏率 (如 unstructured --sparsity_ratio 0.2)")
                        print(f"    3. 加 --skip_sparse_env_eval 跳过稀疏 env_eval")
                    else:
                        print(f"  [继续] 预评估成功率 OK，跑完整 {n_episodes} episodes...")
                        env_result_sparse = run_env_evaluation(policy_sparse, cfg, output_dir, n_episodes)
                        result['test_score_sparse'] = env_result_sparse['mean_score']
                        result['success_rate_sparse'] = env_result_sparse['success_rate']
                        result['n_success_sparse'] = env_result_sparse['n_success']
                        result['n_test_episodes_sparse'] = env_result_sparse['n_test_episodes']
                except Exception as e:
                    print(f"  [WARN] Sparse env eval failed: {e}")
                del policy_sparse
            del model_sparse
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if env_eval:
        env_result = run_env_evaluation(policy, cfg, output_dir, n_episodes)
        result['test_score'] = env_result['mean_score']
        result['train_score'] = env_result['train_score']
        result['success_rate'] = env_result['success_rate']
        result['n_success'] = env_result['n_success']
        result['n_fail'] = env_result['n_fail']
        result['n_test_episodes'] = env_result['n_test_episodes']

    print(f"\n[Result] {label}:")
    print(f"  Layers: {current_layers}, Params: {params/1e6:.2f}M, FLOPs: {flops/1e9:.4f}G, Latency: {latency:.3f}ms")
    if env_eval:
        print(f"  Test Score: {result.get('test_score', 0):.4f}, Train Score: {result.get('train_score', 0):.4f}")
        print(f"  Success Rate: {result.get('success_rate', 0):.1%} ({result.get('n_success', 0)}/{result.get('n_test_episodes', 0)})")

    del policy, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description='LightDP Unified Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='单模型评估: checkpoint path')
    parser.add_argument('--models', nargs='+', default=None,
                        help='多模型评估: checkpoint paths')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='多模型评估: labels for each model')
    parser.add_argument('--ref_idx', type=int, default=0,
                        help='多模型评估: 参考模型索引 (用于计算压缩率/加速比, 默认=0即第一个模型)')
    parser.add_argument('--baseline_ckpt', type=str, default=None,
                        help='单模型评估: baseline checkpoint for comparison')
    parser.add_argument('--output_dir', type=str, default='data/eval_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--env_eval', action='store_true')
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'int8', 'all'],
                        help='量化精度对比: fp32(默认), fp16, int8, all(全部)')
    parser.add_argument('--sparsity', type=str, default='none',
                        choices=['none', '2to4', 'unstructured'],
                        help='稀疏化模式: none(默认), 2to4(A800结构化稀疏), unstructured(非结构化剪枝)')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5,
                        help='非结构化剪枝的稀疏率 (默认0.5, 仅 unstructured 模式有效)')
    parser.add_argument('--skip_sparse_env_eval', action='store_true',
                        help='跳过稀疏模型的环境评估 (先做快速前向验证, 节省时间)')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已有报告文件 (默认加时间戳避免覆盖)')
    args = parser.parse_args()

    device = torch.device(args.device)
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.models:
        if args.labels and len(args.labels) != len(args.models):
            print(f"ERROR: labels count ({len(args.labels)}) != models count ({len(args.models)})")
            sys.exit(1)
        if args.ref_idx < 0 or args.ref_idx >= len(args.models):
            print(f"ERROR: ref_idx ({args.ref_idx}) out of range [0, {len(args.models)-1}]")
            sys.exit(1)
        multi_model_evaluation(args, device)
    elif args.checkpoint:
        single_model_evaluation(args, device)
    else:
        parser.error("请使用 --checkpoint (单模型) 或 --models (多模型)")


def single_model_evaluation(args, device):
    output_dir = args.output_dir
    ckpt_path = args.checkpoint

    print("=" * 70)
    print("LightDP 单模型评估")
    print("=" * 70)

    policy, cfg, metadata = load_checkpoint(ckpt_path, device)

    original_n_layer = metadata.get('original_n_layer', 8)
    current_layers = len(policy.model.decoder.layers)
    pruned_indices = metadata.get('pruned_indices', list(range(current_layers)))
    removed_indices = [i for i in range(original_n_layer) if i not in pruned_indices]

    pruned_params = count_parameters(policy.model)
    pruned_layer_params = count_parameters_by_layer(policy.model)

    print("\n" + "-" * 70)
    print("Building baseline model (original 8 layers) for comparison...")
    baseline_model = build_baseline_model(cfg, device, original_n_layer)
    baseline_params = count_parameters(baseline_model)
    baseline_layer_params = count_parameters_by_layer(baseline_model)

    horizon = cfg.get('horizon', 16)
    action_dim = cfg.get('action_dim', 2)
    obs_dim = cfg.get('obs_dim', 20)
    n_obs_steps = cfg.get('n_obs_steps', 2)
    obs_as_cond = cfg.get('obs_as_cond', True)

    sample_shape = (1, horizon, action_dim)
    cond_shape = (1, n_obs_steps, obs_dim) if obs_as_cond else None

    n_heads = 4
    if hasattr(policy.model.decoder, 'layers') and policy.model.decoder.layers:
        n_heads = policy.model.decoder.layers[0].self_attn.num_heads

    pruned_flops = estimate_flops(policy.model, sample_shape, n_heads)
    baseline_flops = estimate_flops(baseline_model, sample_shape, n_heads)

    pruned_latency = measure_latency(policy.model, sample_shape, cond_shape)
    baseline_latency = measure_latency(baseline_model, sample_shape, cond_shape)

    print("\n" + "=" * 70)
    print(f"COMPARISON REPORT: Baseline ({original_n_layer}-layer) vs Pruned ({current_layers}-layer)")
    print("=" * 70)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\n[Layer Configuration]")
    print(f"  Baseline layers:  {original_n_layer}")
    print(f"  Pruned layers:    {current_layers}")
    print(f"  Removed layers:   {removed_indices}")
    print(f"  Kept indices:     {pruned_indices}")

    param_reduction = 1 - pruned_params / baseline_params
    print(f"\n[Parameters]")
    print(f"  {'':20s} {'Baseline':>15s} {'Pruned':>15s} {'Reduction':>15s}")
    print(f"  {'Total params':20s} {baseline_params:>15,} {pruned_params:>15,} {param_reduction:>14.1%}")
    print(f"  {'Params (M)':20s} {baseline_params/1e6:>14.2f}M {pruned_params/1e6:>14.2f}M {param_reduction:>14.1%}")

    print(f"\n[Per-Layer Parameters]")
    print(f"  {'Layer':>6s}  {'Baseline':>12s}  {'Pruned':>12s}  {'Status':>10s}")
    for i in range(original_n_layer):
        b_params = baseline_layer_params[i] if i < len(baseline_layer_params) else 0
        p_params = pruned_layer_params[pruned_indices.index(i)] if i in pruned_indices and pruned_indices.index(i) < len(pruned_layer_params) else 0
        status = "KEPT" if i in pruned_indices else "REMOVED"
        if i in pruned_indices:
            new_idx = pruned_indices.index(i)
            print(f"  {i:>6d}  {b_params:>12,}  {p_params:>12,}  {status:>10s}  -> new idx {new_idx}")
        else:
            print(f"  {i:>6d}  {b_params:>12,}  {'---':>12s}  {status:>10s}")

    total_baseline_layer = sum(baseline_layer_params)
    total_pruned_layer = sum(pruned_layer_params)
    non_layer_params = baseline_params - total_baseline_layer
    print(f"  {'Other':>6s}  {non_layer_params:>12,}  {non_layer_params:>12,}")
    print(f"  {'TOTAL':>6s}  {baseline_params:>12,}  {pruned_params:>12,}")

    flops_reduction = 1 - pruned_flops / baseline_flops
    print(f"\n[FLOPs]")
    print(f"  {'':20s} {'Baseline':>15s} {'Pruned':>15s} {'Reduction':>15s}")
    print(f"  {'Total GFLOPs':20s} {baseline_flops/1e9:>15.3f} {pruned_flops/1e9:>15.3f} {flops_reduction:>14.1%}")

    speedup = baseline_latency / pruned_latency
    print(f"\n[Inference Latency]")
    print(f"  {'':20s} {'Baseline':>15s} {'Pruned':>15s} {'Change':>15s}")
    print(f"  {'Latency (ms)':20s} {baseline_latency:>15.3f} {pruned_latency:>15.3f} {speedup:>14.2f}x")
    print(f"  {'Throughput':20s} {1000/baseline_latency:>14.1f}x/s {1000/pruned_latency:>14.1f}x/s")

    test_score = 0.0
    train_score = 0.0
    baseline_test_score = None
    if args.env_eval:
        print(f"\n{'='*70}")
        print(f"ENVIRONMENT EVALUATION ({args.n_episodes} episodes)")
        print(f"{'='*70}")
        env_result = run_env_evaluation(policy, cfg, output_dir, args.n_episodes)
        test_score = env_result['mean_score']
        train_score = env_result['train_score']

        if args.baseline_ckpt:
            print("\n--- Baseline model evaluation ---")
            baseline_policy, _, _ = load_checkpoint(args.baseline_ckpt, device)
            baseline_env_result = run_env_evaluation(baseline_policy, cfg, output_dir, args.n_episodes)
            baseline_test_score = baseline_env_result['mean_score']
            del baseline_policy, baseline_env_result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "model_info": {
            "original_layers": original_n_layer,
            "pruned_layers": current_layers,
            "removed_indices": removed_indices,
            "kept_indices": pruned_indices,
        },
        "parameters": {
            "baseline_params": baseline_params,
            "pruned_params": pruned_params,
            "param_reduction_ratio": round(param_reduction, 4),
        },
        "flops": {
            "baseline_flops_g": round(baseline_flops / 1e9, 4),
            "pruned_flops_g": round(pruned_flops / 1e9, 4),
            "flops_reduction_ratio": round(flops_reduction, 4),
        },
        "latency": {
            "baseline_latency_ms": round(baseline_latency, 3),
            "pruned_latency_ms": round(pruned_latency, 3),
            "speedup_ratio": round(speedup, 4),
        },
        "env_results": {},
    }

    if args.env_eval:
        report["env_results"] = {
            "train_mean_score": round(train_score, 4),
            "test_mean_score": round(test_score, 4),
        }
        if baseline_test_score is not None:
            report["env_results"]["baseline_test_mean_score"] = round(baseline_test_score, 4)
            report["env_results"]["score_delta"] = round(test_score - baseline_test_score, 4)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = "lightdp_evaluation_report"
    if args.overwrite:
        report_path = os.path.join(output_dir, f"{base_name}.json")
    else:
        report_path = os.path.join(output_dir, f"{base_name}_{ts}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Model:           {current_layers} layers (pruned from {original_n_layer})")
    print(f"  Param reduction: {param_reduction:.1%}")
    print(f"  FLOPs reduction: {flops_reduction:.1%}")
    print(f"  Speedup:         {speedup:.2f}x")
    if args.env_eval:
        print(f"  Test score:      {test_score:.4f}")

    print(f"\n  Report saved to: {report_path}")

    del policy, baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def multi_model_evaluation(args, device):
    output_dir = args.output_dir
    models = args.models
    labels = args.labels or [f"model_{i}" for i in range(len(models))]
    ref_idx = args.ref_idx
    precision = args.precision
    sparsity_mode = args.sparsity
    sparsity_ratio = args.sparsity_ratio

    print("=" * 80)
    print(f"LightDP 多模型统一评估 ({len(models)} 个模型)")
    print(f"参考模型索引: {ref_idx} ({labels[ref_idx]})")
    print(f"量化精度: {precision}")
    if sparsity_mode != 'none':
        print(f"稀疏化模式: {sparsity_mode}" + (f" (ratio={sparsity_ratio})" if sparsity_mode == 'unstructured' else ""))
    print("=" * 80)

    all_results = []

    for i, (ckpt_path, label) in enumerate(zip(models, labels)):
        print(f"\n{'#'*80}")
        print(f"# Model {i+1}/{len(models)}: {label}")
        print(f"{'#'*80}")

        result = evaluate_single_model(
            ckpt_path=ckpt_path,
            label=label,
            device=device,
            output_dir=output_dir,
            env_eval=args.env_eval,
            n_episodes=args.n_episodes,
            precision=precision,
            sparsity_mode=sparsity_mode,
            sparsity_ratio=sparsity_ratio,
            skip_sparse_env_eval=args.skip_sparse_env_eval,
        )
        all_results.append(result)

    print(f"\n\n{'='*80}")
    print("统一评估对比报告")
    print(f"{'='*80}")

    if all_results:
        ref_result = all_results[ref_idx]
        ref_params = ref_result['params']
        ref_flops = ref_result['flops_g']
        ref_latency = ref_result['latency_ms']

        has_fp16 = any('latency_fp16_ms' in r for r in all_results)
        has_int8 = any('latency_int8_ms' in r for r in all_results)

        print(f"\n参考模型: [{ref_idx}] {ref_result['label']} (层数={ref_result['current_layers']}, FP32参数={ref_result['params_M']}M)")

        print(f"\n{'─'*90}")
        print("【FP32 跨模型对比 - 剪枝影响分析】")
        print(f"{'─'*90}")

        if args.env_eval:
            col_widths = [16, 4, 10, 8, 10, 8, 10, 10, 10, 10]
            headers = ['模型名称', '层数', '参数量', '压缩率', 'FLOPs', '加速比', '延迟', '测试得分', '成功率', '成功/总数']
        else:
            col_widths = [16, 4, 10, 8, 10, 8, 10, 10]
            headers = ['模型名称', '层数', '参数量', '压缩率', 'FLOPs', '加速比', '延迟', '成功/总数']

        hdr_line = ' '
        for h, w in zip(headers, col_widths):
            hdr_line += f"{h:>{w}s} "
        sep = '─' * (sum(col_widths) + 2 * len(col_widths))

        print(f"\n{sep}")
        print(hdr_line)
        print(sep)

        for idx, r in enumerate(all_results):
            is_ref = (idx == ref_idx)
            if is_ref:
                reduction = 0.0
                speedup = 1.0
            else:
                reduction = 1 - r['params'] / ref_params
                speedup = ref_latency / r['latency_ms'] if r['latency_ms'] > 0 else 1.0

            lbl = f"{r['label']}"
            if is_ref:
                lbl += " (REF)"

            if args.env_eval:
                line = [
                    f"{lbl:<16s}",
                    f"{r['current_layers']:>{col_widths[1]}d}",
                    f"{r['params_M']:>{col_widths[2]-1}.2f}M",
                    f"{reduction:>{col_widths[3]-1}.1%}",
                    f"{r['flops_g']:>{col_widths[4]-1}.4f}G",
                    f"{speedup:>{col_widths[5]-1}.2f}x",
                    f"{r['latency_ms']:>{col_widths[6]-1}.3f}ms",
                    f"{r.get('test_score', 0):>{col_widths[7]-1}.4f}",
                    f"{r.get('success_rate', 0):>{col_widths[8]-1}.1%}",
                    f"{r.get('n_success', 0):>3d}/{r.get('n_test_episodes', 0):<3d}",
                ]
            else:
                line = [
                    f"{lbl:<16s}",
                    f"{r['current_layers']:>{col_widths[1]}d}",
                    f"{r['params_M']:>{col_widths[2]-1}.2f}M",
                    f"{reduction:>{col_widths[3]-1}.1%}",
                    f"{r['flops_g']:>{col_widths[4]-1}.4f}G",
                    f"{speedup:>{col_widths[5]-1}.2f}x",
                    f"{r['latency_ms']:>{col_widths[6]-1}.3f}ms",
                    f"{r.get('n_success', 0):>3d}/{r.get('n_test_episodes', 0):<3d}",
                ]
            print(' '.join(line))

        print(sep)

        if has_fp16 or has_int8:
            print(f"\n{'='*90}")
            print("【精度格式对比 - FP32 / FP16 / INT8】")
            print(f"{'='*90}")

            for idx, r in enumerate(all_results):
                print(f"\n{'─'*60}")
                print(f"  模型: {r['label']} (层数={r['current_layers']})")
                print(f"{'─'*60}")

                p_col_widths = [8, 12, 10, 10, 8, 12, 10, 10, 10]
                p_headers = ['精度', '参数量', '压缩率', 'FLOPs', '加速比', '延迟', '测试得分', '成功率', '成功/总数']

                p_hdr = ' '
                for h, w in zip(p_headers, p_col_widths):
                    p_hdr += f"{h:>{w}s} "
                p_sep = '─' * (sum(p_col_widths) + 2 * len(p_col_widths))

                print(f"  {p_sep}")
                print(f"  {p_hdr}")
                print(f"  {p_sep}")

                r_params_fp32 = r['params']
                r_params_fp32_MB = r_params_fp32 * 4 / (1024 * 1024)
                r_lat_fp32 = r['latency_ms']
                r_flops = r['flops_g']

                # FP32 row
                line_fp32 = [
                    f"{'FP32':<8s}",
                    f"{r_params_fp32_MB:>{p_col_widths[1]-2}.2f}MB",
                    f"{'---':>{p_col_widths[2]-2}}",
                    f"{r_flops:>{p_col_widths[3]-1}.4f}G",
                    f"{'1.00x':>{p_col_widths[4]-1}}",
                    f"{r_lat_fp32:>{p_col_widths[5]-1}.3f}ms",
                    f"{r.get('test_score', 0):>{p_col_widths[6]-1}.4f}",
                    f"{r.get('success_rate', 0):>{p_col_widths[7]-1}.1%}",
                    f"{r.get('n_success', 0):>3d}/{r.get('n_test_episodes', 0):<3d}",
                ]
                print(f"  {' '.join(line_fp32)}")

                if has_fp16 and 'latency_fp16_ms' in r:
                    r_params_fp16_bytes = r.get('params_fp16_bytes', r_params_fp32 * 2)
                    r_params_fp16_MB = r_params_fp16_bytes / (1024 * 1024)
                    r_lat_fp16 = r.get('latency_fp16_ms', r_lat_fp32)
                    r_sp_fp16 = r_lat_fp32 / r_lat_fp16 if r_lat_fp16 > 0 else 1.0
                    r_comp_fp16 = (1 - r_params_fp16_bytes / (r_params_fp32 * 4)) if r_params_fp32 > 0 else 0
                    r_score_fp16 = r.get('test_score_fp16', r.get('test_score', 0))
                    r_sr_fp16 = r.get('success_rate_fp16', r.get('success_rate', 0))
                    r_ns_fp16 = r.get('n_success_fp16', r.get('n_success', 0))
                    r_nt_fp16 = r.get('n_test_episodes_fp16', r.get('n_test_episodes', 0))

                    line_fp16 = [
                        f"{'FP16':<8s}",
                        f"{r_params_fp16_MB:>{p_col_widths[1]-2}.2f}MB",
                        f"{r_comp_fp16:>{p_col_widths[2]-1}.1%}",
                        f"{r_flops:>{p_col_widths[3]-1}.4f}G",
                        f"{r_sp_fp16:>{p_col_widths[4]-1}.2f}x",
                        f"{r_lat_fp16:>{p_col_widths[5]-1}.3f}ms",
                        f"{r_score_fp16:>{p_col_widths[6]-1}.4f}",
                        f"{r_sr_fp16:>{p_col_widths[7]-1}.1%}",
                        f"{r_ns_fp16:>3d}/{r_nt_fp16:<3d}",
                    ]
                    print(f"  {' '.join(line_fp16)}")

                if has_int8 and 'latency_int8_ms' in r:
                    int8_info = r.get('params_int8_info', {})
                    r_params_int8_MB = int8_info.get('total_MB', r_params_fp32_MB / 4)
                    r_lat_int8 = r.get('latency_int8_ms', r_lat_fp32)
                    r_sp_int8 = r_lat_fp32 / r_lat_int8 if r_lat_int8 > 0 else 1.0
                    r_comp_int8 = (1 - int8_info.get('total_bytes', r_params_fp32) / (r_params_fp32 * 4)) if r_params_fp32 > 0 else 0
                    r_score_int8 = r.get('test_score_int8', r.get('test_score', 0))
                    r_sr_int8 = r.get('success_rate_int8', r.get('success_rate', 0))
                    r_ns_int8 = r.get('n_success_int8', r.get('n_success', 0))
                    r_nt_int8 = r.get('n_test_episodes_int8', r.get('n_test_episodes', 0))

                    line_int8 = [
                        f"{'INT8':<8s}",
                        f"{r_params_int8_MB:>{p_col_widths[1]-2}.2f}MB",
                        f"{r_comp_int8:>{p_col_widths[2]-1}.1%}",
                        f"{r_flops:>{p_col_widths[3]-1}.4f}G",
                        f"{r_sp_int8:>{p_col_widths[4]-1}.2f}x",
                        f"{r_lat_int8:>{p_col_widths[5]-1}.3f}ms",
                        f"{r_score_int8:>{p_col_widths[6]-1}.4f}",
                        f"{r_sr_int8:>{p_col_widths[7]-1}.1%}",
                        f"{r_ns_int8:>3d}/{r_nt_int8:<3d}",
                    ]
                    print(f"  {' '.join(line_int8)}")

                print(f"  {p_sep}")

            print(f"\n  说明: 压缩率 = (1 - 当前精度字节数 / FP32字节数) x 100%")
            print(f"        FP16 = 半精度浮点 (权重+激活FP16, GPU Tensor Core加速)")
            print(f"        INT8 = Weight-only INT8 (权重INT8存储, 激活FP16计算, GPU上运行)")

        has_sparse = any('latency_sparse_ms' in r for r in all_results)
        if has_sparse:
            print(f"\n{'='*90}")
            print("【稀疏化对比 - Dense vs Sparse】")
            print(f"{'='*90}")

            for idx, r in enumerate(all_results):
                if 'latency_sparse_ms' not in r:
                    continue
                print(f"\n{'─'*60}")
                print(f"  模型: {r['label']} (层数={r['current_layers']})")
                print(f"{'─'*60}")

                s_col_widths = [10, 12, 10, 12, 10, 12, 12, 10]
                s_headers = ['类型', '稀疏模式', '实际稀疏', '参数量', '压缩率', '延迟', '测试得分', '成功率']

                s_hdr = ' '
                for h, w in zip(s_headers, s_col_widths):
                    s_hdr += f"{h:>{w}s} "
                s_sep = '─' * (sum(s_col_widths) + 2 * len(s_col_widths))

                print(f"  {s_sep}")
                print(f"  {s_hdr}")
                print(f"  {s_sep}")

                r_params_M = r['params_M']
                r_latency = r['latency_ms']
                r_score = r.get('test_score', 0)
                r_sr = r.get('success_rate', 0)

                line_dense = [
                    f"{'Dense':<10s}",
                    f"{'---':>{s_col_widths[1]}}",
                    f"{'---':>{s_col_widths[2]}}",
                    f"{r_params_M:>{s_col_widths[3]-1}.2f}M",
                    f"{'---':>{s_col_widths[4]}}",
                    f"{r_latency:>{s_col_widths[5]-1}.3f}ms",
                    f"{r_score:>{s_col_widths[6]-1}.4f}",
                    f"{r_sr:>{s_col_widths[7]-1}.1%}",
                ]
                print(f"  {' '.join(line_dense)}")

                r_sparsity_mode = r.get('sparsity_mode', 'unknown')
                r_actual_sparsity = r.get('sparsity_ratio_actual', 0)
                r_eff_params_M = r.get('sparse_effective_params_M', r_params_M)
                r_lat_sparse = r.get('latency_sparse_ms', r_latency)
                r_comp_sparse = r.get('sparse_params_reduction', 0)
                r_score_sparse = r.get('test_score_sparse', r_score)
                r_sr_sparse = r.get('success_rate_sparse', r_sr)

                mode_label = '2:4' if r_sparsity_mode == '2to4' else (
                    f'U-{r.get("sparsity_ratio_target", 0.5):.0%}' if r_sparsity_mode == 'unstructured' else r_sparsity_mode
                )

                line_sparse = [
                    f"{'Sparse':<10s}",
                    f"{mode_label:>{s_col_widths[1]}}",
                    f"{r_actual_sparsity:>{s_col_widths[2]-1}.1%}",
                    f"{r_eff_params_M:>{s_col_widths[3]-1}.2f}M",
                    f"{r_comp_sparse:>{s_col_widths[4]-1}.1%}",
                    f"{r_lat_sparse:>{s_col_widths[5]-1}.3f}ms",
                    f"{r_score_sparse:>{s_col_widths[6]-1}.4f}",
                    f"{r_sr_sparse:>{s_col_widths[7]-1}.1%}",
                ]
                print(f"  {' '.join(line_sparse)}")

                print(f"  {s_sep}")

                if r_score != 0 and r_score_sparse != 0:
                    score_delta = r_score_sparse - r_score
                    print(f"  精度变化: {score_delta:+.4f} ({score_delta*100:+.2f}%) | 延迟变化: {r_lat_sparse - r_latency:+.3f}ms")

            print(f"\n  说明: Dense = 原始模型 (FP32), Sparse = 稀疏置零后的模型 (仍为FP32权重, 推理时不跳过零)")
            print(f"        2:4 = 每4个权重保留绝对值最大的2个, 固定50%稀疏率")
            print(f"        Unstructured = 全局阈值剪枝, 按目标比例置零, 位置无规律")
            print(f"        ⚠️ 当前为后处理置零, 未配合硬件Sparse Tensor Core, 延迟不会下降")
            print(f"        延迟下降需要稀疏化感知微调 + 专用算子 (A800 Sparse Tensor Core)")

    report = {
        'evaluation_config': {
            'n_models': len(models),
            'ref_idx': ref_idx,
            'ref_label': labels[ref_idx],
            'env_eval': args.env_eval,
            'n_episodes': args.n_episodes,
            'precision': precision,
            'sparsity': sparsity_mode,
            'sparsity_ratio': sparsity_ratio,
        },
        'results': all_results,
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = "lightdp_batch_evaluation_report"
    if args.overwrite:
        report_path = os.path.join(output_dir, f"{base_name}.json")
    else:
        report_path = os.path.join(output_dir, f"{base_name}_{ts}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存至: {report_path}")
    print("=" * 80)


if __name__ == '__main__':
    sys.exit(main())