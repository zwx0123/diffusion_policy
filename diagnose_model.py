#!/usr/bin/env python3
"""Diagnose model performance in PushT environment."""
import os
import sys
import torch
import numpy as np
import hydra
import copy
from omegaconf import OmegaConf

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    ckpt_path = 'data/outputs/2026.08.20/15.06.22_train_diffusion_transformer_lightdp_pusht_lowdim/checkpoints/final.ckpt'
    device = 'cuda:0'
    
    print(f"Loading checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'))
    cfg = payload['cfg']
    metadata = payload.get('metadata', {})
    
    print(f"Metadata: {metadata}")
    
    # Check state dict keys to understand EMA structure
    state_dicts = payload.get('state_dicts', {})
    for k in ['ema_model', 'policy']:
        if k in state_dicts:
            sd = state_dicts[k]
            decoder_keys = [key for key in sd.keys() if 'decoder.layers' in key]
            layer_indices = sorted(set(int(key.split('.')[2]) for key in decoder_keys))
            print(f"\n{k} decoder layers: {layer_indices}")
            gate_logits = sd.get('pruning_module.gate_logits', None)
            if gate_logits is not None:
                print(f"{k} gate_logits shape: {gate_logits.shape}")
    
    # Build model
    from diffusion_policy.policy.diffusion_transformer_lightdp_policy import DiffusionTransformerLightDPPolicy
    
    original_n_layer = metadata.get('original_n_layer', 8)
    current_n_layer = metadata.get('current_n_layer', 6)
    _hard_prune_done = metadata.get('_hard_prune_done', False)
    pruned_indices = metadata.get('pruned_indices', [0, 1, 2, 3, 6, 7])
    
    n_layer_override = current_n_layer if _hard_prune_done and current_n_layer > 0 else None
    
    policy = DiffusionTransformerLightDPPolicy(
        **cfg.policy.model,
        target_layers=cfg.policy.model.get('target_layers', 8),
        warmup_epochs=cfg.policy.model.get('warmup_epochs', 30),
        device=device
    )
    
    if n_layer_override is not None and n_layer_override != cfg.policy.model.get('num_layers', 8):
        policy.model.n_layer = n_layer_override
        from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
        policy.model.decoder.layers = policy.model.decoder.layers[:n_layer_override]
    
    policy.to(device)
    policy.train()
    
    print(f"\nBuilt model: {len(policy.model.decoder.layers)} layers")
    
    # Load EMA weights
    if 'ema_model' in state_dicts:
        model_state_dict = state_dicts['ema_model']
    else:
        model_state_dict = state_dicts.get('policy', {})
    
    # Filter and load
    filtered_sd = {}
    for k, v in model_state_dict.items():
        if k.startswith('normalizer.') or k == 'pruning_module.gate_logits':
            continue
        filtered_sd[k] = v
    
    missing, unexpected = policy.load_state_dict(filtered_sd, strict=False)
    print(f"\nLoaded weights: missing={missing[:3] if missing else 'none'}, unexpected={unexpected[:3] if unexpected else 'none'}")
    
    # Load normalizer
    if 'pickles' in payload and 'normalizer' in payload['pickles']:
        normalizer_state_dict = payload['pickles']['normalizer']
        policy.normalizer.load_state_dict(normalizer_state_dict)
        print("Normalizer loaded from pickles")
    else:
        print("WARNING: No normalizer in pickles!")
    
    # Handle gate_logits
    gate_logits = model_state_dict.get('pruning_module.gate_logits', None)
    if gate_logits is not None:
        print(f"EMA gate_logits: {gate_logits}")
        if len(gate_logits) == original_n_layer and current_n_layer < original_n_layer:
            new_logits = gate_logits[pruned_indices].clone()
            policy.pruning_module.gate_logits = torch.nn.Parameter(new_logits)
            policy.pruning_module.num_layers = len(pruned_indices)
            print(f"Remapped gate_logits: [{original_n_layer}] -> [{len(pruned_indices)}]")
    
    # Set pruned state
    policy._hard_prune_done = True
    policy.pruned_indices = list(range(current_n_layer))
    policy.current_n_layer = current_n_layer
    policy.original_n_layer = original_n_layer
    policy.pruning_module.gate_logits.data = torch.ones(current_n_layer) * 10.0
    
    policy.eval()
    
    # Test 1: Quick inference with mean observation
    print(f"\n{'='*60}")
    print("TEST 1: Quick inference with mean observation")
    print(f"{'='*60}")
    
    obs_norm = policy.normalizer['obs']
    obs_stats = obs_norm.params_dict
    obs_mean = obs_stats['mean'].data
    obs_std = obs_stats['std'].data
    
    print(f"obs mean: {obs_mean}")
    print(f"obs std: {obs_std}")
    
    test_obs = obs_mean.unsqueeze(0).unsqueeze(0).expand(1, 2, -1).clone()
    test_obs += torch.randn_like(test_obs) * obs_std * 0.1
    test_obs = test_obs.to(device)
    
    print(f"test_obs shape: {test_obs.shape}")
    print(f"test_obs range: [{test_obs.min().item():.2f}, {test_obs.max().item():.2f}]")
    
    with torch.no_grad():
        result = policy.predict_action({'obs': test_obs})
    
    action = result['action']
    print(f"Output action shape: {action.shape}")
    print(f"Output action range: [{action.min().item():.4f}, {action.max().item():.4f}]")
    print(f"Output action mean: {action.mean().item():.4f}, std: {action.std().item():.4f}")
    
    action_norm = policy.normalizer['action']
    action_stats = action_norm.params_dict
    print(f"Action valid range: [{action_stats['min'].data.min().item():.1f}, {action_stats['max'].data.max().item():.1f}]")
    
    # Test 2: Run in environment
    print(f"\n{'='*60}")
    print("TEST 2: Run in PushT environment (5 episodes)")
    print(f"{'='*60}")
    
    try:
        from diffusion_policy.env_runner.pusht_keypoints_runner import PushTKeypointsRunner
        
        eval_dir = '/tmp/pusht_diagnose'
        os.makedirs(eval_dir, exist_ok=True)
        
        runner = PushTKeypointsRunner(
            output_dir=eval_dir,
            n_test=5,
            n_obs_steps=2,
            n_latency_steps=0
        )
        
        import wandb
        wandb.init(mode='disabled')
        
        runner_log = runner.run(policy)
        
        for key, value in sorted(runner_log.items()):
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            elif hasattr(value, '__float__'):
                print(f"  {key}: {float(value):.4f}")
        
        mean_score = runner_log.get('test/mean_score', 0)
        print(f"\n  Mean score: {mean_score:.4f}")
        
    except Exception as e:
        print(f"Error in env test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check if weights are properly loaded
    print(f"\n{'='*60}")
    print("TEST 3: Weight verification")
    print(f"{'='*60}")
    
    # Compare EMA weights with loaded weights
    ema_sd = state_dicts.get('ema_model', {})
    policy_sd = policy.state_dict()
    
    for key in ema_sd:
        if 'decoder.layers' in key and 'weight' in key:
            if key in policy_sd:
                if not torch.equal(ema_sd[key].cpu(), policy_sd[key].cpu()):
                    print(f"  MISMATCH: {key}")
                    print(f"    EMA: mean={ema_sd[key].mean().item():.6f}, std={ema_sd[key].std().item():.6f}")
                    print(f"    Loaded: mean={policy_sd[key].mean().item():.6f}, std={policy_sd[key].std().item():.6f}")
                else:
                    print(f"  OK: {key} (shape={ema_sd[key].shape})")
    
    print(f"\nDiagnosis complete!")

if __name__ == '__main__':
    main()