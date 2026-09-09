"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output -d cuda:0 --precision fp32
python eval.py --checkpoint data/image/pusht/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output -d cuda:0 --precision fp16
python eval.py --checkpoint data/image/pusht/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output -d cuda:0 --precision int8
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import numpy as np
import pandas as pd
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--warmup', default=10, help='Number of warmup iterations before timing')
@click.option('--timing-runs', default=100, help='Number of forward passes for timing measurement')
@click.option('--enable-timing/--disable-timing', default=True, help='Enable or disable block-level timing')
@click.option('--precision', default='fp32', type=click.Choice(['fp32', 'fp16', 'int8']), help='Model precision for timing')
def main(checkpoint, output_dir, device, warmup, timing_runs, enable_timing, precision):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if enable_timing:
        timing_dir = os.path.join(output_dir, 'timing_results')
        pathlib.Path(timing_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

     # ===== 新增：打印剪枝信息 =====
    print("\n" + "="*60)
    print("MODEL PRUNING INFO")
    print("="*60)
    
    if hasattr(policy, 'get_pruning_info'):
        info = policy.get_pruning_info()
        if info.get('is_pruned', False):
            print(f"  Pruned: YES")
            print(f"  Original layers: {info['n_layers']}")
            print(f"  Kept layers: {info['kept_layers']}")
            print(f"  Pruned layers: {info['pruned_layers']}")
            print(f"  Keep ratio: {info['keep_ratio']:.2f}")
            print(f"  Kept indices: {info['kept_indices']}")
        else:
            print(f"  Pruned: NO")
            print(f"  Total layers: {info.get('n_layers', 'unknown')}")
    else:
        # 尝试从 transformer 获取
        if hasattr(policy, 'model') and hasattr(policy.model, 'get_pruning_info'):
            info = policy.model.get_pruning_info()
            print(f"  Pruned: {info.get('is_pruned', False)}")
            if info.get('is_pruned', False):
                print(f"  Kept layers: {info['kept_layers']}/{info['n_layers']}")
                print(f"  Kept indices: {info['kept_indices']}")
    print("="*60 + "\n")
    # ==============================
    
    # ============================================================
    # 精度转换
    # ============================================================
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
    }
    
    if precision == 'fp16':
        print(f"\nConverting model to FP16...")
        policy = policy.half()  # 转换所有参数为 fp16
    elif precision == 'int8':
        print(f"\nConverting model to INT8 (dynamic quantization)...")
        # 动态量化：只量化 Linear 层，推理时动态量化激活值
        policy = torch.quantization.quantize_dynamic(
            policy, 
            {torch.nn.Linear, torch.nn.MultiheadAttention},  # 量化这些层
            dtype=torch.qint8
        )
        print("Dynamic quantization complete.")
    # ============================================================
    
    # get model dimension info
    horizon = None
    action_dim = None
    obs_dim = None
    n_obs_steps = None
    
    if hasattr(policy, 'horizon'):
        horizon = policy.horizon
    if hasattr(policy, 'action_dim'):
        action_dim = policy.action_dim
    if hasattr(policy, 'obs_dim'):
        obs_dim = policy.obs_dim
    if hasattr(policy, 'n_obs_steps'):
        n_obs_steps = policy.n_obs_steps
    
    if horizon is None and hasattr(cfg, 'horizon'):
        horizon = cfg.horizon
    if action_dim is None and hasattr(cfg, 'action_dim'):
        action_dim = cfg.action_dim
    if obs_dim is None and hasattr(cfg, 'cond_dim'):
        obs_dim = cfg.cond_dim
    if n_obs_steps is None and hasattr(cfg, 'n_obs_steps'):
        n_obs_steps = cfg.n_obs_steps
    
    if horizon is None and hasattr(policy, 'model'):
        if hasattr(policy.model, 'horizon'):
            horizon = policy.model.horizon
    if action_dim is None and hasattr(policy, 'model'):
        if hasattr(policy.model, 'input_emb'):
            action_dim = policy.model.input_emb.in_features
    if obs_dim is None and hasattr(policy, 'model'):
        if hasattr(policy.model, 'cond_obs_emb') and policy.model.cond_obs_emb is not None:
            obs_dim = policy.model.cond_obs_emb.in_features
    
    print(f"Model info - horizon: {horizon}, action_dim: {action_dim}, obs_dim: {obs_dim}, n_obs_steps: {n_obs_steps}")
    print(f"Precision: {precision}")
    
    # CUDA timing
    if enable_timing and device.type == 'cuda':
        print("\n" + "="*60)
        print(f"CUDA TIMING MODE - {precision.upper()}")
        print("="*60)
        
        if hasattr(policy, 'model'):
            transformer_model = policy.model
        else:
            transformer_model = policy
        
        if not hasattr(transformer_model, 'enable_cuda_timing'):
            print("WARNING: Model does not have enable_cuda_timing method.")
            print("Please modify transformer_for_diffusion.py with timing support first.")
            print("Skipping block-level timing...")
        else:
            transformer_model.enable_cuda_timing()
            
            batch_size = 1
            use_cond = False
            if hasattr(transformer_model, 'obs_as_cond') and transformer_model.obs_as_cond:
                use_cond = True
            elif obs_dim is not None and obs_dim > 0 and n_obs_steps is not None:
                use_cond = True
            
            if horizon is None:
                horizon = 16
            if action_dim is None:
                action_dim = 2
            
            # 创建对应精度的 dummy 输入
            if precision == 'fp16':
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            dummy_sample = torch.randn(batch_size, horizon, action_dim, device=device, dtype=dtype)
            
            if use_cond:
                if n_obs_steps is None:
                    n_obs_steps = 2
                if obs_dim is None:
                    obs_dim = 10
                dummy_cond = torch.randn(batch_size, n_obs_steps, obs_dim, device=device, dtype=dtype)
            else:
                dummy_cond = None
            
            # warmup
            print(f"\nWarming up for {warmup} iterations...")
            with torch.no_grad():
                for i in range(warmup):
                    timestep = torch.tensor(i % 100, device=device)
                    if use_cond:
                        _ = transformer_model(dummy_sample, timestep, dummy_cond)
                    else:
                        _ = transformer_model(dummy_sample, timestep)
                    torch.cuda.synchronize()
            print("Warmup complete.")
            
            # timing runs
            print(f"\nRunning {timing_runs} timing iterations...")
            transformer_model.timing_stats.clear()
            total_times = []
            
            with torch.no_grad():
                for i in range(timing_runs):
                    timestep = torch.tensor(i % 100, device=device)
                    
                    torch.cuda.synchronize()
                    iter_start = torch.cuda.Event(enable_timing=True)
                    iter_end = torch.cuda.Event(enable_timing=True)
                    iter_start.record()
                    
                    if use_cond:
                        _ = transformer_model(dummy_sample, timestep, dummy_cond)
                    else:
                        _ = transformer_model(dummy_sample, timestep)
                    
                    iter_end.record()
                    torch.cuda.synchronize()
                    total_time = iter_start.elapsed_time(iter_end)
                    total_times.append(total_time)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Progress: {i+1}/{timing_runs}")
            
            # collect and save results
            block_stats = transformer_model.get_timing_stats()
            
            print("\n" + "="*60)
            print(f"TIMING RESULTS - {precision.upper()}")
            print("="*60)
            
            if block_stats:
                print("\nBlock-level timing (ms):")
                print("-" * 50)
                print(f"{'Block':<25} {'Mean':>10} {'Std':>10} {'%':>8}")
                print("-" * 50)
                
                total_block_mean = sum(s['mean_ms'] for s in block_stats.values())
                
                for block_name in sorted(block_stats.keys()):
                    stats = block_stats[block_name]
                    percentage = (stats['mean_ms'] / total_block_mean * 100) if total_block_mean > 0 else 0
                    print(f"{block_name:<25} {stats['mean_ms']:10.4f} {stats['std_ms']:10.4f} {percentage:7.1f}%")
                
                print("-" * 50)
                print(f"{'Sum of blocks':<25} {total_block_mean:10.4f}")
                
                csv_data = []
                for block_name in sorted(block_stats.keys()):
                    stats = block_stats[block_name]
                    percentage = (stats['mean_ms'] / total_block_mean * 100) if total_block_mean > 0 else 0
                    csv_data.append({
                        'Block': block_name,
                        'Mean_ms': stats['mean_ms'],
                        'Std_ms': stats['std_ms'],
                        'Min_ms': stats['min_ms'],
                        'Max_ms': stats['max_ms'],
                        'Count': stats['count'],
                        'Percentage': percentage
                    })
                
                df_blocks = pd.DataFrame(csv_data)
                df_blocks.to_csv(os.path.join(timing_dir, f'block_timing_{precision}.csv'), index=False)
            
            total_times_array = np.array(total_times)
            total_stats = {
                'mean_ms': float(np.mean(total_times_array)),
                'std_ms': float(np.std(total_times_array)),
                'min_ms': float(np.min(total_times_array)),
                'max_ms': float(np.max(total_times_array)),
                'median_ms': float(np.median(total_times_array)),
                'p95_ms': float(np.percentile(total_times_array, 95)),
                'p99_ms': float(np.percentile(total_times_array, 99)),
            }
            
            print(f"\nTotal forward pass timing (ms):")
            print("-" * 50)
            for key, value in total_stats.items():
                print(f"  {key}: {value:.4f}")
            
            df_total = pd.DataFrame([total_stats])
            df_total.to_csv(os.path.join(timing_dir, f'total_timing_{precision}.csv'), index=False)
            
            df_raw = pd.DataFrame({'iteration': range(len(total_times)), 'total_time_ms': total_times})
            df_raw.to_csv(os.path.join(timing_dir, f'raw_timing_data_{precision}.csv'), index=False)
            
            report_path = os.path.join(timing_dir, f'timing_report_{precision}.txt')
            with open(report_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write(f"DIFFUSION POLICY CUDA TIMING REPORT - {precision.upper()}\n")
                f.write("="*60 + "\n\n")
                f.write(f"Device: {device}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Warmup iterations: {warmup}\n")
                f.write(f"Timing iterations: {timing_runs}\n")
                f.write(f"Batch size: {batch_size}\n")
                f.write(f"Horizon: {horizon}\n")
                f.write(f"Action dim: {action_dim}\n")
                f.write(f"Use condition: {use_cond}\n")
                if use_cond:
                    f.write(f"Obs dim: {obs_dim}\n")
                    f.write(f"Obs steps: {n_obs_steps}\n")
                
                f.write("\n" + "-"*60 + "\n")
                f.write("BLOCK-LEVEL TIMING\n")
                f.write("-"*60 + "\n")
                if block_stats:
                    total_block_mean = sum(s['mean_ms'] for s in block_stats.values())
                    for block_name in sorted(block_stats.keys()):
                        stats = block_stats[block_name]
                        percentage = (stats['mean_ms'] / total_block_mean * 100) if total_block_mean > 0 else 0
                        f.write(f"\n{block_name}:\n")
                        f.write(f"  Mean: {stats['mean_ms']:.4f} ms ({percentage:.1f}%)\n")
                        f.write(f"  Std:  {stats['std_ms']:.4f} ms\n")
                        f.write(f"  Min:  {stats['min_ms']:.4f} ms\n")
                        f.write(f"  Max:  {stats['max_ms']:.4f} ms\n")
                    f.write(f"\nSum of block means: {total_block_mean:.4f} ms\n")
                
                f.write("\n" + "-"*60 + "\n")
                f.write("TOTAL FORWARD PASS TIMING\n")
                f.write("-"*60 + "\n")
                for key, value in total_stats.items():
                    f.write(f"  {key}: {value:.4f} ms\n")
            
            print(f"\nTiming results saved to: {timing_dir}")
            transformer_model.disable_cuda_timing()
            print("\nBlock-level timing disabled.")
    
    # 精度计时完成后不运行评估（避免 int8 模型在环境中出错）
    if precision in ['int8']:
        print(f"\nSkipping environment evaluation for {precision} (model not compatible).")
        print(f"Timing results saved to: {output_dir}")
        return
    
    # run eval (only for fp32/fp16)
    print("\nRunning environment evaluation...")
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()