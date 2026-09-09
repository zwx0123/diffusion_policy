"""
E0 Baseline: FP32 inference latency (strict)

Usage:
python eval_latency_fp32.py \
  -c data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt \
  -o data/eval_fp32_latency
"""

##更新原因：输出结果平均成功率来到0.88，猜测为环境初始状态漂移问题

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import json
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # -------- Load ----------
    payload = torch.load(checkpoint, pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # -------- Device & Strict FP32 --------
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA Event latency benchmark requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.cuda.set_device(device)

    # === STRICT FP32 (disable TF32) ===
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    policy.to(device)
    policy.float()
    policy.eval()

    # -------- Timing config --------
    WARMUP = 50
    MEASURE = 200

    latencies_us = []
    state = {
        "calls": 0,
        "batch_size": None,
    }

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    original_predict_action = policy.predict_action

    def timed_predict_action(obs_dict, *args, **kwargs):
        state["calls"] += 1

        # Capture batch size (first tensor seen)
        if state["batch_size"] is None:
            for v in obs_dict.values():
                if isinstance(v, torch.Tensor):
                    state["batch_size"] = int(v.shape[0])
                    break

        # === Inference + disable autocast ===
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            # Warmup
            if state["calls"] <= WARMUP:
                return original_predict_action(obs_dict, *args, **kwargs)

            # Measurement
            if len(latencies_us) < MEASURE:
                start_evt.record()
                result = original_predict_action(obs_dict, *args, **kwargs)
                end_evt.record()
                torch.cuda.synchronize(device)
                latencies_us.append(start_evt.elapsed_time(end_evt) * 1000.0)
                return result

            return original_predict_action(obs_dict, *args, **kwargs)

    # -------- Latency rollout --------
    latency_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=os.path.join(output_dir, "latency_rollout")
    )

    policy.predict_action = timed_predict_action
    try:
        latency_runner.run(policy)
    finally:
        policy.predict_action = original_predict_action

    # -------- Validation (fail fast) --------
    if len(latencies_us) < MEASURE:
        raise RuntimeError(
            f"[E0 BASELINE] Latency sampling failed:\n"
            f"  collected samples : {len(latencies_us)}/{MEASURE}\n"
            f"  predict_action calls : {state['calls']}\n"
            f"  required >= {WARMUP + MEASURE}\n"
            f"  Increase episode length or reduce WARMUP/MEASURE."
        )

    latencies_us = np.array(latencies_us, dtype=np.float64)

    # Model dtype (robust)
    first_param = next(iter(policy.parameters()), None)
    if first_param is None:
        first_param = next(iter(policy.buffers()), None)
    model_dtype = str(first_param.dtype) if first_param is not None else "unknown"

    stats = {
        "metric": "predict_action_cuda_event_latency",
        "unit": "us",
        "mean_us": float(latencies_us.mean()),
        "std_us": float(latencies_us.std()),
        "p50_us": float(np.percentile(latencies_us, 50)),
        "p95_us": float(np.percentile(latencies_us, 95)),
        "p99_us": float(np.percentile(latencies_us, 99)),
        "num_samples": int(latencies_us.size),
        "warmup_calls": WARMUP,
        "batch_size": state["batch_size"],
        "dtype": model_dtype,
        "device": str(device),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "note": "Synchronized GPU kernel time inside predict_action; excludes CPU/env overhead.",
    }

    with open(os.path.join(output_dir, "latency.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("[E0 LATENCY]", json.dumps(stats, indent=2))

    # -------- Clean eval for success rate --------
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )
    runner_log = env_runner.run(policy)

    json_log = {}
    for k, v in runner_log.items():
        if hasattr(v, "_path"):
            json_log[k] = v._path
        else:
            json_log[k] = v

    with open(os.path.join(output_dir, "eval_log.json"), "w") as f:
        json.dump(json_log, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()