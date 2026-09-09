"""
E0 Baseline: FP32 inference latency (strict)

Usage:
python eval_latency_fp32.py \
  -c data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt \
  -o data/eval_fp32_latency
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
    WARMUP = 5
    MEASURE = 30

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

        if state["batch_size"] is None:
            for v in obs_dict.values():
                if isinstance(v, torch.Tensor):
                    state["batch_size"] = int(v.shape[0])
                    break

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            if state["calls"] <= WARMUP:
                return original_predict_action(obs_dict, *args, **kwargs)

            if len(latencies_us) < MEASURE:
                start_evt.record()
                result = original_predict_action(obs_dict, *args, **kwargs)
                end_evt.record()
                torch.cuda.synchronize(device)
                latencies_us.append(start_evt.elapsed_time(end_evt) * 1000.0)
                return result

            return original_predict_action(obs_dict, *args, **kwargs)

    latency_output_dir = os.path.join(output_dir, "latency_rollout")
    pathlib.Path(os.path.join(latency_output_dir, "media")).mkdir(parents=True, exist_ok=True)

    # -------- Latency rollout --------
    latency_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=latency_output_dir
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
        )

    latencies_us = np.array(latencies_us, dtype=np.float64)

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

    # === CHANGED ===
    # Success-rate evaluation is intentionally NOT performed here.
    # It must be run separately using the canonical eval.py to avoid
    # environment and RNG state pollution.
    print("[E0 LATENCY] Success-rate evaluation skipped (see eval.py)")
    # === END CHANGED ===

if __name__ == "__main__":
    main()