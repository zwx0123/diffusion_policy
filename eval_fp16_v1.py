"""
Strict FP16 evaluation for diffusion_policy.

This script intentionally does NOT use torch.autocast.

Floating-point model parameters, floating-point buffers, observations,
diffusion trajectories, and timestep embeddings are converted to FP16.

Usage:
python eval_fp16_v1.py \
    -c data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt \
    -o data/eval_fp16/eval_fp16_v1 \
    -d cuda:0
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import math
import json
import pathlib

import click
import dill
import hydra
import torch

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


# ============================================================
# Strict-FP16 patch for SinusoidalPosEmb
# ============================================================
#
# Original diffusion_policy implementation is usually:
#
#   emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#   emb = x[:, None].float() * emb[None, :]
#
# The ".float()" makes the positional embedding FP32 even after policy.half().
# Then the next Linear layer has FP16 weights, producing:
#
#   mat1 Float, mat2 Half
#
# This replacement deliberately uses float16 for all floating-point operations.
# Integer timestep indices are allowed as input and are converted to FP16 here.
# ============================================================

def strict_fp16_sinusoidal_pos_emb_forward(self, x):
    """
    Strict FP16 SinusoidalPosEmb forward.

    Input:
        x: diffusion timestep tensor, usually int64.

    Output:
        FP16 sinusoidal positional embedding.
    """
    device = x.device
    dtype = torch.float16

    # timestep might be torch.int64; convert it to FP16 explicitly.
    x = x.to(device=device, dtype=dtype)

    half_dim = self.dim // 2

    # Keep the scalar construction outside of torch tensor math.
    # The actual tensor operations below are FP16.
    scale = math.log(10000.0) / (half_dim - 1)

    # Explicit dtype=torch.float16 is important.
    freq = torch.arange(
        half_dim,
        device=device,
        dtype=dtype
    )

    freq = torch.exp(freq * (-scale))

    emb = x[:, None] * freq[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

    return emb


# Replace the implementation globally before loading/running the policy.
SinusoidalPosEmb.forward = strict_fp16_sinusoidal_pos_emb_forward


# ============================================================
# Strict-FP16 policy wrapper
# ============================================================

class FP16PolicyWrapper(torch.nn.Module):
    """
    Wrapper that converts floating-point observations to FP16.

    Important:
    env_runner calls policy.predict_action(obs_dict), NOT policy(obs_dict).
    Therefore predict_action must be implemented explicitly.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    @property
    def device(self):
        return next(self.policy.parameters()).device

    @property
    def dtype(self):
        return next(self.policy.parameters()).dtype

    def _to_fp16(self, value):
        """
        Recursively move tensors to policy.device.

        - float32 / float64 / bfloat16 tensors -> float16
        - integer/bool tensors remain their original dtype
          (e.g. indices, masks, shape-related tensors).
        """
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.to(
                    device=self.device,
                    dtype=torch.float16,
                    non_blocking=True
                )
            else:
                return value.to(
                    device=self.device,
                    non_blocking=True
                )

        if isinstance(value, dict):
            return {
                k: self._to_fp16(v)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [self._to_fp16(v) for v in value]

        if isinstance(value, tuple):
            return tuple(self._to_fp16(v) for v in value)

        return value

    def forward(self, obs, **kwargs):
        """
        Kept for compatibility if a runner directly invokes policy(obs).
        """
        obs_fp16 = self._to_fp16(obs)
        return self.policy(obs_fp16, **kwargs)

    def predict_action(self, obs_dict, **kwargs):
        """
        This is the method actually called by PushT runners.
        """
        obs_fp16 = self._to_fp16(obs_dict)
        return self.policy.predict_action(obs_fp16, **kwargs)

    def reset(self):
        """
        Some runners call policy.reset() before each episode.
        """
        return self.policy.reset()

    def __getattr__(self, name):
        """
        Forward methods/attributes not defined by this wrapper to self.policy.
        """
        wrapper_attributes = {
            "policy",
            "device",
            "dtype",
            "_to_fp16",
            "forward",
            "predict_action",
            "reset",
        }

        if name in wrapper_attributes:
            return super().__getattr__(name)

        return getattr(self.policy, name)


@click.command()
@click.option(
    "-c",
    "--checkpoint",
    required=True,
    type=click.Path(exists=True)
)
@click.option(
    "-o",
    "--output_dir",
    required=True,
    type=click.Path()
)
@click.option(
    "-d",
    "--device",
    default="cuda:0",
    show_default=True
)
def main(checkpoint, output_dir, device):
    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------
    if os.path.exists(output_dir):
        click.confirm(
            f"Output path {output_dir} already exists! Overwrite?",
            abort=True
        )

    pathlib.Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device} was requested, "
            "but torch.cuda.is_available() is False."
        )

    # --------------------------------------------------------
    # Load checkpoint and workspace
    # --------------------------------------------------------
    with open(checkpoint, "rb") as f:
        payload = torch.load(
            f,
            pickle_module=dill,
            map_location="cpu"
        )

    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)

    workspace = cls(
        cfg,
        output_dir=output_dir
    )
    workspace: BaseWorkspace

    workspace.load_payload(
        payload,
        exclude_keys=None,
        include_keys=None
    )

    # --------------------------------------------------------
    # Select EMA policy if enabled
    # --------------------------------------------------------
    if cfg.training.use_ema:
        policy = workspace.ema_model
        print("Using EMA model.")
    else:
        policy = workspace.model
        print("Using non-EMA model.")

    # --------------------------------------------------------
    # Strict FP16 conversion
    # --------------------------------------------------------
    #
    # No autocast.
    # No FP32 LayerNorm / GroupNorm / BatchNorm exception.
    # policy.half() recursively converts floating parameters and buffers.
    #
    policy = policy.to(device)
    policy = policy.half()
    policy.eval()

    # This is only a verification print; policy.half() already converts
    # floating-point parameters and floating-point buffers recursively.
    first_param = next(policy.parameters())
    print(f"Policy device: {first_param.device}")
    print(f"Policy dtype : {first_param.dtype}")

    # Verify that there are no remaining FP32 floating params/buffers.
    fp32_params = []
    fp32_buffers = []

    for name, param in policy.named_parameters():
        if param.is_floating_point() and param.dtype != torch.float16:
            fp32_params.append((name, str(param.dtype)))

    for name, buffer in policy.named_buffers():
        if buffer.is_floating_point() and buffer.dtype != torch.float16:
            fp32_buffers.append((name, str(buffer.dtype)))

    if len(fp32_params) > 0:
        print("WARNING: non-FP16 floating parameters found:")
        for item in fp32_params:
            print("  ", item)

    if len(fp32_buffers) > 0:
        print("WARNING: non-FP16 floating buffers found:")
        for item in fp32_buffers:
            print("  ", item)

    if len(fp32_params) == 0 and len(fp32_buffers) == 0:
        print("Verified: all floating policy parameters and buffers are FP16.")

    # --------------------------------------------------------
    # Wrap policy: observations passed to predict_action become FP16
    # --------------------------------------------------------
    policy = FP16PolicyWrapper(policy)

    # --------------------------------------------------------
    # Build environment runner
    # --------------------------------------------------------
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )

    # --------------------------------------------------------
    # Strict FP16 inference
    # --------------------------------------------------------
    #
    # No torch.autocast here by design.
    #
    with torch.inference_mode():
        runner_log = env_runner.run(policy)

    # --------------------------------------------------------
    # Save JSON log
    # --------------------------------------------------------
    json_log = {}

    for k, v in runner_log.items():
        if hasattr(v, "_path"):
            json_log[k] = v._path
        elif isinstance(v, torch.Tensor):
            if v.numel() == 1:
                json_log[k] = v.item()
            else:
                json_log[k] = v.detach().cpu().tolist()
        else:
            json_log[k] = v

    out_path = os.path.join(output_dir, "eval_log.json")

    with open(out_path, "w") as f:
        json.dump(
            json_log,
            f,
            indent=2,
            sort_keys=True,
            default=str
        )

    print(f"Evaluation log saved to: {out_path}")


if __name__ == "__main__":
    main()
