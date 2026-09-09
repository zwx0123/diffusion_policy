"""
Usage:
python eval_seed.py \
    --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt \
    -o data/pusht_eval_output \
    --eval_seed 1000
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import json
import random

import click
import dill
import hydra
import numpy as np
import torch
import wandb

from diffusion_policy.workspace.base_workspace import BaseWorkspace


def set_eval_seed(seed: int):
    """
    Set all random number generators used by evaluation.

    This primarily controls diffusion sampling noise and any other
    evaluation-time RNG. Environment episode seeds are usually separately
    fixed by the env_runner configuration.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option(
    '--eval_seed',
    default=0,
    type=int,
    show_default=True,
    help='Random seed for diffusion sampling and evaluation-time RNG.'
)
def main(checkpoint, output_dir, device, eval_seed):
    if os.path.exists(output_dir):
        click.confirm(
            f"Output path {output_dir} already exists! Overwrite?",
            abort=True
        )

    pathlib.Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------
    payload = torch.load(
        open(checkpoint, 'rb'),
        pickle_module=dill,
        map_location='cpu'
    )

    cfg = payload['cfg']
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
    # Get policy
    # --------------------------------------------------------
    policy = workspace.model

    if cfg.training.use_ema:
        policy = workspace.ema_model
        print("Using EMA model.")
    else:
        print("Using non-EMA model.")

    device = torch.device(device)

    policy.to(device)
    policy.eval()

    # --------------------------------------------------------
    # Build runner
    # --------------------------------------------------------
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )

    # --------------------------------------------------------
    # Set evaluation RNG immediately before rollout
    # --------------------------------------------------------
    set_eval_seed(eval_seed)
    print(f"Evaluation RNG seed: {eval_seed}")

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------
    with torch.inference_mode():
        runner_log = env_runner.run(policy)

    # --------------------------------------------------------
    # Save JSON log
    # --------------------------------------------------------
    json_log = {
        'eval_seed': eval_seed
    }

    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif isinstance(value, torch.Tensor):
            if value.numel() == 1:
                json_log[key] = value.item()
            else:
                json_log[key] = value.detach().cpu().tolist()
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, 'eval_log.json')

    with open(out_path, 'w') as f:
        json.dump(
            json_log,
            f,
            indent=2,
            sort_keys=True,
            default=str
        )

    print(f"Evaluation log saved to: {out_path}")


if __name__ == '__main__':
    main()
