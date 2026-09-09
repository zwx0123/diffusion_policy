"""
Usage:
python eval_autocast.py \
    --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt \
    -o data/pusht_eval_output_fp16_autocast
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
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')


def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # -------- load checkpoint --------
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # -------- build policy --------
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy = policy.to(device)
    policy.eval()

    # -------- AMP / cuDNN opts --------
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    # -------- env runner --------
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )

    # -------- FP16 autocast inference --------
    with torch.cuda.amp.autocast():
        runner_log = env_runner.run(policy)

    # -------- dump log --------
    json_log = {}
    for k, v in runner_log.items():
        if hasattr(v, '_path'):  # wandb Video
            json_log[k] = v._path
        else:
            json_log[k] = v

    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()