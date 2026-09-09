# diffusion_policy/workspace/train_diffusion_transformer_lightdp_workspace.py
import os
os.environ['WANDB_MODE'] = 'disabled'

import gc
import json
import pathlib
import logging
import copy
import random
from typing import Dict, Optional

import torch
import hydra
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_lightdp_policy import DiffusionTransformerLightDPPolicy
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def _cleanup_dataloader_workers(dl):
    if dl is None:
        return
    try:
        dl._iterator = None
    except Exception:
        pass
    try:
        if hasattr(dl, '_worker_pids'):
            dl._worker_pids = None
    except Exception:
        pass
    try:
        if hasattr(dl, '_index_queue'):
            dl._index_queue = None
    except Exception:
        pass


class TrainDiffusionTransformerLightDPWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg, output_dir=output_dir)
        
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.policy: DiffusionTransformerLightDPPolicy
        self.policy = hydra.utils.instantiate(cfg.policy)
        
        self.ema_model: DiffusionTransformerLightDPPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.policy)
        
        self.optimizer = self.policy.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.epoch = 0
        
        self.lr_scheduler = None
        self.ema = None
        self.env_runner = None
        self.topk_manager = None
        self.wandb_run = None
        self.train_sampling_batch = None

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        output_dir = pathlib.Path(self.output_dir)
        
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                logger.info(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
        
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = torch.utils.data.DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = torch.utils.data.DataLoader(val_dataset, **cfg.val_dataloader)
        
        self.policy.set_normalizer(normalizer)
        if cfg.training.use_ema and self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        
        device = torch.device(cfg.training.device)
        
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )
        
        if cfg.training.use_ema and self.ema_model is not None:
            self.ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            self.ema.averaged_model.to(device)
        
        self.env_runner: BaseLowdimRunner
        self.env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(self.env_runner, BaseLowdimRunner)
        
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config_dict['output_dir'] = str(output_dir)
        
        self.wandb_run = wandb.init(
            dir=str(output_dir),
            config=config_dict,
            **cfg.logging
        )
        
        ckpt_dir = output_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.topk_manager = TopKCheckpointManager(
            save_dir=str(ckpt_dir),
            **cfg.checkpoint.topk
        )
        
        self.policy.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        
        self.train_sampling_batch = None

        try:
            self._run_training_loop(cfg, device, train_dataloader, val_dataloader, output_dir)
        finally:
            try:
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            
            try:
                _cleanup_dataloader_workers(train_dataloader)
                _cleanup_dataloader_workers(val_dataloader)
            except Exception:
                pass
            
            del train_dataloader, val_dataloader
            gc.collect()
            
            if self.env_runner is not None:
                try:
                    self.env_runner.close()
                except Exception as e:
                    logger.warning(f"Failed to close env_runner: {e}")
                self.env_runner = None
            
            self._cleanup_resources()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("All resources cleaned up.")

    def _run_training_loop(self, cfg, device, train_dataloader, val_dataloader, output_dir):
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = output_dir / 'logs.json.txt'
        normalizer = self.policy.normalizer

        with JsonLogger(str(log_path)) as json_logger:
            for _epoch_idx in range(cfg.training.num_epochs):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                step_log = dict()

                warmup_epochs = cfg.lightdp.get('warmup_epochs', 5)
                pruning_epochs = cfg.lightdp.get('pruning_epochs', 10)
                hard_prune_epoch = warmup_epochs + pruning_epochs

                is_warmup = self.epoch < warmup_epochs
                is_pruning = warmup_epochs <= self.epoch < hard_prune_epoch
                is_finetune = self.epoch >= hard_prune_epoch

                if self.epoch == warmup_epochs and not self.policy.model._warmup_done:
                    logger.info(f"=== Warmup complete at epoch {self.epoch}, computing layer importance ===")
                    svd_rank = cfg.lightdp.get('svd_rank', 4)
                    self.policy.compute_layer_importance(svd_rank=svd_rank)
                    self.optimizer = self.policy.get_optimizer(**cfg.optimizer)
                    if self.lr_scheduler is not None:
                        total_steps = (
                            len(train_dataloader) * cfg.training.num_epochs) \
                                // cfg.training.gradient_accumulate_every
                        remaining_steps = total_steps - self.global_step
                        self.lr_scheduler = get_scheduler(
                            cfg.training.lr_scheduler,
                            optimizer=self.optimizer,
                            num_warmup_steps=cfg.training.lr_warmup_steps,
                            num_training_steps=max(remaining_steps, 1),
                            last_epoch=-1
                        )

                if self.epoch == hard_prune_epoch and not self.policy.model._hard_prune_done:
                    logger.info(f"=== Performing hard pruning at epoch {self.epoch} ===")
                    retained_layers = self.policy.hard_prune()
                    logger.info(f"Retained {retained_layers} layers: {self.policy.model.pruned_indices}")

                    self.optimizer = self.policy.get_optimizer(**cfg.optimizer)
                    if self.lr_scheduler is not None:
                        total_steps = (
                            len(train_dataloader) * cfg.training.num_epochs) \
                                // cfg.training.gradient_accumulate_every
                        remaining_steps = total_steps - self.global_step
                        self.lr_scheduler = get_scheduler(
                            cfg.training.lr_scheduler,
                            optimizer=self.optimizer,
                            num_warmup_steps=cfg.training.lr_warmup_steps,
                            num_training_steps=max(remaining_steps, 1),
                            last_epoch=-1
                        )

                    self._save_checkpoint(tag='pruned')
                    if self.ema_model is not None:
                        logger.info("Rebuilding EMA model for pruned architecture...")
                        old_ema_model = self.ema_model
                        ema_state_dict = old_ema_model.state_dict()

                        pruned_indices = self.policy.model.pruned_indices
                        new_ema_sd = {}
                        remap_count = 0
                        skip_count = 0
                        for k, v in ema_state_dict.items():
                            if k.startswith('model.decoder.layers.'):
                                parts = k.split('.')
                                if len(parts) >= 4 and parts[3].isdigit():
                                    orig_idx = int(parts[3])
                                    if orig_idx in pruned_indices:
                                        new_idx = pruned_indices.index(orig_idx)
                                        new_key = f"model.decoder.layers.{new_idx}." + '.'.join(parts[4:])
                                        new_ema_sd[new_key] = v
                                        remap_count += 1
                                    else:
                                        skip_count += 1
                                else:
                                    new_ema_sd[k] = v
                            else:
                                new_ema_sd[k] = v
                        logger.info(f"EMA weight remap: {remap_count} layers remapped, {skip_count} removed layers skipped")

                        self.ema_model = copy.deepcopy(self.policy)
                        self.ema_model.load_state_dict(new_ema_sd, strict=False)
                        self.ema_model.set_normalizer(normalizer)
                        self.ema_model.to(device)

                        del old_ema_model, ema_state_dict, new_ema_sd
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        self.ema = hydra.utils.instantiate(
                            cfg.ema, model=self.ema_model)
                        self.ema.averaged_model.to(device)
                        self.ema.optimization_step = self.global_step

                    logger.info("=== Hard pruning complete, entering finetuning phase ===")

                train_losses = list()
                with tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if self.train_sampling_batch is None:
                            self.train_sampling_batch = {
                                k: v.clone().detach().cpu()
                                for k, v in batch.items()
                                if isinstance(v, torch.Tensor)
                            }

                        raw_loss = self.policy.compute_loss(batch, epoch=self.epoch)
                        loss = raw_loss['loss'] / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if cfg.training.grad_clip_norm > 0:
                                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.training.grad_clip_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            if self.lr_scheduler is not None:
                                self.lr_scheduler.step()

                        if cfg.training.use_ema and self.ema is not None:
                            self.ema.step(self.policy)

                        raw_loss_cpu = raw_loss['loss'].item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 0
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                        if not is_last_batch:
                            self.wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)

                        self.global_step += 1

                        del batch, raw_loss, loss
                        del raw_loss_cpu

                if self.epoch % cfg.training.val_every == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm(val_dataloader, desc=f"Val epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch in tepoch:
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss_dict = self.policy.compute_loss(batch, epoch=self.epoch)
                                val_losses.append(loss_dict['loss'].item())
                                del batch, loss_dict

                        val_loss = np.mean(val_losses) if val_losses else 0
                        step_log['val_loss'] = val_loss
                        del val_losses, val_loss
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        if hasattr(self.policy.model, 'get_gate_scores'):
                            gate_scores = self.policy.model.get_gate_scores().detach().cpu().numpy()
                            step_log['gate_mean'] = float(np.mean(gate_scores))
                            retained = int((gate_scores > 0.5).sum())
                            step_log['retained_layers'] = retained
                            if is_pruning:
                                logger.info(f"Epoch {self.epoch} gates: [{', '.join([f'{g:.3f}' for g in gate_scores])}] retained={retained}")

                if self.epoch % cfg.training.rollout_every == 0:
                    logger.info(f"Running env rollout at epoch {self.epoch}")

                    policy_to_eval = self.policy
                    if cfg.training.use_ema and self.ema_model is not None:
                        policy_to_eval = self.ema_model

                    rollout_logs = None
                    try:
                        rollout_logs = self.env_runner.run(policy_to_eval)

                        for key, value in rollout_logs.items():
                            step_log[key] = value

                        if 'test_mean_score' in rollout_logs:
                            step_log['test_mean_score'] = rollout_logs['test_mean_score']

                        logger.info(f"Epoch {self.epoch} rollout: mean_score={rollout_logs.get('test_mean_score', 'N/A')}")
                    except Exception as e:
                        logger.warning(f"Env rollout failed: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        del policy_to_eval, rollout_logs
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if self.epoch % cfg.training.checkpoint_every == 0:
                    self._save_checkpoint(tag='latest')

                    if 'test_mean_score' in step_log:
                        ckpt_path = self.get_checkpoint_path()
                        self.topk_manager.save(
                            global_step=self.global_step,
                            epoch=self.epoch,
                            monotonic_measure={'test_mean_score': step_log['test_mean_score']},
                            save_fn=lambda path: self._save_checkpoint(path=path)
                        )

                if self.epoch % cfg.training.sample_every == 0 and self.train_sampling_batch is not None:
                    logger.info(f"Saving sample at epoch {self.epoch}")
                    sample_dir = output_dir / 'samples'
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        self._save_sample(self.train_sampling_batch, str(sample_dir), self.epoch, device)
                    except Exception as e:
                        logger.warning(f"Failed to save sample: {e}")

                self.wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)

                phase = "WARMUP" if is_warmup else ("PRUNING" if is_pruning else "FINETUNE")
                logger.info(f"Epoch {self.epoch} [{phase}] train_loss={np.mean(train_losses):.4f}")

                del train_losses, step_log, phase
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.epoch += 1

        self._save_checkpoint(tag='final')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        policy_to_eval = self.policy
        if cfg.training.use_ema and self.ema_model is not None:
            policy_to_eval = self.ema_model

        if self.env_runner is not None:
            try:
                final_logs = self.env_runner.run(policy_to_eval)
                logger.info(f"Final rollout: {final_logs}")
                del final_logs
            except Exception as e:
                logger.warning(f"Final rollout failed: {e}")

            try:
                self.env_runner.close()
            except Exception as e:
                logger.warning(f"Failed to close env_runner: {e}")
            self.env_runner = None

        del policy_to_eval
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Training completed!")
    
    def _cleanup_resources(self):
        try:
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if self.wandb_run is not None:
                try:
                    wandb.finish()
                except Exception:
                    pass
                self.wandb_run = None

            self.policy = None
            self.ema_model = None
            self.optimizer = None
            self.lr_scheduler = None
            self.ema = None
            self.train_sampling_batch = None
            self.env_runner = None
            self.topk_manager = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("All resources cleaned up successfully.")

        except Exception as e:
            logger.warning(f"Resource cleanup warning: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _save_checkpoint(self, tag='latest', path=None):
        if path is None:
            if tag == 'latest':
                path = self.get_checkpoint_path()
            else:
                path = os.path.join(str(self.output_dir), 'checkpoints', f'{tag}.ckpt')
        path = str(path)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        payload = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'cfg': self.cfg,
            'state_dicts': {},
            'pickles': {},
            'metadata': {
                '_warmup_done': self.policy.model._warmup_done,
                '_hard_prune_done': self.policy.model._hard_prune_done,
                'pruned_indices': getattr(self.policy.model, 'pruned_indices', []),
                'pruned_layer_indices': getattr(self.policy.model, 'pruned_layer_indices', None),
                'original_n_layer': getattr(self.policy.model, 'original_n_layer', 8),
                'current_n_layer': getattr(self.policy.model, 'current_n_layer', 8),
            }
        }
        
        payload['state_dicts']['policy'] = self.policy.state_dict()
        
        if self.ema_model is not None:
            payload['state_dicts']['ema_model'] = self.ema_model.state_dict()
        
        if self.optimizer is not None:
            payload['state_dicts']['optimizer'] = self.optimizer.state_dict()
        
        if self.lr_scheduler is not None:
            payload['state_dicts']['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        payload['pickles']['normalizer'] = self.policy.normalizer.state_dict()
        
        torch.save(payload, path)
        logger.info(f"Saved checkpoint to {path}")

    def _save_sample(self, batch, save_dir, epoch, device):
        policy_to_eval = self.policy
        if self.ema_model is not None:
            policy_to_eval = self.ema_model
        
        policy_to_eval.eval()
        with torch.no_grad():
            result = policy_to_eval.predict_action(batch)
            action_pred = result['action'][0].cpu().numpy()
            
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(action_pred[:, 0], label='action dim 0')
            axes[0].set_title(f'Epoch {epoch} - Action Prediction')
            axes[0].legend()
            axes[1].plot(action_pred[:, 1], label='action dim 1')
            axes[1].set_title(f'Epoch {epoch} - Action Prediction')
            axes[1].legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'epoch_{epoch:04d}.png'))
            plt.close(fig)
            del fig, axes, result, action_pred
        
        policy_to_eval.train()

    def load_checkpoint(self, path=None, tag=None):
        if path is None:
            if tag is None:
                path = self.get_checkpoint_path()
            else:
                path = os.path.join(str(self.output_dir), 'checkpoints', f'{tag}.ckpt')
        
        payload = torch.load(path, map_location='cpu')
        
        if 'cfg' in payload:
            self.cfg = payload['cfg']
        
        if 'epoch' in payload:
            self.epoch = payload['epoch']
        if 'global_step' in payload:
            self.global_step = payload['global_step']
        
        if 'metadata' in payload:
            meta = payload['metadata']
            self.policy.model._warmup_done = meta.get('_warmup_done', False)
            self.policy.model._hard_prune_done = meta.get('_hard_prune_done', False)
            if 'pruned_indices' in meta:
                self.policy.model.pruned_indices = meta['pruned_indices']
            if 'pruned_layer_indices' in meta and meta['pruned_layer_indices'] is not None:
                self.policy.model.pruned_layer_indices = meta['pruned_layer_indices']
            if 'original_n_layer' in meta:
                self.policy.model.original_n_layer = meta['original_n_layer']
            if 'current_n_layer' in meta:
                self.policy.model.current_n_layer = meta['current_n_layer']
                self.policy.model.n_layer = meta['current_n_layer']
        
        if 'state_dicts' in payload:
            sd = payload['state_dicts']
            if 'policy' in sd:
                self.policy.load_state_dict(sd['policy'], strict=False)
            if 'ema_model' in sd and self.ema_model is not None:
                self.ema_model.load_state_dict(sd['ema_model'], strict=False)
            if 'optimizer' in sd and self.optimizer is not None:
                self.optimizer.load_state_dict(sd['optimizer'])
            if 'lr_scheduler' in sd and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(sd['lr_scheduler'])
        
        if 'pickles' in payload and 'normalizer' in payload['pickles']:
            self.policy.normalizer.load_state_dict(payload['pickles']['normalizer'])
        
        logger.info(f"Loaded checkpoint from {path} (epoch={self.epoch}, step={self.global_step})")
        
        del payload, sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_diffusion_transformer_lightdp_pusht")
def main(cfg):
    workspace = TrainDiffusionTransformerLightDPWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()