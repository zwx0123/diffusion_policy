import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import gc
import logging
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.train_start_seed = train_start_seed
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed
        self.legacy_test = legacy_test
        self.keypoint_visible_rate = keypoint_visible_rate
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.past_action = past_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.output_dir = output_dir

        if n_envs is None:
            n_envs = min(n_train + n_test, 16)
        self.n_envs = n_envs

        self._cached_env_fns = None
        self._cached_env_seeds = None
        self._cached_env_prefixs = None
        self._cached_env_init_fn_dills = None

        self.env = None

    def _build_env_fns(self):
        if self._cached_env_fns is not None:
            return self._cached_env_fns, self._cached_env_seeds, self._cached_env_prefixs, self._cached_env_init_fn_dills

        env_n_obs_steps = self.n_obs_steps + self.n_latency_steps
        env_n_action_steps = self.n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=self.legacy_test,
                        keypoint_visible_rate=self.keypoint_visible_rate,
                        agent_keypoints=self.agent_keypoints,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=self.fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=self.crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=self.max_steps
            )

        env_fns = [env_fn] * self.n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        for i in range(self.n_train):
            seed = self.train_start_seed + i
            enable_render = i < self.n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(self.output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        for i in range(self.n_test):
            seed = self.test_start_seed + i
            enable_render = i < self.n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(self.output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        self._cached_env_fns = env_fns
        self._cached_env_seeds = env_seeds
        self._cached_env_prefixs = env_prefixs
        self._cached_env_init_fn_dills = env_init_fn_dills

        return env_fns, env_seeds, env_prefixs, env_init_fn_dills

    def _create_env(self, env_fns_chunk):
        return AsyncVectorEnv(env_fns_chunk)

    def _close_env(self):
        if self.env is not None:
            try:
                self.env.close(timeout=2, terminate=True)
            except Exception:
                pass
            self.env = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _clear_caches(self):
        self._cached_env_fns = None
        self._cached_env_seeds = None
        self._cached_env_prefixs = None
        self._cached_env_init_fn_dills = None
    
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device

        env_fns, env_seeds, env_prefixs, env_init_fn_dills = self._build_env_fns()

        n_envs = self.n_envs
        n_inits = len(env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        logger.info(f"Starting evaluation: {n_chunks} chunks, {n_envs} envs per chunk, {n_inits} total episodes")

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_env_fns = env_fns[:this_n_active_envs]
            this_init_fns = env_init_fn_dills[this_global_slice]

            if len(this_init_fns) < this_n_active_envs:
                n_diff = this_n_active_envs - len(this_init_fns)
                this_init_fns.extend([env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == this_n_active_envs

            self.env = self._create_env(this_env_fns)

            try:
                self.env.call_each('run_dill_function',
                    args_list=[(x,) for x in this_init_fns])

                obs = self.env.reset()
                past_action = None
                policy.reset()

                pbar = tqdm.tqdm(total=self.max_steps,
                    desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}",
                    leave=False, mininterval=self.tqdm_interval_sec)
                done = False
                while not done:
                    Do = obs.shape[-1] // 2
                    np_obs_dict = {
                        'obs': obs[..., :self.n_obs_steps, :Do].astype(np.float32),
                        'obs_mask': obs[..., :self.n_obs_steps, Do:] > 0.5
                    }
                    if self.past_action and (past_action is not None):
                        np_obs_dict['past_action'] = past_action[
                            :, -(self.n_obs_steps - 1):].astype(np.float32)

                    obs_dict = dict_apply(np_obs_dict,
                        lambda x: torch.from_numpy(x).to(device=device))

                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)

                    np_action_dict = dict_apply(action_dict,
                        lambda x: x.detach().to('cpu').numpy())

                    action = np_action_dict['action'][:, self.n_latency_steps:]

                    obs, reward, done, info = self.env.step(action)
                    done = np.all(done)
                    past_action = action

                    pbar.update(action.shape[1])
                pbar.close()

                all_video_paths[this_global_slice] = self.env.render()[this_local_slice]
                all_rewards[this_global_slice] = self.env.call('get_attr', 'reward')[this_local_slice]

                del obs, reward, done, info, action, np_action_dict, action_dict
                del obs_dict, np_obs_dict, past_action
            finally:
                self._close_env()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        n_available = min(n_inits, len(all_rewards), len(all_video_paths))
        for i in range(n_available):
            seed = env_seeds[i]
            prefix = env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        mean_score = log_data.get('test/mean_score', log_data.get('test_mean_score', 0))
        logger.info(f"Evaluation complete: test/mean_score={mean_score:.4f}")

        for i in range(n_available):
            all_video_paths[i] = None

        for i in range(min(n_inits, len(env_seeds))):
            key_to_remove = None
            for k in list(log_data.keys()):
                if f'sim_video_{env_seeds[i]}' in k:
                    key_to_remove = k
                    break
            if key_to_remove:
                del log_data[key_to_remove]

        self._clear_caches()
        
        del all_video_paths, all_rewards, max_rewards
        del env_fns, env_seeds, env_prefixs, env_init_fn_dills
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return log_data

    def close(self):
        self._close_env()

        self._cached_env_fns = None
        self._cached_env_seeds = None
        self._cached_env_prefixs = None
        self._cached_env_init_fn_dills = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass