import os
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder, SubprocVecEnv
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm
import time
import numpy as np
import pandas as pd

os.environ["SDL_VIDEODRIVER"] = "dummy"  # required when running on remote server without GUI

# Testing:
# TOTAL_TIMESTEPS = 10_000
# EVAL_FREQ = 2_000
# CHECKPOINT_FREQ = 5_000
# VIDEO_EVERY = 5_000
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000
CHECKPOINT_FREQ = 50_000 # saving freq
VIDEO_EVERY = 100_000
VIDEO_LENGTH = 1000 # num steps
VIDEO_FILENAME = "training.mp4"
VIDEO_DIR = "./training_videos"
MODEL_FILENAME_BASE = "carcontrol-dqn"
TENSORBOARD_LOG = "./tensorboard/"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def make_env():
    env = gym.make('versaille_env:versaille_env/acc-discrete-v0')
    env = Monitor(env)
    return env

class TQDMCallback(BaseCallback):
    '''
    Custom callback for updating tqdm
    '''
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.progress_bar = None
        self.total_timesteps = total_timesteps
        
    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training!!")
        
    def _on_step(self):
        self.progress_bar.update(1)
        return True
        
    def _on_training_end(self):
        self.progress_bar.close()

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, record_every=VIDEO_EVERY, video_length=VIDEO_LENGTH, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.record_every = record_every
        self.video_length = video_length
        self.video_dir = os.path.join(RESULTS_DIR, "training_videos")
        os.makedirs(self.video_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_every == 0 and self.num_timesteps > 0:
            video_path = os.path.join(self.video_dir, f"step_{self.num_timesteps}")
            
            # Create a fresh environment for recording to avoid state contamination
            temp_env = DummyVecEnv([make_env])
            temp_env = VecNormalize.load(
                os.path.join(RESULTS_DIR, "vecnormalize_stats.pkl"), 
                temp_env
            ) if os.path.exists(os.path.join(RESULTS_DIR, "vecnormalize_stats.pkl")) else temp_env
            temp_env.training = False
            temp_env.norm_reward = False
            
            video_env = VecVideoRecorder(
                temp_env,
                video_path,
                record_video_trigger=lambda x: x == 0,  # Record only first episode
                video_length=self.video_length,
                name_prefix=f"step_{self.num_timesteps}"
            )
            
            # Record one episode
            obs = video_env.reset()
            for step in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, _ = video_env.step(action)
                if dones[0]:
                    break
            
            video_env.close()
            temp_env.close()
            
            if self.verbose > 0:
                print(f"\nRecorded video at step {self.num_timesteps}")
                
        return True
    
class ProgressStatsCallback(BaseCallback):
    '''
    Track training progress stats.
    '''
    def __init__(self, eval_env, eval_freq=EVAL_FREQ, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.timesteps = []
        self.mean_rewards = []
        self.std_rewards = []
        self.crash_rates = []
        
    def _on_step(self) -> bool:
        # Record evaluation statistics
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            try:
                mean_reward, std_reward = self._evaluate_policy()
                crash_rate = self._estimate_crash_rate()
                
                self.timesteps.append(self.num_timesteps)
                self.mean_rewards.append(mean_reward)
                self.std_rewards.append(std_reward)
                self.crash_rates.append(crash_rate)
                
                if self.verbose > 0:
                    print(f"\nStep {self.num_timesteps}: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}, Crash Rate = {crash_rate:.2f}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"\nEvaluation failed at step {self.num_timesteps}: {e}")
        return True
    
    def _evaluate_policy(self):
        '''Evaluation of current policy'''
        # eval_env = self.model.eval_env
        n_eval_episodes = 3  # super quick
        episode_rewards = []
        
        for _ in range(n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0]
                if done[0]:
                    break
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    def _estimate_crash_rate(self):
        # eval_env = self.model.eval_env
        n_eval_episodes = 3
        crashes = 0
        
        for _ in range(n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                if info[0].get('crash', False):
                    crashes += 1
                    break
                if done[0]:
                    break
        
        return crashes / n_eval_episodes
    
    def plot_progress(self, save_path=None):
        if not self.timesteps:
            print("No progress data to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot rewards
        ax1.plot(self.timesteps, self.mean_rewards, 'b-', label='Mean Reward')
        ax1.fill_between(self.timesteps, 
                        np.array(self.mean_rewards) - np.array(self.std_rewards),
                        np.array(self.mean_rewards) + np.array(self.std_rewards),
                        alpha=0.2, label='±1 std')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress - Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot crash rates
        ax2.plot(self.timesteps, self.crash_rates, 'r-', label='Crash Rate')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Crash Rate')
        ax2.set_title('Training Progress - Crash Rates')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Progress plot saved to {save_path}")
        
        return fig
    
    def save_stats(self, save_path=None):
        '''Save to CSV file.'''
        if not self.timesteps:
            print("No statistics to save")
            return
            
        stats_df = pd.DataFrame({
            'timesteps': self.timesteps,
            'mean_reward': self.mean_rewards,
            'std_reward': self.std_rewards,
            'crash_rate': self.crash_rates
        })
        
        if save_path:
            stats_df.to_csv(save_path, index=False)
            print(f"Training statistics saved to {save_path}")
        
        return stats_df
    
def print_training_summary(progress_callback):
    if not progress_callback.timesteps:
        print("No training statistics available")
        return
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    # Initial performance
    initial_reward = progress_callback.mean_rewards[0] if progress_callback.mean_rewards else 0
    initial_crash = progress_callback.crash_rates[0] if progress_callback.crash_rates else 0
    
    # Final performance
    final_reward = progress_callback.mean_rewards[-1] if progress_callback.mean_rewards else 0
    final_crash = progress_callback.crash_rates[-1] if progress_callback.crash_rates else 0
    
    # Best performance
    if progress_callback.mean_rewards:
        best_reward_idx = np.argmax(progress_callback.mean_rewards)
        best_reward = progress_callback.mean_rewards[best_reward_idx]
        best_crash = progress_callback.crash_rates[best_reward_idx]
        best_step = progress_callback.timesteps[best_reward_idx]
    
    print(f"Initial Performance:")
    print(f"  Mean Reward: {initial_reward:.2f}")
    print(f"  Crash Rate:  {initial_crash:.2f}")
    
    print(f"\nFinal Performance:")
    print(f"  Mean Reward: {final_reward:.2f}")
    print(f"  Crash Rate:  {final_crash:.2f}")
    
    if progress_callback.mean_rewards:
        print(f"\nBest Performance (at step {best_step:,}):")
        print(f"  Mean Reward: {best_reward:.2f}")
        print(f"  Crash Rate:  {best_crash:.2f}")
        
        # Find checkpoints with reasonable performance (not perfect)
        print(f"\nCheckpoints with Crash Rate > 0.1 (good for training data):")
        for i, (step, crash_rate) in enumerate(zip(progress_callback.timesteps, progress_callback.crash_rates)):
            if crash_rate > 0.1:  # Models that still crash sometimes
                reward = progress_callback.mean_rewards[i]
                print(f"  Step {step:,}: Reward = {reward:.2f}, Crash Rate = {crash_rate:.2f}")

def main():
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10) # normalise observation space!

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10) # rewards not normalised here
    # eval_env.set_running_stats(env)

    model = sb3.DQN(
        'MlpPolicy',
        env,
        policy_kwargs={'net_arch': [256, 256]},
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        verbose=0,
        tensorboard_log=TENSORBOARD_LOG,
        exploration_final_eps=0.1,  # diverse data
        exploration_fraction=0.3,   # longer exploration period
    )

    # Callbacks
    progress_callback = TQDMCallback(total_timesteps=TOTAL_TIMESTEPS)
    stats_callback = ProgressStatsCallback(eval_env=eval_env, eval_freq=EVAL_FREQ, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(RESULTS_DIR, "best_model"),
        log_path=os.path.join(RESULTS_DIR, "logs"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(RESULTS_DIR, "checkpoints"),
        name_prefix="dqn_checkpoint",
    )

    video_callback = VideoRecorderCallback(eval_env)

    # Training
    print("Continuous training ---")
    start_time = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[progress_callback, eval_callback, checkpoint_callback, video_callback, stats_callback],
        log_interval=100,
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    model.save(os.path.join(RESULTS_DIR, f"{MODEL_FILENAME_BASE}_final"))
    env.save(os.path.join(RESULTS_DIR, "vecnormalize_stats.pkl"))
    
    # Generate and save progress plots
    stats_callback.plot_progress(os.path.join(RESULTS_DIR, "training_progress.png"))
    stats_callback.save_stats(os.path.join(RESULTS_DIR, "training_stats.csv"))
    
    # Print training summary
    print_training_summary(stats_callback)

    print("Training complete ---")

if __name__ == "__main__":
    main()
