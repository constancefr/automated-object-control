import os
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder, SubprocVecEnv
import matplotlib.pyplot as plt
import cv2

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
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = Monitor(env)
    return env

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
    learning_starts=2_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    verbose=1,
    tensorboard_log=TENSORBOARD_LOG,
)

# Callbacks
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

class VideoRecorderCallback(BaseCallback):
    def __init__(self, env, record_every=VIDEO_EVERY, video_length=VIDEO_LENGTH, folder=VIDEO_DIR, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.record_every = record_every
        self.video_length = video_length
        self.folder = folder

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_every == 0:
            video_path = os.path.join(self.folder, f"rl_video_{self.num_timesteps}")
            # Wrap env with VecVideoRecorder
            video_env = VecVideoRecorder(
                self.env,
                video_path,
                record_video_trigger=lambda step: True,
                video_length=self.video_length,
                name_prefix=f"step{self.num_timesteps}"
            )
            obs = video_env.reset()
            done = False
            steps = 0
            while steps < self.video_length and not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # obs, reward, terminated, truncated, info = video_env.step(action)
                # done = terminated or truncated
                obs, rewards, dones, infos = video_env.step(action) # 4-tuple for compatibility with VecEnv!
                done = dones[0]
                steps += 1
            video_env.close()
            if self.verbose > 0:
                print(f"Recorded video at step {self.num_timesteps}")
        return True

video_callback = VideoRecorderCallback(eval_env)

# Training
print("Continuous training ---")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback, video_callback],
    log_interval=10,
)

model.save(os.path.join(RESULTS_DIR, f"{MODEL_FILENAME_BASE}_final"))
env.save(os.path.join(RESULTS_DIR, "vecnormalize_stats.pkl"))
print("Training complete ---")


# plt.plot(video_callback.timesteps, video_callback.avg_rewards)
# plt.xlabel("Timestep")
# plt.ylabel("Average Reward (10 episodes)")
# plt.title("Training Progress")
# plt.savefig(os.path.join(RESULTS_DIR, "training_progress.png"))
# plt.show()
