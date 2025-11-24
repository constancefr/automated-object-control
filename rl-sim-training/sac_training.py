import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, TimeLimit
import torch


# ==============================================
# Logging callback to store reward + episode length
# ==============================================
class EpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards = []
        self.ep_lengths = []

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        if "episode" in info:
            self.ep_rewards.append(info["episode"]["r"])
            self.ep_lengths.append(info["episode"]["l"])
        return True


# ==============================================
# Training & Plotting
# ==============================================
def plot_training(avg_rewards, avg_ep_lengths, save_dir):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(avg_rewards, label="Avg reward")
    plt.xlabel("Round")
    plt.ylabel("Average Reward")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(avg_ep_lengths, label="Avg length")
    plt.xlabel("Round")
    plt.ylabel("Average Episode Length")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_plot.png"))
    plt.close()


# ==============================================
# Training Script
# ==============================================
if __name__ == "__main__":

    # Create directories
    TRIAL_DIR = "./sac_trial"
    os.makedirs(TRIAL_DIR, exist_ok=True)
    os.makedirs(os.path.join(TRIAL_DIR, "sac_models"), exist_ok=True)

    # Create wrapped environment (minimal structural changes)
    env = gym.make("custom_car_env:custom_car_env/acc-continuous-v0", render_mode="rgb_array")


    # --- added normalization wrappers ---
    env = TimeLimit(env, max_episode_steps=2000)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    # -------------------------------------

    env = Monitor(env)

    # Improve SAC hyperparameters (minimal patch)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,                # lower LR => more stable critics
        buffer_size=300_000,               # larger replay for stiff dynamics
        batch_size=128,                    # smaller batch => smoother updates
        train_freq=64,                     # frequent small updates
        gradient_steps=64,                 # match train_freq
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        target_entropy="auto",
        learning_starts=5000,              # let buffer fill before learning
        use_sde=False,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
    )

    # Logging callback
    callback = EpisodeStatsCallback()

    # Training rounds
    NUM_ROUNDS = 20
    STEPS_PER_ROUND = 50_000

    round_avg_rewards = []
    round_avg_ep_lengths = []

    for r in range(NUM_ROUNDS):
        print(f"\n=== Round {r+1}/{NUM_ROUNDS} ===")

        callback.ep_rewards.clear()
        callback.ep_lengths.clear()

        # Train for one round
        model.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, callback=callback)

        # Aggregate statistics
        avg_r = np.mean(callback.ep_rewards[-50:]) if len(callback.ep_rewards) > 0 else 0
        avg_len = np.mean(callback.ep_lengths[-50:]) if len(callback.ep_lengths) > 0 else 0

        round_avg_rewards.append(avg_r)
        round_avg_ep_lengths.append(avg_len)

        print(f"  Avg reward: {avg_r:.2f}")
        print(f"  Avg episode length: {avg_len:.1f}")

        # Save model
        model.save(os.path.join(TRIAL_DIR, f"sac_models/sac_round_{r+1}"))

        # Plot so far
        plot_training(round_avg_rewards, round_avg_ep_lengths, TRIAL_DIR)

    print("\nTraining complete!")
    env.close()
