import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

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
# Plot helper
# ==============================================
def plot_training(round_rewards, round_lengths, out_dir):
    plt.figure(figsize=(12, 5))

    # Reward
    plt.subplot(1, 2, 1)
    plt.plot(round_rewards, label="Avg episodic reward")
    plt.xlabel("Training round")
    plt.ylabel("Reward")
    plt.grid(True)

    # Episode length
    plt.subplot(1, 2, 2)
    plt.plot(round_lengths, label="Avg episode length")
    plt.xlabel("Training round")
    plt.ylabel("Length")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_plots.png"))
    plt.close()


# ==============================================
# Main training loop
# ==============================================
if __name__ == "__main__":

    # Output directory for all training runs
    TRIAL_DIR = "./results_sac"
    os.makedirs(TRIAL_DIR, exist_ok=True)

    # Create monitored environment
    env = gym.make("custom_car_env:custom_car_env/acc-continuous-v0", render_mode=None)

    env = Monitor(env)

    # SAC model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=os.path.join(TRIAL_DIR, "tb_logs")
    )

    # Training settings
    NUM_ROUNDS = 80
    # STEPS_PER_ROUND = 10_000
    STEPS_PER_ROUND = 5_000

    round_avg_rewards = []
    round_avg_ep_lengths = []

    stats_callback = EpisodeStatsCallback()

    # Training rounds
    for r in range(NUM_ROUNDS):

        print(f"\n=== Round {r+1}/{NUM_ROUNDS} ===")

        # Reset callback stats
        stats_callback.ep_rewards.clear()
        stats_callback.ep_lengths.clear()

        # Train
        model.learn(
            total_timesteps=STEPS_PER_ROUND,
            reset_num_timesteps=False,
            callback=stats_callback,
        )

        # Compute stats
        if len(stats_callback.ep_rewards) > 0:
            avg_r = np.mean(stats_callback.ep_rewards)
            avg_len = np.mean(stats_callback.ep_lengths)
        else:
            avg_r = 0.0
            avg_len = 0.0

        round_avg_rewards.append(avg_r)
        round_avg_ep_lengths.append(avg_len)

        print(f"  Avg reward: {avg_r:.2f}")
        print(f"  Avg episode length: {avg_len:.1f}")

        # Save model
        model.save(os.path.join(TRIAL_DIR, f"sac_round_{r+1}"))

        # Plot so far
        plot_training(round_avg_rewards, round_avg_ep_lengths, TRIAL_DIR)

    print("\nTraining complete!")
    env.close()
