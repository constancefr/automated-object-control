# eval_models.py

import os
import glob
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN

# ======================================================
# CONFIG – CHANGE THESE TO MATCH YOUR SETUP
# ======================================================

# Use your env ID from gym.register in acc.py
ENV_ID = "custom_car_env:custom_car_env/acc-discrete-v0"   # or "acc-discrete-v0" if that's the one you use

# Folder where your trained models (.zip) are saved
# e.g. from your logs: results_dqn/trial_12/dqn_models or results_dqn/trial_12/car_dqn
MODELS_DIR = "results_dqn/trial_12/dqn_models"

# How many episodes to run per model for evaluation
N_EPISODES = 50

# Whether to render during evaluation (will slow things down a lot)
RENDER = False


def evaluate_model(env_id: str, model_path: str, n_episodes: int, render: bool = False):
    """Evaluate a single DQN model on the given env."""
    print(f"\nEvaluating model: {os.path.basename(model_path)}")

    render_mode = "rgb_array" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    model = DQN.load(model_path, env=env, device="cpu")

    ep_lengths = []
    ep_rewards = []
    crash_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        crashed = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if info.get("crash", False):
                crashed = True

            if render:
                _ = env.render()  # ignore frame; just show animation if using human/rgb_array + external viewer

        ep_lengths.append(steps)
        ep_rewards.append(total_reward)
        if crashed:
            crash_count += 1

    env.close()

    ep_lengths = np.array(ep_lengths, dtype=np.float32)
    ep_rewards = np.array(ep_rewards, dtype=np.float32)
    crash_rate = crash_count / n_episodes

    mean_len = float(ep_lengths.mean())
    mean_rew = float(ep_rewards.mean())

    print(
        f"  -> mean_len = {mean_len:.1f}, "
        f"mean_rew = {mean_rew:.2f}, "
        f"crash_rate = {crash_rate:.1%}"
    )

    return mean_len, mean_rew, crash_rate


def main():
    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    model_paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.zip")))
    if not model_paths:
        raise FileNotFoundError(f"No .zip models found in {MODELS_DIR}")

    print(f"Found {len(model_paths)} models in {MODELS_DIR}")

    results = []  # list of (path, mean_len, mean_rew, crash_rate)

    for model_path in model_paths:
        mean_len, mean_rew, crash_rate = evaluate_model(
            ENV_ID, model_path, N_EPISODES, render=RENDER
        )
        results.append((model_path, mean_len, mean_rew, crash_rate))

    # Sort by mean reward (descending)
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

    print("\n================ SUMMARY (sorted by mean reward) ================")
    for path, mean_len, mean_rew, crash_rate in results_sorted:
        print(
            f"{os.path.basename(path):25s} | "
            f"len={mean_len:6.1f} | "
            f"rew={mean_rew:7.2f} | "
            f"crash_rate={crash_rate:6.1%}"
        )

    best_path, best_len, best_rew, best_crash = results_sorted[0]
    print("\nBest model by reward:")
    print(
        f"{os.path.basename(best_path)} | "
        f"len={best_len:.1f}, rew={best_rew:.2f}, crash_rate={best_crash:.1%}"
    )


if __name__ == "__main__":
    main()
