import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import trange
from stable_baselines3 import DQN
import torch
import os

# make sure env gets registered
import custom_car_env.envs.acc

trial_round = 12
trial_dir = f"results_dqn/trial_{trial_round}"


def test_model(env, model, video_writer=None, msg=None):
    """
    Runs one episode and returns:
        ep_len, ep_rew, crashed, frames, safe_fraction, avg_ego_speed

    crashed is True if info['crash'] was ever True during the episode.
    safe_fraction is the fraction of steps where the front headway is >= MIN_SEPARATION.
    avg_ego_speed is the mean ego speed over the episode.
    """
    obs, info = env.reset()
    ep_len, ep_rew = 0, 0.0
    frames = []
    crashed = False

    # --- NEW METRICS ---
    safe_steps = 0
    total_steps = 0
    ego_speed_sum = 0.0
    min_sep = env.unwrapped.MIN_SEPARATION

    while True:
        # deterministic policy for evaluation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rew += reward
        ep_len += 1

        # track crash flag (front or back)
        if info.get("crash", False):
            crashed = True

        # obs[0] is rel_front_dist = ego_pos - front_pos
        rel_front_dist = float(obs[0])
        front_gap = max(0.0, -rel_front_dist)  # distance to front car

        if front_gap >= min_sep:
            safe_steps += 1

        # ego velocity is state[1] in your env
        ego_vel = float(env.unwrapped.state[1])
        ego_speed_sum += ego_vel

        total_steps += 1

        frame = None
        if video_writer is not None:
            frame = env.render()

        if frame is not None:
            if video_writer is not None:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if msg is not None:
                    frame_bgr = cv2.putText(
                        frame_bgr,
                        msg,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                video_writer.write(frame_bgr)

            frames.append(frame)

        if terminated or truncated:
            break

    safe_fraction = safe_steps / total_steps if total_steps > 0 else 0.0
    avg_ego_speed = ego_speed_sum / total_steps if total_steps > 0 else 0.0

    return ep_len, ep_rew, crashed, frames, safe_fraction, avg_ego_speed


def setup_environment():
    env = gym.make("acc-continuous-v0", render_mode="rgb_array")

    # Test rendering once to get frame size
    obs, info = env.reset()
    frame = env.render()
    if frame is None:
        raise RuntimeError("Environment rendering returned None")

    height, width = frame.shape[0], frame.shape[1]
    return env, width, height


def main():
    env, width, height = setup_environment()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-5,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=2,
        target_update_interval=20_000,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=f"{trial_dir}/car_dqn/",
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.5,  # decay over first 50% of *total* steps
    )

    NUM_ROUNDS = 200
    NUM_TRAINING_STEPS_PER_ROUND = 5_000
    NUM_TESTS_PER_ROUND = 30
    MODEL_FILENAME_BASE = f"{trial_dir}/dqn_models/acc_dqn"

    FPS = 30
    VIDEO_INTERVAL = 10
    VIDEO_DIR = f"{trial_dir}/videos"
    os.makedirs(VIDEO_DIR, exist_ok=True)

    avg_ep_lens = []
    avg_ep_rews = []
    avg_crash_rates = []
    avg_safe_headway = []
    avg_ego_speeds = []

    for rnd in trange(NUM_ROUNDS, desc="Training rounds"):
        # train for one block of timesteps
        model.learn(total_timesteps=NUM_TRAINING_STEPS_PER_ROUND, reset_num_timesteps=False, progress_bar=True)
        model.save(f"{MODEL_FILENAME_BASE}_{rnd}")

        total_len = 0.0
        total_rew = 0.0
        crash_count = 0
        total_safe_fraction = 0.0
        total_ego_speed = 0.0

        video_writer = None
        if rnd % VIDEO_INTERVAL == 0:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            video_path = os.path.join(VIDEO_DIR, f"training_round_{rnd:03d}.mp4")
            video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))
            print(f"Recording video: {video_path}")

        for ep in range(NUM_TESTS_PER_ROUND):
            if ep == 0 and video_writer is not None:
                ep_len, ep_rew, crashed, frames, safe_frac, avg_speed = test_model(
                    env, model, video_writer, f"Round {rnd}"
                )
                print(
                    f"Round {rnd}, Episode {ep}: "
                    f"Length={ep_len}, Reward={ep_rew:.2f}, Frames={len(frames)}, "
                    f"Crashed={crashed}, SafeFrac={safe_frac:.2f}, "
                    f"AvgSpeed={avg_speed:.2f}"
                )
            else:
                ep_len, ep_rew, crashed, _, safe_frac, avg_speed = test_model(env, model)

            total_len += ep_len
            total_rew += ep_rew
            total_safe_fraction += safe_frac
            total_ego_speed += avg_speed

            if crashed:
                crash_count += 1

        if video_writer is not None:
            video_writer.release()
            print(f"Video saved: {video_path}")

        avg_len = total_len / NUM_TESTS_PER_ROUND
        avg_rew = total_rew / NUM_TESTS_PER_ROUND
        crash_rate = crash_count / NUM_TESTS_PER_ROUND
        avg_safe = total_safe_fraction / NUM_TESTS_PER_ROUND
        avg_speed = total_ego_speed / NUM_TESTS_PER_ROUND

        avg_ep_lens.append(avg_len)
        avg_ep_rews.append(avg_rew)
        avg_crash_rates.append(crash_rate)
        avg_safe_headway.append(avg_safe)
        avg_ego_speeds.append(avg_speed)

        print(
            f"Round {rnd:02d} | "
            f"Avg Len: {avg_len:.1f} | "
            f"Avg Reward: {avg_rew:.2f} | "
            f"Crash Rate: {crash_rate:.1%} | "
            f"Safe Headway: {avg_safe:.2%} | "
            f"Avg Ego Speed: {avg_speed:.2f}"
        )

    env.close()

    # ---- PLOTTING (4 graphs) ----
    os.makedirs(trial_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))

    rounds = np.arange(len(avg_ep_lens))

    # 1) Avg episode length
    plt.subplot(2, 2, 1)
    plt.plot(rounds, avg_ep_lens, label="Episode Length")
    plt.xlabel("Round")
    plt.ylabel("Length")
    plt.legend()

    # 2) Crash rate (%)
    plt.subplot(2, 2, 2)
    crash_percent = [c * 100.0 for c in avg_crash_rates]
    plt.plot(rounds, crash_percent, label="Crash Rate (%)")
    plt.xlabel("Round")
    plt.ylabel("Crash Rate (%)")
    plt.legend()

    # 3) Avg episode reward
    plt.subplot(2, 2, 3)
    plt.plot(rounds, avg_ep_rews, label="Avg Episode Reward")
    plt.xlabel("Round")
    plt.ylabel("Reward")
    plt.legend()

    # 4) Safe headway time (%)
    plt.subplot(2, 2, 4)
    safe_percent = [s * 100.0 for s in avg_safe_headway]
    plt.plot(rounds, safe_percent, label="Safe Headway Time (%)")
    plt.xlabel("Round")
    plt.ylabel("Safe Headway Time (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{trial_dir}/training_stats_4plots.png")
    plt.show()


if __name__ == "__main__":
    main()
