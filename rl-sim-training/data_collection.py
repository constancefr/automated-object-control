import os
import json
import cv2
from tqdm import trange
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize, DummyVecEnv
from continuous_training import make_env

def collect_data(
    env,
    model,
    jsonl_path="dataset.jsonl",
    total_samples=20_000,
    record_video=False,
    video_folder="./videos",
    video_every=5,          # record every N episodes
    video_length=1000,
    msg="collecting",
):
    """
    TODO: change this description!!
    Collect rollouts from a trained model into a JSONL dataset.

    Each line of the JSONL file contains:
    {
      "episode_id": int,
      "t": int,
      "obs": list[float],
      "action": int,
      "reward": float,
      "done": bool,
      "collision": bool
    }
    """

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    if record_video:
        os.makedirs(video_folder, exist_ok=True)

    total_collected = 0
    episode_id = 0

    with open(jsonl_path, "w") as f:
        with trange(total_samples, desc="Collecting data") as pbar:
            while total_collected < total_samples:
                obs, info = env.reset()
                done = False
                # crashed = False
                t = 0

                # Get initial front state from info
                front_state = info[0].get('front_state', np.zeros(2))
                front_action = info[0].get('front_action', 1)

                # Handle VecNormalize to get unnormalized obs for saving
                # if isinstance(env, VecNormalize):
                #     raw_obs = env.get_original_obs()
                #     if isinstance(raw_obs, np.ndarray):
                #         obs_save = raw_obs.copy()
                #     else:
                #         obs_save = np.array(raw_obs)
                # else:
                #     obs_save = obs.copy() if isinstance(obs, np.ndarray) else np.array(obs)

                if record_video and episode_id % video_every == 0:
                    video_path = os.path.join(video_folder, f"episode_{episode_id}.mp4")
                    frame = env.render()
                    height, width = frame.shape[:2]
                    video = cv2.VideoWriter(
                        video_path, cv2.VideoWriter.fourcc(*"mp4v"), 30, (width, height)
                    )
                else:
                    video = None

                while not done and total_collected < total_samples:
                    # Get unnormalized obs for saving
                    if hasattr(env, 'get_original_obs'):
                        raw_obs = env.get_original_obs()
                        obs_save = raw_obs[0].copy() if isinstance(raw_obs, np.ndarray) else np.array(raw_obs)
                    else:
                        obs_save = obs[0].copy() if isinstance(obs, np.ndarray) else np.array(obs)


                    # Predict action
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated[0] or truncated[0]

                    # Front car info
                    front_state = info[0].get('front_state', front_state)
                    front_action = info[0].get('front_action', front_action)
                    crashed = info[0].get("crash", False)

                    record = {
                        "episode_id": episode_id,
                        "timestep": t,
                        "ego_state": obs_save.tolist(), # [pos, vel]
                        "ego_action": int(action[0]),
                        "front_state": front_state.tolist() if hasattr(front_state, 'tolist') else front_state,
                        "front_action": int(front_action),
                        "reward": float(reward),
                        "terminated": bool(terminated[0]),
                        "truncated": bool(truncated[0]),
                        "collision": bool(crashed),
                        "done": bool(done)
                    }
                    f.write(json.dumps(record) + "\n")

                    total_collected += 1
                    t += 1
                    pbar.update(1)

                    if record_video and video is not None:
                        frame = env.render()
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame = cv2.putText(
                            frame,
                            f"{msg} ep:{episode_id} t:{t}",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        video.write(frame)

                    if done:
                        break

                if video is not None:
                    video.release()
                episode_id += 1

    print(f"Collected {total_collected} samples across {episode_id} episodes.")

def main():
    # Load environment & trained model
    env = DummyVecEnv([make_env])
    env = VecNormalize.load("results/vecnormalize_stats.pkl", env)
    env.training = False
    env.norm_reward = False

    model = DQN.load("results/carcontrol-dqn_final", env=env)

    # Collect dataset
    collect_data(
        env,
        model,
        jsonl_path="./results/dataset.jsonl",
        total_samples=20_000,
        record_video=False,       # turn off if running on cluster?
        video_folder="./results/videos",
        video_every=10,
        msg="data collection"
    )

if __name__ == "__main__":
    main()