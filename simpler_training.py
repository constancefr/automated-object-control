import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import trange
from stable_baselines3 import DQN
import torch
import os

def test_model(env, model, video_writer=None, msg=None):
    '''
    Runs one episode and returns reward + frames.
    '''
    obs, info = env.reset()
    ep_len, ep_rew = 0, 0
    frames = []
    frame_count = 0

    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rew += reward
        ep_len += 1

        # Render frame only if recording video
        frame = None
        if video_writer is not None:
            frame = env.render()
        
        if frame is not None:
            if video_writer is not None:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if msg is not None:
                    frame_bgr = cv2.putText(
                        frame_bgr, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                    )
                video_writer.write(frame_bgr)
                frame_count += 1

            frames.append(frame)

        if terminated or truncated:
            break
            
    return ep_len, ep_rew, frames

def setup_environment():
    env = gym.make("versaille_env:versaille_env/acc-discrete-v0", render_mode="rgb_array")
    
    # Test rendering
    obs, info = env.reset()
    frame = env.render()
    
    if frame is None:
        raise RuntimeError("Environment rendering returned None")
    
    height = frame.shape[0]
    width = frame.shape[1]
    
    return env, width, height

def main():
    env, width, height = setup_environment()
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=128,
        gamma=0.98,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_final_eps=0.02,   # Lower final exploration
        exploration_fraction=0.3,     # Longer exploration period
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU),
        verbose=0,
    )

    NUM_ROUNDS = 100
    NUM_TRAINING_STEPS_PER_ROUND = 10000
    NUM_TESTS_PER_ROUND = 100
    MODEL_FILENAME_BASE = "models/acc_dqn"

    FPS = 30
    VIDEO_INTERVAL = 10
    VIDEO_DIR = "videos"

    os.makedirs(VIDEO_DIR, exist_ok=True)

    avg_ep_lens, avg_ep_rews = [], []

    # Training loop
    for rnd in trange(NUM_ROUNDS, desc="Training rounds"):
        model.learn(total_timesteps=NUM_TRAINING_STEPS_PER_ROUND, progress_bar=True)
        model.save(f"{MODEL_FILENAME_BASE}_{rnd}")

        total_len, total_rew = 0, 0
        
        # Create video writer for first test episode of recorded rounds
        video_writer = None
        if rnd % VIDEO_INTERVAL == 0:
            fourcc = cv2.VideoWriter.fourcc(*'avc1')
            # fourcc = cv2.VideoWriter.fourcc('m','p','4','v') # for some reason this doesn't work
            video_path = os.path.join(VIDEO_DIR, f"training_round_{rnd:03d}.mp4")
            video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))
            print(f"Recording video: {video_path}")

        for ep in range(NUM_TESTS_PER_ROUND):
            if ep == 0 and video_writer is not None:
                ep_len, ep_rew, frames = test_model(env, model, video_writer, f"Round {rnd}")
                print(f"Round {rnd}, Episode {ep}: Length={ep_len}, Reward={ep_rew}, Frames={len(frames)}")
            else:
                ep_len, ep_rew, _ = test_model(env, model)
                
            total_len += ep_len
            total_rew += ep_rew

        if video_writer is not None:
            video_writer.release()
            print(f"Video saved: {video_path}")

        avg_len = total_len / NUM_TESTS_PER_ROUND
        avg_rew = total_rew / NUM_TESTS_PER_ROUND
        avg_ep_lens.append(avg_len)
        avg_ep_rews.append(avg_rew)

        print(f"Round {rnd:02d} | Avg Len: {avg_len:.1f} | Avg Reward: {avg_rew:.2f}")

    env.close()
    
    # Plot stats
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(avg_ep_lens, label="Episode Length")
    plt.xlabel("Round")
    plt.ylabel("Length")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(avg_ep_rews, label="Episode Reward")
    plt.xlabel("Round")
    plt.ylabel("Reward")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_stats.png")
    plt.show()

if __name__ == "__main__":
    main()