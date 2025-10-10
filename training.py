import gymnasium as gym
import stable_baselines3 as sb3
# import pprint
import matplotlib.pyplot as plt
import cv2
# import acc
import os
# import json
from tqdm import trange

NUM_ROUNDS = 30
NUM_TRAINING_STEPS_PER_ROUND = 5000
NUM_TESTS_PER_ROUND = 100
MODEL_FILENAME_BASE = "carcontrol-dqn"

def test_model(env, model, video=None, msg=None):
    obs, info = env.reset()
    frame = env.render()
    ep_len = 0
    ep_rew = 0

    while True:
        action, _ = model.predict(obs)

        # Perform action and update total reward
        obs, reward, terminated, truncated, info = env.step(action) # info contains {'crash': crash}
        ep_rew += reward

        # Record frame to video
        if video:
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.putText(
                frame,
                msg,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,0),
                1,
                cv2.LINE_AA
            )
            video.write(frame)

        ep_len += 1

        if terminated or truncated:
            return ep_len, ep_rew
        
def plot_averages(avg_ep_lens, avg_ep_rews):
    fig, axs = plt.subplots(1,2)
    fig.tight_layout(pad=2.0)
    axs[0].plot(avg_ep_lens)
    axs[0].set_ylabel("Average episode length")
    axs[0].set_xlabel("Round")
    axs[1].plot(avg_ep_rews)
    axs[1].set_ylabel("Average episode reward")
    axs[1].set_xlabel("Round")

    plt.savefig("avg_lens_rews.png")

if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy" # required when running on remote server without GUI

    env = gym.make('versaille_env:versaille_env/acc-discrete-v0')
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="./training_videos",
        episode_trigger=lambda x: True,
        name_prefix="rl-video"
    )

    obs, info = env.reset()
    frame = env.render()
    # plt.imshow(frame)

    # Initialise model ---
    model = sb3.DQN( # DQN for discrete actions?
        'MlpPolicy',
        env,
        policy_kwargs={'net_arch': [256,256]},
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        verbose=0,
        tensorboard_log="car_control_dqn/"
    )
    
    # Training and testing ---
    FPS = 30
    FOURCC = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    VIDEO_FILENAME = "training.mp4"
    width = frame.shape[1]
    height = frame.shape[0]
    video = cv2.VideoWriter(VIDEO_FILENAME, FOURCC, FPS, (width, height))

    avg_ep_lens = []
    avg_ep_rews = []
    for rnd in trange(NUM_ROUNDS, desc="Training rounds"):
        model.learn(total_timesteps=NUM_TRAINING_STEPS_PER_ROUND, reset_num_timesteps=False) # train model
        model.save(f"{MODEL_FILENAME_BASE}_{rnd}")

        # Test the model in several episodes
        avg_ep_len = 0
        avg_ep_rew = 0
        for ep in trange(NUM_TESTS_PER_ROUND, desc=f"Testing round {rnd}", leave=False):
            # Only record video for the first test
            if ep == 0:
                ep_len, ep_rew = test_model(env, model, video, f"Round {rnd}")
            else:
                ep_len, ep_rew = test_model(env, model)

            # Accumulate avgs
            avg_ep_len += ep_len
            avg_ep_rew += ep_rew

        # Record and display avgs
        avg_ep_len /= NUM_TESTS_PER_ROUND
        avg_ep_lens.append(avg_ep_len)
        avg_ep_rew /= NUM_TESTS_PER_ROUND
        avg_ep_rews.append(avg_ep_rew)
        print(f"Round {rnd} | average test length: {avg_ep_len}, average test reward: {avg_ep_rew}")

    video.release() # close video writer

    plot_averages(avg_ep_lens, avg_ep_rews)

    print(f"Best performing model: {avg_ep_rews.index(max(avg_ep_rews))}")
    print(f"Avg episode reward for that model: {max(avg_ep_rews)}")






    