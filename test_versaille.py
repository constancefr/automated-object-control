"""
Run VerifiedHighwayEnv and generate videos
"""

import gymnasium as gym
import versaille_env
import numpy as np
from pathlib import Path

def setup_video_recording(env, video_folder="./videos"):
    Path(video_folder).mkdir(exist_ok=True)
    
    # Wrap the environment with video recording
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,  # record every episode
        name_prefix="verified_highway"
    )
    return env

def run_episode_with_video(env, episode_num, max_steps=100):
    print(f"Starting Episode {episode_num}...")
    
    obs, info = env.reset()
    total_reward = 0
    safety_violations = 0
    
    for step in range(max_steps):
        # Random action for now
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        if info.get('safety_violation', False):
            safety_violations += 1
        
        # env.render()
        
        if done:
            break
    
    print(f"Episode {episode_num} finished:")
    print(f"  Steps: {step + 1}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Safety violations: {safety_violations}")
    
    return total_reward, safety_violations

def run_demo_episodes(num_episodes=3):    
    # Create environment
    # env = gym.make('VerifiedHighway-v1', render_mode='rgb_array')
    env = gym.make('versaille_env:versaille_env/acc-discrete-v0')
    env = setup_video_recording(env)
    
    for episode in range(num_episodes):
        run_episode_with_video(env, episode + 1)
    
    env.close()

def run_with_policy(env, policy_type="cautious", max_steps=1000):
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        if policy_type == "brake":
            action = 2 # brake
        elif policy_type == "accelerate":
            action = 0 # accelerate
        elif policy_type == "idle":
            action = 1
        elif policy_type == "random":
            action = env.action_space.sample()
        else:
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return total_reward

def compare_policies():    
    policies = ["brake", "accelerate", "idle", "random"]
    
    for policy in policies:
        print(f"\n--- Testing {policy} policy ---")
        
        # env = gym.make('VerifiedHighway-v1', render_mode='rgb_array')
        env = gym.make('versaille_env:versaille_env/acc-discrete-v0')
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="./videos",
            episode_trigger=lambda x: True,
            name_prefix=f"policy_{policy}"
        )
        
        reward = run_with_policy(env, policy)
        print(f"{policy} policy total reward: {reward:.2f}")
        
        env.close()

if __name__ == "__main__":
    import os
    os.makedirs("./videos", exist_ok=True)
    
    # print("\nRunning simple demo episodes")
    # run_demo_episodes(num_episodes=2)
    
    print("\nComparing different driving policies")
    compare_policies()