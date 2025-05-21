# plot_metrics.py
import os
import json
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def extract_metrics_from_runs(runs_dir="runs"):
    """Extract rewards and losses from all runs in the runs directory."""
    rewards = []
    losses = []
    for subdir in os.listdir(runs_dir):
        log_path = os.path.join(runs_dir, subdir)
        if not os.path.isdir(log_path):
            continue
        event_acc = EventAccumulator(log_path)
        event_acc.Reload()
        
        # Extract rewards
        if 'Reward' in event_acc.Tags()['scalars']:
            reward_scalars = event_acc.Scalars('Reward')
            rewards.extend([(s.step + 1, s.value) for s in reward_scalars])  # step + 1 for 1-based episodes
        
        # Extract losses
        if 'Loss' in event_acc.Tags()['scalars']:
            loss_scalars = event_acc.Scalars('Loss')
            losses.extend([(s.step, s.value) for s in loss_scalars])
    
    # Sort by step to ensure chronological order
    rewards = sorted(rewards, key=lambda x: x[0])
    losses = sorted(losses, key=lambda x: x[0])
    return rewards, losses

def load_rewards_from_json(json_path="episode_rewards.json"):
    """Load rewards from episode_rewards.json if it exists."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            rewards = json.load(f)
        return [(i + 1, r) for i, r in enumerate(rewards)]
    return []

def plot_metrics(runs_dir="runs", json_path="episode_rewards.json"):
    """Plot rewards and losses using Matplotlib."""
    # Try loading rewards from JSON first, fall back to runs
    rewards = load_rewards_from_json(json_path)
    if not rewards:
        rewards, losses = extract_metrics_from_runs(runs_dir)
    else:
        _, losses = extract_metrics_from_runs(runs_dir)
    
    if not rewards and not losses:
        print("No data found in runs or episode_rewards.json")
        return
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    if rewards:
        episodes, reward_values = zip(*rewards)
        ax1.plot(episodes, reward_values, label='Reward', color='blue')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Rewards')
        ax1.grid(True)
        ax1.legend()
    
    # Plot losses
    if losses:
        steps, loss_values = zip(*losses)
        ax2.plot(steps, loss_values, label='Loss', color='red')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True)
        ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

if __name__ == "__main__":
    plot_metrics()