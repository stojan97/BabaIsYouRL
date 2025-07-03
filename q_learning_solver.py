# q_learning_solver.py (with Metrics and Plotting)

import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

# Import the refactored environment and wrapper
from baba_is_you_env import BabaIsYouGridEnv, PositionalWrapper


def plot_metrics(metrics):
    """
    Generates and displays plots for the training metrics.
    """

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    window_size = 100  # Window for smoothing the plots

    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Q-Learning Agent Performance Metrics', fontsize=16)

    # 1. Plot Episode Rewards
    rewards = metrics['rewards']
    smoothed_rewards = moving_average(rewards, window_size)

    axs[0, 0].plot(rewards, alpha=0.3, label='Raw Reward')
    axs[0, 0].plot(np.arange(window_size - 1, len(rewards)), smoothed_rewards, color='red',
                   label=f'Moving Avg (n={window_size})')
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Plot Steps per Episode
    steps = metrics['steps']
    smoothed_steps = moving_average(steps, window_size)
    axs[0, 1].plot(steps, alpha=0.3, label='Raw Steps')
    axs[0, 1].plot(np.arange(window_size - 1, len(steps)), smoothed_steps, color='green',
                   label=f'Moving Avg (n={window_size})')
    axs[0, 1].set_title('Steps per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Steps')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Plot Success Rate
    successes = metrics['successes']
    success_rate = moving_average(successes, window_size) * 100  # As a percentage
    axs[1, 0].plot(np.arange(window_size - 1, len(successes)), success_rate, color='purple')
    axs[1, 0].set_title(f'Success Rate (Moving Avg n={window_size})')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Success Rate (%)')
    axs[1, 0].set_ylim(0, 100)  # Percentage
    axs[1, 0].grid(True)

    # 4. Plot Epsilon Decay
    epsilons = metrics['epsilons']
    axs[1, 1].plot(epsilons, color='orange')
    axs[1, 1].set_title('Epsilon Decay')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Epsilon Value')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Main script ---

# --- Define the Level Layout ---
LEVEL_LAYOUT = [
    "WWWWWWW",
    "W.....W",
    "W.B.i.W",
    "W...F.W",
    "W.f...W",
    "W...w.W",
    "WWWWWWW"
]

# --- Environment Setup ---
# Use render_mode="rgb_array" during training for speed, "human" for final eval
env = BabaIsYouGridEnv(level_map=LEVEL_LAYOUT, render_mode="rgb_array")
env = PositionalWrapper(env)

state_size = env.observation_space.n
action_size = env.action_space.n
Q_table = np.zeros((state_size, action_size))

# --- Hyperparameters ---
episodes = 20000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
# Linear decay over the first 75% of episodes
epsilon_decay_rate = (epsilon - epsilon_min) / (episodes * 0.75)

# --- Metrics Collection ---
# Use a dictionary to store lists of metrics for each episode
metrics = {
    'rewards': [],
    'steps': [],
    'successes': [],  # 1 if win, 0 otherwise
    'epsilons': []
}
# Use a deque for efficient calculation of recent performance
recent_rewards = deque(maxlen=100)

# --- Training Loop ---
# Wrap the range with tqdm for a nice progress bar
progress_bar = tqdm(range(episodes), desc="Training Q-Learning Agent")

for episode in progress_bar:
    state, info = env.reset()
    done = False

    total_episode_reward = 0
    episode_steps = 0
    terminated = False

    while not done:
        if random.uniform(0, 1) <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-table update
        old_value = Q_table[state, action]
        next_max = np.max(Q_table[next_state])
        Q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_episode_reward += reward
        episode_steps += 1
        done = terminated or truncated

    # --- Record Metrics for the Episode ---
    metrics['rewards'].append(total_episode_reward)
    metrics['steps'].append(episode_steps)
    metrics['successes'].append(1 if terminated else 0)  # 'terminated' means we reached the goal
    metrics['epsilons'].append(epsilon)
    recent_rewards.append(total_episode_reward)

    # Update Epsilon
    epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)

    # Update the progress bar with the average reward of the last 100 episodes
    avg_reward = sum(recent_rewards) / len(recent_rewards)
    progress_bar.set_postfix(avg_reward=f"{avg_reward:.3f}", epsilon=f"{epsilon:.3f}")

print("\n--- Training Finished ---")
env.close()

# --- Plot the Results ---
print("Generating plots...")
plot_metrics(metrics)