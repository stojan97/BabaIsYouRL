# q_learning_solver.py (with Metrics and Plotting)

import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import time

# Import the refactored environment and wrapper
from baba_is_you_env import BabaIsYouGridEnv, PositionalWrapper, FullStateWrapper


def plot_metrics(metrics):
    """
    Generates and displays plots for the training metrics.
    The success plot now shows a smoothed win percentage from 0% to 100%.
    """

    def moving_average(data, window_size):
        if len(data) < window_size:
            return np.array([])  # Return empty if not enough data
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    window_size = 100  # Window for smoothing the plots

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('BabaIsYou Q-Learning Agent Performance Metrics', fontsize=16)

    # 1. Plot Episode Rewards (No change here)
    rewards = metrics['rewards']
    smoothed_rewards = moving_average(rewards, window_size)
    axs[0, 0].plot(rewards, alpha=0.3, label='Raw Reward')
    if smoothed_rewards.any():
        axs[0, 0].plot(np.arange(window_size - 1, len(rewards)), smoothed_rewards, color='red',
                       label=f'Moving Avg (n={window_size})')
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Plot Steps per Episode (No change here)
    steps = metrics['steps']
    smoothed_steps = moving_average(steps, window_size)
    axs[0, 1].plot(steps, alpha=0.3, label='Raw Steps')
    if smoothed_steps.any():
        axs[0, 1].plot(np.arange(window_size - 1, len(steps)), smoothed_steps, color='green',
                       label=f'Moving Avg (n={window_size})')
    axs[0, 1].set_title('Steps per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Steps')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. --- MODIFIED: Plot Win Percentage ---
    successes = np.array(metrics['successes'])

    # Calculate the moving average of successes and convert to a percentage
    win_percentage = moving_average(successes, window_size) * 100

    # Plot the smoothed win percentage
    if win_percentage.any():
        axs[1, 0].plot(np.arange(window_size - 1, len(successes)), win_percentage,
                       color='purple', label=f'Win Rate (Moving Avg n={window_size})')

    axs[1, 0].set_title('Win Percentage')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Win Percentage (%)')
    axs[1, 0].set_ylim(-5, 105)  # Set Y-axis from -5% to 105% for nice padding
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # 4. Plot Epsilon Decay (No change here)
    epsilons = metrics['epsilons']
    axs[1, 1].plot(epsilons, color='orange')
    axs[1, 1].set_title('Epsilon Decay')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Epsilon Value')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def evaluate_agent(level_map, q_table, n_eval_episodes=10):
    """
    Evaluates the performance of the trained Q-learning agent.
    """
    print("\n--- Evaluating Trained Agent ---")
    # Create a new environment for evaluation, with rendering
    env = BabaIsYouGridEnv(level_map=level_map, render_mode="human")
    env = FullStateWrapper(env)

    successes = 0
    total_steps = 0
    episode_rewards = []

    for episode in range(n_eval_episodes):
        state, info = env.reset()
        done = False
        steps = 0
        print(f"\n--- Evaluation Episode {episode + 1} ---")
        terminated = False
        rewards = 0

        while not done:
            # Take the action with the highest Q-value (no exploration)
            action = np.argmax(q_table[state])

            print(f'Action: {action}, Steps: {steps}')
            next_state, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(2)

            state = next_state
            steps += 1
            done = terminated or truncated
            rewards += reward

        episode_rewards.append(rewards)

        if terminated:  # 'terminated' means a successful win
            successes += 1
            print(f"Result: Success in {steps} steps!")
        else:
            print(f"Result: Failure (timed out after {steps} steps).")

        total_steps += steps

    env.close()

    success_rate = (successes / n_eval_episodes) * 100
    avg_steps = total_steps / n_eval_episodes
    print("\n--- Evaluation Summary ---")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Steps per Episode: {avg_steps:.2f}")


def to_pos(idx, num_cols):
    row = idx // num_cols
    col = idx % num_cols
    return row, col

# --- Main script ---

# --- Define the Level Layout ---
LEVEL_LAYOUT = [
    "WWWWWWW",
    "W.....W",
    "W.B...W",
    "W..fF.W",
    "W.i...W",
    "W..w..W",
    "WWWWWWW"
]

# LEVEL_LAYOUT = [
#     "WWWWWWW",
#     "W.....W",
#     "W.B...W",
#     "W...F.W",
#     "W.....W",
#     "W.....W",
#     "WWWWWWW"
# ]


# --- Environment Setup ---
env = BabaIsYouGridEnv(level_map=LEVEL_LAYOUT, render_mode="rgb_array")
env = FullStateWrapper(env)

state_size = env.observation_space.n
action_size = env.action_space.n

# --- Hyperparameters ---
episodes = 5000
alpha = 1
# alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
# Linear decay over the first 75% of episodes
# epsilon_decay_rate = (epsilon - epsilon_min) / (episodes * 0.75)
epsilon_decay_rate = epsilon / episodes * 0.5
# --- Metrics Collection ---
# Use a dictionary to store lists of metrics for each episode
metrics = {
    'rewards': [],
    'steps': [],
    'successes': [],
    'epsilons': []
}

Q_table = np.full((state_size, action_size), 51)
# Q_table = np.zeros((state_size, action_size))
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
        # if random.uniform(0, 1) <= epsilon:
        #     action = env.action_space.sample()
        # else:
        #     action = np.argmax(Q_table[state])

        action = np.argmax(Q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-table update
        old_value = Q_table[state, action]
        next_max = np.max(Q_table[next_state])
        target = float(reward) + gamma * next_max
        td_difference = target - old_value

        # if episode % 100 == 0:
        #     print(f"Q-table: {Q_table}")
        #     print(f"State: {state}, Action: {action}, Reward: {reward:.2f}, Next State: {next_state}, Next State Q-Values: {Q_table[next_state]}")

        Q_table[state, action] = old_value + alpha * td_difference

        state = next_state
        total_episode_reward += reward
        episode_steps += 1
        done = terminated or truncated

    # --- Record Metrics for the Episode ---
    metrics['rewards'].append(total_episode_reward)
    metrics['steps'].append(episode_steps)
    metrics['successes'].append(1 if terminated else 0)
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

# --- Evaluate the final policy ---
# evaluate_agent(LEVEL_LAYOUT, Q_table, n_eval_episodes=1)
