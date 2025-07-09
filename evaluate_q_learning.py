import numpy as np
from tqdm import tqdm


def moving_average(data, window_size):
    """
    Produces the moving average from a given array with a given window size.
    The window slides along the array and averages the values.

    [1, 2, 3, 4] with window size 3 [1,1,1] -> [2,
    (1 + 2 + 3) / 3 = 6 / 3 = 2 (1/3 + 2/3 + 3/3) = 6/3 = 2)
    (2 + 3 + 4) / 3 = 9 / 3 = 3 (2/3 + 3/3 + 4/3) = 9/3 = 3)

    :param data: data array
    :param window_size: the window size
    :return: the moving average array
    """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def evaluate_agent(env, max_steps, n_eval_episodes, Q_table, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q_table: The Q-table
    :param seed: The evaluation seed array
    """
    episode_rewards = []

    progres = tqdm(range(n_eval_episodes))
    for episode in progres:

        state, info = env.reset()

        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(Q_table[state])
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break

            state = new_state

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


LEVEL_LAYOUT = [
    "WWWWWWW",
    "WB....W",
    "W..i..W",
    "W...F.W",
    "W.f...W",
    "W...w.W",
    "WWWWWWW"
]
