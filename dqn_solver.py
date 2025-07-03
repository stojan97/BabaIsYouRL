import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import warnings

from baba_is_you_env import BabaIsYouGridEnv, MyImgObsWrapper


# Create a custom feature extractor for small observations
class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[2]  # Channels last in your case

        # Using smaller kernels and adding padding to preserve spatial dimensions
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        # Permute the dimensions from [batch, height, width, channel] to [batch, channel, height, width]
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


def main():
    # 1. Create and wrap the environment
    env = BabaIsYouGridEnv(size=7, render_mode="rgb_array")  # Using original size
    env = FullyObsWrapper(env)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")

    print("Environment check passed!")

    # 2. Instantiate the DQN Model with custom policy
    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=256),  # Reduced feature dim
    )

    # Check if tensorboard is installed
    # try:
    #     import tensorboard
    #     tensorboard_log = "./baba_dqn_tensorboard/"
    # except ImportError:
    #     warnings.warn("TensorBoard not installed. Disabling TensorBoard logging.")
    #     tensorboard_log = None
    #
    # model = DQN(
    #     "CnnPolicy",
    #     env,
    #     policy_kwargs=policy_kwargs,  # Use our custom policy architecture
    #     verbose=1,
    #     buffer_size=50000,
    #     learning_starts=1000,
    #     batch_size=32,  # Reduced batch size
    #     learning_rate=1e-4,
    #     gamma=0.99,
    #     target_update_interval=1000,
    #     exploration_fraction=0.5,
    #     exploration_final_eps=0.05,
    #     tensorboard_log=tensorboard_log  # Now conditionally set
    # )
    #
    # # 3. Train the Model
    # print("\n--- Starting Training ---")
    # model.learn(total_timesteps=100000, progress_bar=True)
    # print("--- Training Finished ---")
    #
    # # 4. Save and Evaluate
    model_path = "dqn_baba_is_you_sb3.zip"
    # model.save(model_path)
    # print(f"Model saved to {model_path}")
    # del model
    #
    # print("\n--- Evaluating Trained Model ---")
    loaded_model = DQN.load(model_path)

    eval_env = BabaIsYouGridEnv(size=7, render_mode="human")  # Match training size
    eval_env = FullyObsWrapper(eval_env)

    num_eval_episodes = 10
    success_count = 0
    for episode in range(num_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        if reward > 0:  # A positive final reward indicates a win
            success_count += 1
            print(f"Evaluation Episode {episode + 1}: Solved!")
        else:
            print(f"Evaluation Episode {episode + 1}: Failed.")

    print(f"\nSuccess Rate: {success_count / num_eval_episodes * 100}%")
    eval_env.close()


if __name__ == "__main__":
    main()