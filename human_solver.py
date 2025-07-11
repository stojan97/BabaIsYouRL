from baba_is_you_env import BabaIsYouGridEnv, FullStateWrapper
import time

# --- Define the Level Layout ---
# W = Wall, B = Baba Agent, F = Flag (Goal)
# f = 'FLAG' text, i = 'IS' text, w = 'WIN' text
# '.' = empty space
LEVEL_LAYOUT = [
    ".A...WF",
    "....OW.",
    ".f...W.",
    ".i...W.",
    "..w....",
    "....oie"
]

if __name__ == "__main__":

    env = BabaIsYouGridEnv(LEVEL_LAYOUT, render_mode="human")
    env = FullStateWrapper(env)
    obs, info = env.reset()

    print("Observation :", obs)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    for i in range(10):
        print(env.action_space.sample())

    actions = [0,0,0,0,1]

    # actions.extend([3] * 20)

    for i in range(len(actions)):
        action = actions[i]  # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        print(f"Obs: {obs}, Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()
            break

        time.sleep(1)

    env.close()
