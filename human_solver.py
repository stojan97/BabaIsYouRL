from baba_is_you_env import BabaIsYouGridEnv, PositionalWrapper

# --- Define the Level Layout ---
# W = Wall, B = Baba Agent, F = Flag (Goal)
# f = 'FLAG' text, i = 'IS' text, w = 'WIN' text
# '.' = empty space
LEVEL_LAYOUT = [
    "WWWWWWW",
    "WB....W",
    "W..i..W",
    "W...F.W",
    "W.f...W",
    "W...w.W",
    "WWWWWWW"
]

if __name__ == "__main__":

    env = BabaIsYouGridEnv(LEVEL_LAYOUT, render_mode="human")
    env = PositionalWrapper(env)
    obs, info = env.reset()

    print("Observation :", obs)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    for i in range(10):
        print(env.action_space.sample())

    actions = [3, 3, 3]

    #
    actions.extend([1] * 20)

    actions.extend([3] * 20)

    for i in range(len(actions)):
        action = actions[i]  # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Obs: {obs}, Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()
            break
    env.close()
