import gymnasium as gym
import numpy as np
from new_ot2_gym_wrapper import CustomEnv  

def main():
    # Initialize the custom environment
    env = CustomEnv(render=False, max_steps=1000)

    # Run the environment for a single episode
    observation, info = env.reset()
    total_reward = 0

    for step in range(1000):
        # Sample a random action from the action space
        action = env.action_space.sample()

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Accumulate reward
        total_reward += reward

        # Print the step details
        print(f"Step {step + 1}: Action = {action}, Observation = {observation}, Reward = {reward}")

        # Check if the episode is over
        if terminated or truncated:
            print("Episode ended.")
            break

    print(f"Total Reward: {total_reward}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
