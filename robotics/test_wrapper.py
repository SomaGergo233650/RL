import gymnasium as gym
import numpy as np
from ot2_gym_wrapper import CustomEnv

def run_episode():
    """Run a single episode in the custom environment."""
    # Create the environment instance
    environment = CustomEnv(render=False, max_steps=1000)

    # Initialize the environment
    state, metadata = environment.reset()
    cumulative_reward = 0

    for step_count in range(1000):
        # Select an action randomly from the action space
        chosen_action = environment.action_space.sample()

        # Execute the action and observe the result
        next_state, step_reward, done, truncated, additional_info = environment.step(chosen_action)

        # Update the total reward
        cumulative_reward += step_reward

        # Log the details of the current step
        print(f"Step {step_count + 1}: Action Taken = {chosen_action}, Next State = {next_state}, Reward Gained = {step_reward}")

        # Exit the loop if the episode is completed
        if done or truncated:
            print("The episode has concluded.")
            break

    print(f"Cumulative Reward for the Episode: {cumulative_reward}")

    # Release resources used by the environment
    environment.close()

def main():
    """Main function to execute the episode."""
    run_episode()

if __name__ == "__main__":
    main()
