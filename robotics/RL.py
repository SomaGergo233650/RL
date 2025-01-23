import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
from stable_baselines3 import PPO
import time

class CustomEnv(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(CustomEnv, self).__init__()
        self.render_enabled = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render_enabled)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, 0]), high=np.array([1, 1, 1, 1]), shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Track steps and goal position
        self.steps = 0
        self.goal_position = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Set a random goal position within the valid range
        self.goal_position = np.array([
            np.random.uniform(-0.18700, 0.25300),
            np.random.uniform(-0.17050, 0.21950),
            np.random.uniform(0.16940, 0.28950)
        ], dtype=np.float32)

        # Reset the simulation environment
        observation = self.sim.reset(num_agents=1)

        # Extract pipette position and combine with goal position
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position], axis=0)

        self.steps = 0
        self.prev_distance = np.linalg.norm(pipette_position - self.goal_position)  # Initialize prev_distance
        return observation, {}


    def step(self, action):
        # Extract current pipette position from the environment
        observation = self.sim.run([action])
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)

        # Update observation
        observation = np.concatenate([pipette_position, self.goal_position], axis=0)

        # Calculate distance to the goal
        distance = np.linalg.norm(pipette_position - self.goal_position)

        # Calculate reward based on distance to the goal
        reward = -distance  # Base reward
        if distance <= 0.001:  # If the agent reaches the goal
            reward += 100  # Large positive reward for achieving the goal
        elif self.steps > 0 and self.prev_distance > distance:  # If the agent is moving closer to the goal
            reward += 10  # Additional reward for getting closer
        else:  # If the agent is moving away from the goal
            reward -= 5  # Penalty for moving further away

        # Update the previous distance
        self.prev_distance = distance

        # Check termination condition (10 mm accuracy requirement)
        terminated = distance <= 0.001  # 10 mm accuracy

        # Check truncation
        truncated = self.steps >= self.max_steps

        self.steps += 1
        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_enabled:
            self.sim.render()

    def close(self):
        self.sim.close()

def benchmark(goal_position=None, max_steps=2000, position_change_threshold=1e-3, max_stuck_steps=2000):
    """
    Run the benchmark for the RL controller and detect if the position doesn't change for max_stuck_steps.

    :param goal_position: The goal position to reach.
    :param max_steps: Maximum steps for the benchmark.
    :param position_change_threshold: Threshold to detect when position has not changed.
    :param max_stuck_steps: Maximum number of consecutive steps where the position doesn't change.
    :return: Dictionary with benchmark results.
    """
    env = CustomEnv(render=True)
    model = PPO.load("model.zip")  # Load the trained PPO model

    # If no goal position is provided, generate a random one
    if goal_position is None:
        goal_position = np.array([
            np.random.uniform(-0.18700, 0.25300),
            np.random.uniform(-0.17050, 0.21950),
            np.random.uniform(0.16940, 0.28950)
        ], dtype=np.float32)

    print(f"Random Goal Position: {goal_position}")

    # Reset the environment with the goal position
    observation, _ = env.reset()
    goal_position = observation[3:]  # The goal position is in the last 3 values of the observation

    # Initialize tracking variables
    rewards = []
    distances = []
    steps = []
    start_time = time.time()

    prev_position = observation[:3]
    stuck_steps = 0

    for step in range(max_steps):  # Simulate up to max_steps
        # Predict the action from the trained model
        action, _ = model.predict(observation, deterministic=True)  # Use deterministic=True for testing
        observation, reward, terminated, truncated, _ = env.step(action)

        # Calculate the accuracy (distance to the goal) in mm
        current_position = observation[:3]  # Extract the current position from the observation
        distance_to_goal = np.linalg.norm(goal_position - current_position) * 1000  # Convert to mm

        # Store the metrics for benchmarking
        rewards.append(reward)
        distances.append(distance_to_goal)
        steps.append(step + 1)  # Store the step count (1-based indexing)

        # Print debug information for every 100 steps
        if step % 100 == 0 or step == max_steps - 1:
            print(f"Step {step + 1}/{max_steps}:")
            print(f"  Current Position: {current_position}")
            print(f"  Goal Position: {goal_position}")
            print(f"  Distance to Goal: {distance_to_goal:.2f} mm")
            print(f"  Accumulated Reward: {sum(rewards):.4f}")

        # Check if position has changed significantly
        if np.linalg.norm(current_position - prev_position) < position_change_threshold:
            stuck_steps += 1
        else:
            stuck_steps = 0

        # If the position hasn't changed for max_stuck_steps, consider it as truncated
        if stuck_steps >= max_stuck_steps:
            print("Position hasn't changed significantly for too long. Continuing...")
            stuck_steps = 0  # Reset stuck_steps to allow continuation

        # Update previous position for the next step
        prev_position = current_position

    # Calculate the total time taken to reach the goal
    total_time = time.time() - start_time

    # Collect results
    results = {
        'total_steps': len(steps),
        'total_rewards': sum(rewards),
        'distance_per_step': distances,
        'total_time': total_time,
        'goal_position': goal_position,
        'final_position': current_position,
    }

    env.close()

    return results

# Testing Process
if __name__ == "__main__":
    results = benchmark(max_steps=2000)

    # Display results
    print("\nBenchmark Results:")
    for key, value in results.items():
        if isinstance(value, list):
            print(f"{key}: {value[:10]} ...")  # Display first 10 values for long lists
        else:
            print(f"{key}: {value}")
