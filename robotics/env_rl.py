import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class CustomEnv(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(CustomEnv, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define goal position ranges
        self.x_min, self.x_max = -0.187, 0.2531
        self.y_min, self.y_max = -0.1705, 0.2195
        self.z_min, self.z_max = 0.1195, 0.2895

        # Add a small margin to handle precision issues
        margin = 1e-4

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_min - margin, self.y_min - margin, self.z_min - margin, 
                          -self.x_max - margin, -self.y_max - margin, -self.z_max - margin], dtype=np.float32),
            high=np.array([self.x_max + margin, self.y_max + margin, self.z_max + margin, 
                           self.x_max + margin, self.y_max + margin, self.z_max + margin], dtype=np.float32),
            dtype=np.float32
        )
        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Randomize goal position
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        z = np.random.uniform(self.z_min, self.z_max)
        self.goal_pos = np.array([x, y, z], dtype=np.float32)

        # Reset simulation and get pipette position
        observation = self.sim.reset(num_agents=1)
        pipette_pos = self.sim.get_pipette_position(self.sim.robotIds[0])

        # Combine pipette position and goal position into the observation
        observation = np.concatenate((pipette_pos, self.goal_pos), axis=0).astype(np.float32)

        # Clamp observation to fit within bounds
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)

        # Debugging logs
        print(f"Reset Observation: {observation}")
        print(f"Bounds: Low = {self.observation_space.low}, High = {self.observation_space.high}")

        self.steps = 0
        return observation, {}

    def step(self, action):
        self.steps += 1
        action = np.append(action, 0)  # Append zero for compatibility with the simulation

        # Get simulation result
        observation = self.sim.run([action])
        pipette_pos = self.sim.get_pipette_position(self.sim.robotIds[0])

        # Combine pipette position and goal position into the observation
        observation = np.concatenate((pipette_pos, self.goal_pos), axis=0).astype(np.float32)

        # Clamp observation to fit within bounds
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)

        # Debugging logs
        print(f"Step Observation: {observation}")
        print(f"Bounds: Low = {self.observation_space.low}, High = {self.observation_space.high}")

        # Calculate distance to goal
        distance = np.linalg.norm(observation[:3] - observation[3:])

        # Reward function
        reward = -distance  # Penalize based on distance
        reward += 10 / (distance + 1e-6)  # Strong reward for proximity to the goal

        # Velocity penalty: penalize large changes in actions
        action_penalty = np.linalg.norm(action) * 0.01
        reward -= action_penalty

        # Bonus for being very close to the goal
        if distance < 0.05:  # Close to the goal
            reward += 20
        if distance <= 0.01:  # Termination distance
            reward += 100  # Large bonus for reaching the goal

        # Penalize longer episodes
        reward -= 0.01 * self.steps

        # Check if the episode is terminated
        terminated = distance <= 0.01  # Goal is reached
        truncated = self.steps >= self.max_steps  # Maximum steps reached

        return observation, reward, terminated, truncated, {}


    def render(self, mode='human'):
        # Optional rendering function
        pass

    def close(self):
        if hasattr(self, "physicsClient") and self.physicsClient >= 0:
            p.disconnect(self.physicsClient)

