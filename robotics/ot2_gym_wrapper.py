import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation  # Assuming your custom Simulation class

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=10000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        
        # Create the simulation environment, passing the render flag
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),  # x, y, z velocities
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),   # x, y, z velocities
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.steps = 0

        # Define the cube boundaries based on provided coordinates
        self.bounds = {
            "x_min": -0.18700662642653432,
            "x_max": 0.2530890387697392,
            "y_min": -0.170609364469082,
            "y_max": 0.21950138703355168,
            "z_min": 0.16949820263744853,
            "z_max": 0.28952088202505344,
        }

        # Set an initial goal position
        self.goal_position = self._randomize_goal_position()

    def _randomize_goal_position(self):
        # Randomly generate a goal position within the defined boundaries
        return np.array([
            np.random.uniform(self.bounds["x_min"], self.bounds["x_max"]),
            np.random.uniform(self.bounds["y_min"], self.bounds["y_max"]),
            np.random.uniform(self.bounds["z_min"], self.bounds["z_max"]),
        ], dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation environment
        observation = self.sim.reset(num_agents=1)

        # Check if the output is valid
        if not isinstance(observation, dict) or len(observation) == 0:
            raise ValueError("Unexpected output from self.sim.reset():", observation)

        robot_id = next(iter(observation))
        self.state = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)

        # Randomize the goal position each time the environment is reset
        self.goal_position = self._randomize_goal_position()

        # Append goal position to the state
        observation = np.concatenate([self.state, self.goal_position])

        self.steps = 0

        # Return observation and an empty info dictionary
        return observation, {}

    def step(self, action):
        # Proceed with the usual step functionality
        action = np.append(action, 0)  # Adding 0 for drop action
        observation = self.sim.run([action])

        robot_id = next(iter(observation))  
        self.state = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)
        
        observation = np.concatenate([self.state, self.goal_position])

        # Compute reward based on the current position
        reward = self._compute_reward()

        # Goal detection: Terminate when within a threshold distance
        distance_to_goal = np.linalg.norm(self.state[:3] - self.goal_position)
        terminated = distance_to_goal < 0.02 

        truncated = self.steps >= self.max_steps
        info = {}

        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        if self.render:
            self.sim.render(mode)

    def close(self):
        self.sim.close()

    def _compute_reward(self):
        pipette_position = np.array(self.state[:3])
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        
        # More linear reward
        progress_reward = 1 - (distance_to_goal / np.linalg.norm([
            self.bounds["x_max"] - self.bounds["x_min"],
            self.bounds["y_max"] - self.bounds["y_min"],
            self.bounds["z_max"] - self.bounds["z_min"],
        ]))
        
        # Less harsh step penalty
        step_penalty = -0.005
        
        # Bonus for getting very close to the goal
        goal_proximity_bonus = 1.0 if distance_to_goal < 0.05 else 0.0
        
        return progress_reward + step_penalty + goal_proximity_bonus
