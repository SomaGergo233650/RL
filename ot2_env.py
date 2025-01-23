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
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32),  # Observation space can be larger to accommodate the full 3D space
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.steps = 0
        self.goal_position = np.array([0.5, 0.5, 0.2])

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulation
        observation = self.sim.reset(num_agents=1)
        robot_id = next(iter(observation))  
        self.state = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)

        # Randomize goal position within work envelope
        self.goal_position = np.random.uniform([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])

        # Append goal position to the state
        observation = np.concatenate([self.state, self.goal_position])

        self.steps = 0

        return observation, {}



    def step(self, action):
        # Add "drop action" as needed
        action = np.append(action, 0)
        observation = self.sim.run([action])

        robot_id = next(iter(observation))  
        self.state = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)
        observation = np.concatenate([self.state, self.goal_position])

        # Compute reward
        reward = self._compute_reward()

        # Termination criteria
        distance_to_goal = np.linalg.norm(self.state[:3] - self.goal_position)
        terminated = distance_to_goal < 0.001  # Terminate if within 1 mm precision
        truncated = self.steps >= self.max_steps
        info = {"distance_to_goal": distance_to_goal}

        self.steps += 1

        return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        # Check if the render flag is True before attempting to render
        if self.render:
            self.sim.render(mode)  # Assuming the Simulation class has a render method

    def close(self):
        self.sim.close()

    def _compute_reward(self):
        pipette_position = np.array(self.state[:3])
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)

        # Rewards for accuracy
        if distance_to_goal < 0.001:  # 1 mm
            reward = 100  # Highly precise bonus
        elif distance_to_goal < 0.01:  # 10 mm
            reward = 50  # Medium precision bonus
        else:
            reward = -distance_to_goal  # Penalize based on distance

        # Penalty for steps
        step_penalty = -0.01

        # Strong penalty for leaving the workspace (work envelope)
        out_of_bounds_penalty = -5.0 if not np.all((pipette_position >= -1) & (pipette_position <= 1)) else 0.0

        # Combine rewards and penalties
        return reward + step_penalty + out_of_bounds_penalty

