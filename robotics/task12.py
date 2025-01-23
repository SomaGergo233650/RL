import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class SimplePIDController:
    def __init__(self, kp, ki, kd, target=0.0):
        """
        A basic PID controller implementation.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.error_sum = 0.0
        self.last_error = 0.0

    def compute(self, current_value):
        """
        Compute the control signal based on the error.
        """
        error = self.target - current_value
        self.error_sum += error
        delta_error = error - self.last_error
        self.last_error = error

        return self.kp * error + self.ki * self.error_sum + self.kd * delta_error

class RobotEnv(gym.Env):
    def __init__(self, render=False, max_steps=10000):
        super().__init__()
        self.render_enabled = render
        self.max_steps = max_steps

        # Initialize the simulation
        self.sim = Simulation(num_agents=1, render=self.render_enabled)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, 0]), high=np.array([1, 1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # PID controllers for each axis
        self.pid_x = SimplePIDController(kp=1.0, ki=0.1, kd=0.01)
        self.pid_y = SimplePIDController(kp=1.0, ki=0.1, kd=0.01)
        self.pid_z = SimplePIDController(kp=1.0, ki=0.1, kd=0.01)

        self.steps = 0
        self.goal_position = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Set a random goal position
        self.goal_position = np.random.uniform(
            low=[-0.18700, -0.17050, 0.16940],
            high=[0.25300, 0.21950, 0.28950],
            size=(3,)
        ).astype(np.float32)

        # Update PID targets
        self.pid_x.target = self.goal_position[0]
        self.pid_y.target = self.goal_position[1]
        self.pid_z.target = self.goal_position[2]

        # Reset the simulation
        observation = self.sim.reset(num_agents=1)
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)

        # Create observation combining pipette and goal positions
        observation = np.concatenate([pipette_position, self.goal_position])

        self.steps = 0
        return observation, {}

    def step(self, action):
        # Get the pipette's current position
        observation = self.sim.run([action])
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)

        # Compute control signals using PID
        control_x = self.pid_x.compute(pipette_position[0])
        control_y = self.pid_y.compute(pipette_position[1])
        control_z = self.pid_z.compute(pipette_position[2])

        # Formulate the action
        pid_action = np.array([control_x, control_y, control_z, 0.0], dtype=np.float32)

        # Simulate the PID action
        self.sim.run([pid_action])

        # Create a new observation
        observation = np.concatenate([pipette_position, self.goal_position])

        # Calculate distance and reward
        distance = np.linalg.norm(pipette_position - self.goal_position)
        reward = -distance

        if distance <= 0.001:
            reward += 100  # Reward for reaching the goal

        # Determine if the episode is finished
        terminated = distance <= 0.001
        truncated = self.steps >= self.max_steps

        # Debugging information
        print(f"Step {self.steps}:")
        print(f"  Current Position: {pipette_position}")
        print(f"  Goal Position: {self.goal_position}")
        print(f"  Distance to Goal: {distance * 1000:.2f} mm")

        self.steps += 1
        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_enabled:
            self.sim.render()

    def close(self):
        self.sim.close()

if __name__ == "__main__":
    env = RobotEnv(render=True)
    observation, _ = env.reset()

    print(f"Goal Position: {env.goal_position}")

    for _ in range(10000):
        action = env.action_space.sample()
        observation, reward, terminated, is_truncated, _ = env.step(action)

        if terminated or is_truncated:
            print("Goal Achieved!" if terminated else "Simulation Truncated.")
            break

    env.close()
