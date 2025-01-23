from env_wrapper_10 import OT2Env
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
import argparse
from clearml import Task
import wandb
import os
from datetime import datetime
import typing_extensions
import tensorboard

# Set WandB API key and initialize the project
os.environ["WANDB_API_KEY"] = "15e8594f3ad74a144d38b801c5d29665e723e549"  # Replace with your actual key
wandb.login()
wandb.init(project="sb3_custom_env", sync_tensorboard=True)

# Training timesteps
timesteps = 5000000

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.98)
parser.add_argument("--policy", type=str, default="MlpPolicy")
parser.add_argument("--clip_range", type=float, default=0.15)
parser.add_argument("--value_coefficient", type=float, default=0.5)
args = parser.parse_args()

# Initialize environment
env = OT2Env(render=False, max_steps=args.n_steps)
check_env(env)  # Validate the custom environment

# Initialize ClearML task for remote training
task = Task.init(project_name='Mentor Group M/Group 2', task_name='new_hope')
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# Create PPO model
save_path = f"models/{wandb.run.id}"
os.makedirs(save_path, exist_ok=True)

model = PPO(
    args.policy,
    env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    clip_range=args.clip_range,
    vf_coef=args.value_coefficient,
    tensorboard_log=f"runs/{wandb.run.id}"
)

# WandB callback
wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=save_path,
    verbose=2,
)

# Train the model
model.learn(
    total_timesteps=timesteps,
    callback=wandb_callback,
    reset_num_timesteps=False,
    tb_log_name=f"runs/{wandb.run.id}"
)

# Save the model
model.save(f"{save_path}/{timesteps}_baseline")
wandb.save(f"{save_path}/{timesteps}_baseline")
