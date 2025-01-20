import wandb
import os
import argparse
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from clearml import Task
from env_V3_final_2 import CustomEnv  # Assuming CustomEnv is in this file
from datetime import datetime
import typing_extensions
import tensorboard

# Initialize command-line argument parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)  # Updated learning rate
parser.add_argument("--batch_size", type=int, default=128)  # Updated batch size
parser.add_argument("--n_steps", type=int, default=4096)  # Increased n_steps
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--episodes", type=int, default=50)  # Increased episodes
parser.add_argument("--iterations_per_episode", type=int, default=10000)  # Reduced iterations per episode
parser.add_argument("--gamma", type=float, default=0.995)  # Updated gamma
parser.add_argument("--ent_coef", type=float, default=0.05)  # Increased entropy coefficient
parser.add_argument("--clip_range", type=float, default=0.2)  # Updated clip_range
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--policy", type=str, default="MlpPolicy")
args = parser.parse_args()

# Set WandB API key and initialize the project
os.environ["WANDB_API_KEY"] = "15e8594f3ad74a144d38b801c5d29665e723e549"
wandb.login()
wandb.init(project="sb3_custom_env", sync_tensorboard=True)

# Set up the environment
env = CustomEnv(render=False, max_steps=args.n_steps)

# Initialize ClearML task for remote training setup
task = Task.init(project_name='Mentor Group M/Group 2', task_name='hope')
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# Set up PPO model with the custom environment
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
    ent_coef=args.ent_coef,
    clip_range=args.clip_range,
    vf_coef=args.vf_coef,
    tensorboard_log=f"./runs/{wandb.run.id}/tensorboard/"
)

# Training loop
for episode in range(1, args.episodes + 1):
    print(f"Starting episode {episode}/{args.episodes}")

    total_timesteps = args.iterations_per_episode * args.n_steps

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        tb_log_name=f"PPO_run_{wandb.run.id}_episode_{episode}",
        callback=WandbCallback(
            model_save_freq=10000,
            model_save_path=save_path,
            verbose=2
        )
    )

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model.save(f"{save_path}/ppo_model_episode_{episode}_{timestamp}")
    wandb.save(f"{save_path}/ppo_model_episode_{episode}_{timestamp}.zip")

wandb.finish()
