import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import os

# Create the environment with monitoring to log rewards and steps
env = HoverAviary(gui=True)
env = Monitor(env)  # Add monitor for logging

# Define where to save models and logs
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

# Define a callback to save models periodically
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="ppo_drone")

# Define an evaluation callback to monitor performance
eval_env = HoverAviary(gui=False)
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=10000)

# Create the PPO agent
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

# Train the agent and use the callback for saving and evaluating the model
model.learn(total_timesteps=50000, callback=[checkpoint_callback, eval_callback])

# Save the final trained model
model.save(os.path.join(log_dir, "final_ppo_drone_model"))

# Load and evaluate the trained model
model = PPO.load(os.path.join(log_dir, "final_ppo_drone_model"))

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs = env.reset()

env.close()
