import gymnasium as gym
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import os

# Create the environment
env = HoverAviary(gui=True)  # Set gui=True if you want to visualize the drone

# Define the directory where the model is saved
log_dir = "./logs/"
model_path = os.path.join(log_dir, "final_ppo_drone_model.zip")

# Load the trained model
model = PPO.load(model_path)

# Reset the environment and run the saved model
obs = env.reset()
for i in range(1000):  # Run for 1000 steps (adjust as needed)
    action, _states = model.predict(obs)  # Predict actions using the loaded model
    obs, reward, done, truncated, info = env.step(action)  # Apply actions
    env.render()  # Visualize the environment
    if done or truncated:  # Reset environment if episode is done
        obs = env.reset()

# Close the environment after running
env.close()
