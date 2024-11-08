# Drone Reinforcement Learning with PyBullet and Stable-Baselines3

This project implements a reinforcement learning environment for controlling a simulated drone in 3D space. Using the `gym-pybullet-drones` library and the `Stable-Baselines3` library, the project focuses on training a drone to hover using the Proximal Policy Optimization (PPO) algorithm.

## Repository Structure

```
drone_rl
├── gym-pybullet-drones         # Custom environment files for drone simulation
├── rl_scripts                  # Reinforcement learning scripts and logs
│   ├── logs                    # Training logs for TensorBoard visualization
│   └── test_drone_env.py       # Main script for training and evaluation
├── LICENSE                     # License file
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

## Requirements

To get started, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Training the Drone

1. **Setup**: Make sure you have all dependencies installed and `gym-pybullet-drones` configured.
2. **Run Training**: To start training the drone with PPO, run the main training script:

   ```bash
   python rl_scripts/test_drone_env.py
   ```

3. **Logging**: During training, logs are saved in `rl_scripts/logs`, allowing you to monitor progress with TensorBoard.

## Monitoring Training with TensorBoard

1. **Start TensorBoard**:

   ```bash
   tensorboard --logdir=rl_scripts/logs
   ```

2. **Access TensorBoard**: Open [http://localhost:6006](http://localhost:6006) in your browser to visualize training metrics such as reward, policy loss, and value loss.

## Using a Pretrained Model

The repository includes code to load and evaluate a trained model. After training completes, the final model is saved in `rl_scripts/logs` as `final_ppo_drone_model.zip`.

To evaluate a saved model, modify the `test_drone_env.py` script to load `final_ppo_drone_model.zip` and visualize the drone's performance:

```python
from stable_baselines3 import PPO

# Load the model
model = PPO.load("rl_scripts/logs/final_ppo_drone_model")

# Run the environment and see how the trained model performs
obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
```

## Repository Contents

### `gym-pybullet-drones`

Contains the customized environment and configurations for the drone simulation using PyBullet.

### `rl_scripts/test_drone_env.py`

The main script for training the drone with PPO. This script includes model training, evaluation, and logging.

### `rl_scripts/logs`

Contains training logs for monitoring with TensorBoard and saved model checkpoints. **Note**: This folder will initially be empty.

## Requirements

### `requirements.txt`

Contains the following dependencies:
- `gymnasium`
- `stable-baselines3`
- `tensorboard`
- `pybullet`
- `gym-pybullet-drones`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

