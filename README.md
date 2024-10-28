

```markdown
# Drone Reinforcement Learning Project

This project implements reinforcement learning (RL) to train a simulated drone to hover at a target position using [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). The model is trained using the PPO (Proximal Policy Optimization) algorithm with TensorBoard for logging and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Results and Visualization](#results-and-visualization)
- [Future Improvements](#future-improvements)

## Project Overview
The main goal of this project is to train a drone to hover at a target location in a simulated environment using reinforcement learning. The environment includes obstacles, and the model is trained to maintain a stable hover position while avoiding any obstacles.

Key elements:
- **Environment:** HoverAviary, a custom gym environment for drone control.
- **Agent:** PPO agent from Stable-Baselines3.
- **Logging:** Training progress and evaluation metrics are logged using TensorBoard.

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed. The project dependencies include:
- `gymnasium`
- `pybullet`
- `stable-baselines3`
- `tensorboard`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/ranga04/drone_rl.git
   cd drone_rl
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv drone_rl_env
   source drone_rl_env/bin/activate  # On Windows, use drone_rl_env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```plaintext
drone_rl/
├── rl_scripts/
│   ├── test_drone_env.py       # Script to train and test the drone environment
│   ├── run_saved_model.py      # Script to evaluate the saved model
│   ├── logs/                   # Contains TensorBoard logs and checkpoints
│   └── README.md               # This README file
├── gym-pybullet-drones/        # The drone simulation environment
└── requirements.txt            # Python dependencies
```

## Training the Model

To train the PPO agent:
1. Open the `test_drone_env.py` script and adjust `total_timesteps` if needed.
2. Run the script:
   ```bash
   python3 rl_scripts/test_drone_env.py
   ```

This will save checkpoints and logs in the `logs/` directory, including TensorBoard logs.

## Evaluating the Model

After training, you can evaluate the saved model by running:
```bash
python3 rl_scripts/run_saved_model.py
```

The evaluation script loads the saved model and lets the agent interact with the environment to demonstrate the learned behavior.

## Results and Visualization

To visualize the training progress using TensorBoard:
```bash
tensorboard --logdir=./rl_scripts/logs
```

Open [http://localhost:6006/](http://localhost:6006/) in your browser to see metrics like:
- **Value Loss**
- **Policy Gradient Loss**
- **Entropy Loss**
- **Average Episode Reward**

## Future Improvements
- **Complexity**: Add dynamic obstacles or weather conditions to make the environment more challenging.
- **Hyperparameter Tuning**: Experiment with different PPO parameters.
- **Different RL Algorithms**: Test alternative algorithms like SAC or DDPG for continuous action space control.

## Acknowledgments
This project uses [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) by the University of Toronto Institute for Aerospace Studies and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
