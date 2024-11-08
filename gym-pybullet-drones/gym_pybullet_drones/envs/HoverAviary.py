import numpy as np
import pybullet as p  # Import PyBullet

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment."""

        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8
        self.obstacle_id = None  # Initialize the obstacle ID here
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        # Add obstacles to the environment
        self.add_obstacles()

    ################################################################################
    
    def add_obstacles(self):
        """Add obstacles to the environment."""
        print("Adding obstacles to the environment...")
        obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.5])
        obstacle_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.5], rgbaColor=[1, 0, 0, 1])
        self.obstacle_id = p.createMultiBody(baseMass=0,  # Mass 0 means it's static
                                             baseCollisionShapeIndex=obstacle_shape,
                                             baseVisualShapeIndex=obstacle_visual,
                                             basePosition=[0, 0.5, 0.5])  # Position of the obstacle
        print(f"Obstacle created with ID: {self.obstacle_id}")

    ################################################################################

    def reset(self, **kwargs):
        """Reset the environment and ensure the obstacle remains."""
        obs = super().reset()  # Call the parent class's reset method
        print("Environment reset called.")
        
        # Check if the obstacle still exists after reset
        try:
            if p.getBodyInfo(self.obstacle_id) is not None:
                # Reset the obstacle's position
                p.resetBasePositionAndOrientation(self.obstacle_id, [0, 0.5, 0.5], [0, 0, 0, 1])
                print(f"Obstacle with ID {self.obstacle_id} reset.")
            else:
                raise p.error  # If obstacle body info is invalid, recreate it
        except p.error:
            print("Obstacle was missing or invalid, recreating it.")
            self.add_obstacles()  # Add obstacle if missing or invalid

        return obs


    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value."""
        state = self._getDroneStateVector(0)
        distance = np.linalg.norm(self.TARGET_POS - state[0:3])
        reward = max(0, 2 - distance**4)  # Higher reward closer to target
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value."""
        state = self._getDroneStateVector(0)
        return np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.0001

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value."""
        state = self._getDroneStateVector(0)
        too_far = abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0
        too_tilted = abs(state[7]) > 0.4 or abs(state[8]) > 0.4
        time_limit = self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC
        return too_far or too_tilted or time_limit

    ################################################################################
    
    def _computeInfo(self):
        """Return additional information."""
        return {"info": "Optional debug info"}  # Customize if needed
