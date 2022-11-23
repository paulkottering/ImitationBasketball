import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time
import gym
from gym import spaces

class BballScape(gym.Env):

  """Custom Environment that follows gym interface"""

  def __init__(self, arg1, arg2, ...):
    super(BballScape, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    self.observation_space = spaces.Box( low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                                         high=np.array([50,100,50,100,50,100,50,100,50,100,50,100,50,100,50,100,50,100,50,100,50,100]),
                                         dtype=np.float16)

    self.action_space = spaces.Box( low=np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]),
                                         high=np.array([1,1,1,1,1,1,1,1,1,1]),
                                         dtype=np.float16)
    self.state = np.random.rand(22)*50

  def step(self, action):
    # Execute one time step within the environment
    for i in range(10):
        self.state[i+12] = self.state[i+12] + action[i]

    return self.state, 0, False, {}


  def reset(self):
    # Reset the state of the environment to an initial random state
    return np.random.rand(22)*50
