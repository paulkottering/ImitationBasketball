import numpy as np
import gym
from gym import spaces


class BballScape1(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        # Define action and observation space
        # They must be gym.spaces objects
        self.observation_space = spaces.Box(low=np.zeros(22, ),
                                            high=np.ones(22, ),
                                            shape=(22,),
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=np.ones(10, ) / -5,
                                       high=np.ones(10, ) / 5,
                                       shape=(10,),
                                       dtype=np.float32)
        self.state = np.random.rand(22, )

    def step(self, action):
        # Execute one time step within the environment
        current_state = self.state
        for i in range(10):
            current_state[i + 10] = current_state[i + 10] + action[i]

        attack_move = (np.random.rand(10, ) - 1) / 10
        ball_move = (np.random.rand(2, ) - 1) / 10

        clipped_next_state = np.clip((current_state[:10] + attack_move), 0, 1)
        clipped_next_ball = np.clip((current_state[20:] + ball_move), 0, 1)

        current_state[:10] = clipped_next_state
        current_state[20:] = clipped_next_ball

        reward = np.sqrt((current_state[20] - 0.5) ** 2
                         + (current_state[21]) ** 2)
        self.state = np.array(current_state)

        return np.array(self.state, dtype=np.float32), reward, False, {}

    def reset(self):
        # Reset the state of the environment to an initial random state
        self.state = np.clip(np.random.rand(22, ), 0, 1)
        return self.state

    def render(self, mode='human'):
        # render the environment to the screen
        return self.state


class BballScape2(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        # Define action and observation space
        # They must be gym.spaces objects

        lowbound = np.append(np.append(np.zeros(10, ), np.ones(10, ) / -5), np.zeros(12, ))
        upbound = np.append(np.append(np.ones(10, ), np.ones(10, ) / 5), np.ones(12, ))

        self.observation_space = spaces.Box(low=lowbound,
                                            high=upbound,
                                            shape=(32,),
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=np.ones(10, ) / -5,
                                       high=np.ones(10, ) / 5,
                                       shape=(10,),
                                       dtype=np.float32)
        self.state = np.random.rand(32, )

    def step(self, action):
        # Execute one time step within the environment
        current_state = self.state
        for i in range(10):
            current_state[i + 20] = current_state[i + 20] + action[i]

        attack_move = (np.random.rand(20, ) - 1) / 10
        ball_move = (np.random.rand(2, ) - 1) / 10

        clipped_next_state = np.clip((current_state[:20] + attack_move), 0, 1)
        clipped_next_ball = np.clip((current_state[30:] + ball_move), 0, 1)

        current_state[:20] = clipped_next_state
        current_state[30:] = clipped_next_ball

        reward = np.sqrt((current_state[30] - 0.5) ** 2
                         + (current_state[31]) ** 2)
        self.state = np.array(current_state)

        return np.array(self.state, dtype=np.float32), reward, False, {}

    def reset(self):
        # Reset the state of the environment to an initial random state
        self.state = np.clip(
            np.append(np.append(np.random.rand(10, ), np.random.rand(10, ) / 10), np.random.rand(12, )), 0, 1)
        return self.state

    def render(self, mode='human'):
        # render the environment to the screen
        return self.state


class BballScape3(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        # Define action and observation space
        # They must be gym.spaces objects

        self.observation_space = spaces.Box(low=np.zeros(12, ),
                                            high=np.ones(12,),
                                            shape=(12,),
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=np.zeros(10, ),
                                            high=np.ones(10, ),
                                            shape=(10,),
                                            dtype=np.float32)

        self.state = np.random.rand(12, )

    def _step(self, action):

        random_move = (np.random.rand(12,)-0.5)/10

        self.state = np.array(self.state + random_move,dtype=np.float32)
        self.state = np.clip(self.state,0,1)

        reward = np.sqrt((self.state[10] - 0.5) ** 2
                         + (self.state[11]) ** 2)

        return self.state, reward, False, {}

    def _reset(self):
        # Reset the state of the environment to an initial random state
        self.state = (np.random.rand(12,), 0, 1)
        return self.state

    def _render(self, mode='human'):
        # render the environment to the screen
        return self.state

