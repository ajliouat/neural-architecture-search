import gym
from gym import spaces

class SearchEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.dataset = config["dataset"]
        self.action_space = spaces.Discrete(10)  # Example action space
        self.observation_space = spaces.Box(low=0, high=1, shape=(32, 32, 3))

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0, False, {}