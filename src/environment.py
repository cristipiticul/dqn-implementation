import retro
import gym
import numpy as np


def create_env():
    env = retro.make(game="BeamRider-Atari2600")
    env = RewardWrapper(env)
    env = FrameSkip(env)
    env = gym.wrappers.GrayScaleObservation(env)  # RGB -> gray
    env = CropObservationWrapper(env, 40, 200, 0, 160)
    env = gym.wrappers.ResizeObservation(env, 84)  # Rescale to 84x84
    env = RemoveDimensionsOfSize1Wrapper(env)
    env = gym.wrappers.FrameStack(env, 4)  # Keep history of 4 frames
    # gym.wrappers.FrameStack returns LazyFrames -- convert to numpy
    env = ConvertLazyFramesToNumpy(env)
    return env


class FrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, frame_skip=4):
        super().__init__(env)
        self.frame_skip = frame_skip

    def step(self, action):
        R = 0.0
        observation, reward, done, info = None, None, None, None
        for t in range(self.frame_skip):
            observation, reward, done, info = self.env.step(action)
            R += reward
            if done:
                break
        return observation, R, done, info


class RewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.prev_lives_remaining = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        lives_remaining = info["terminal"]
        if self.prev_lives_remaining is None:
            self.prev_lives_remaining = lives_remaining
        died = lives_remaining < self.prev_lives_remaining
        if died:
            # Bug: sometimes when dying (eg. lives 2-->1), the life comes back for a small while, then goes away again
            self.prev_lives_remaining = lives_remaining
            reward = -1.0
        if reward > 0:
            reward = +1.0
        return observation, reward, done, info

    def reset(self):
        self.prev_lives_remaining = None
        return self.env.reset()


class CropObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, top, bottom, left, right):
        super().__init__(env)
        self.top, self.bottom, self.left, self.right = top, bottom, left, right
        # TODO self.observation_space = ... (crop also the obs space)

    def observation(self, observation):
        return observation[self.top : self.bottom, self.left : self.right]


class RemoveDimensionsOfSize1Wrapper(gym.ObservationWrapper):
    def observation(self, observation):
        assert (
            1 in observation.shape
        ), f"There is no dimension of size 1: {observation.shape}! This wrapper is useless"
        return observation.squeeze()


class ConvertLazyFramesToNumpy(gym.ObservationWrapper):
    def observation(self, observation):
        return np.asarray(observation)
