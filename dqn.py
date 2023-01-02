# Run with
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python dqn.py

# TODO:
# - fix replay memory getting out of memory (maybe use PyTables)
# - also save replay memory in checkpoints
# - try different discount factors (gamma)

import numpy as np
import retro
import gym
import gym.envs.registration
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import color
from skimage.transform import resize
from matplotlib import pyplot as plt
from collections import deque
import random
import os
import csv
import tables


class Net(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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


# class ReplayMemory:
#     """Store multiple (prev_obs, action, reward, obs, done) in separate numpy arrays."""

#     def __init__(self, max_len):
#         self.initialized = False
#         self.max_len = max_len
#         self.n_items = 0
#         self.curr_index = 0  # circular array - if max_len is reached, overwrite index 0

#         self.prev_obs_arr = np.zeros(1)  # placeholder
#         self.obs_arr = np.zeros(1)  # placeholder
#         self.action_arr = np.zeros(max_len, dtype=np.int8)
#         self.reward_arr = np.zeros(max_len, dtype=np.float32)
#         self.done_arr = np.zeros(max_len, dtype=np.int8)

#     def add(
#         self,
#         prev_obs: np.ndarray,
#         action: int,
#         reward: float,
#         obs: np.ndarray,
#         done: bool,
#     ):
#         if not self.initialized:
#             self.obs_arr = np.zeros((self.max_len, *obs.shape), dtype=prev_obs.dtype)
#             self.prev_obs_arr = np.zeros(
#                 (self.max_len, *prev_obs.shape), dtype=obs.dtype
#             )
#             self.initialized = True
#         self.prev_obs_arr[self.curr_index, :, :, :] = prev_obs
#         self.action_arr[self.curr_index] = action
#         self.reward_arr[self.curr_index] = reward
#         self.obs_arr[self.curr_index, :, :, :] = obs
#         self.done_arr[self.curr_index] = 1 if done else 0

#         self.curr_index += 1
#         self.curr_index %= self.max_len
#         if self.n_items < self.max_len:
#             self.n_items += 1

#     def get_minibatch(self, indices: list):
#         return [
#             np.take(arr, indices, axis=0)
#             for arr in [
#                 self.prev_obs_arr,
#                 self.action_arr,
#                 self.reward_arr,
#                 self.obs_arr,
#                 self.done_arr,
#             ]
#         ]

#     def __len__(self):
#         return self.n_items


class ReplayMemoryRowDescription(tables.IsDescription):
    prev_obs = tables.UInt8Col(shape=(4, 84, 84))
    action = tables.UInt8Col()
    reward = tables.Float32Col()
    obs = tables.UInt8Col(shape=(4, 84, 84))
    done = tables.BoolCol()


class ReplayMemory:
    """Store multiple (prev_obs, action, reward, obs, done) in separate numpy arrays."""

    def __init__(self, max_len):
        self.initialized = False
        self.max_len = max_len
        self.n_items = 0
        self.curr_index = 0  # circular array - if max_len is reached, overwrite index 0
        self.h5file = tables.open_file(
            "replay_memory.hdf5", mode="a", title="replay_memory"
        )
        self.h5file.create_table(
            self.h5file.root, "replay_memory_table", ReplayMemoryRowDescription
        )

    def add(
        self,
        prev_obs: np.ndarray,
        action: int,
        reward: float,
        obs: np.ndarray,
        done: bool,
    ):
        print(prev_obs.dtype, action, reward, obs.shape, done)
        pass

    def get_minibatch(self, indices: list):
        return None

    def __len__(self):
        return self.n_items


def replay_memory_test():
    print("Testing replay memory")
    r = ReplayMemory(max_len=4)
    prev_obs = np.zeros((4, 84, 84), dtype=np.float32)
    obs = np.zeros((4, 84, 84), dtype=np.float32)
    r.add(prev_obs, 0, 0.0, obs, False)
    r.add(prev_obs + 1, 1, 1.0, obs + 1, False)
    r.add(prev_obs + 2, 2, 2.0, obs + 2, True)
    mb = r.get_minibatch([0, 2])
    assert mb[0][0][0, 0, 0] == 0
    assert mb[1][0] == 0
    assert mb[2][0] == 0
    assert mb[3][0][0, 0, 0] == 0
    assert mb[4][0] == False
    assert mb[0][1][0, 0, 0] == 2
    assert mb[1][1] == 2
    assert mb[2][1] == 2
    assert mb[3][1][0, 0, 0] == 2
    assert mb[4][1] == True

    # Fill completely and overflow
    r.add(prev_obs + 3, 3, 3.0, obs + 3, True)
    r.add(prev_obs + 4, 4, 4.0, obs + 4, True)
    mb = r.get_minibatch([0, 2])
    assert mb[0][0][0, 0, 0] == 4
    assert mb[1][0] == 4
    assert mb[2][0] == 4
    assert mb[3][0][0, 0, 0] == 4
    assert mb[4][0] == True
    print("Done testing replay memory")


criterion = torch.nn.MSELoss()
gamma = 0.95  # TODO: choose a discount factor
running_loss = 0


def gradient_descent_step(
    prev_obs_minibatch: np.ndarray,
    action_minibatch: np.ndarray,
    rew_minibatch: np.ndarray,
    obs_minibatch: np.ndarray,
    done_minibatch: np.ndarray,
    optimizer: torch.optim.Optimizer,
    prev_net: nn.Module,
    trained_net: nn.Module,
):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    inp = torch.tensor(prev_obs_minibatch).double()
    outputs = trained_net(inp)

    # We only care about the predicted Q-values of actions we've chosen, select those
    actions_torch = torch.unsqueeze(
        torch.tensor(action_minibatch, dtype=torch.int64), dim=1
    )
    outputs_for_actions = torch.gather(outputs, 1, actions_torch).squeeze()

    with torch.no_grad():
        # Target = reward + gamma * max_a prev_net(s')
        rewards = torch.tensor(rew_minibatch)
        prev_net_outputs = prev_net(torch.tensor(obs_minibatch).double())
        # select maximum
        state_values = prev_net_outputs.max(dim=1).values
        # ignore state values for terminal states
        state_values = state_values * (1 - torch.tensor(done_minibatch))
        # add them
        target_outputs = rewards + gamma * state_values

    loss = criterion(outputs_for_actions, target_outputs)
    loss.backward()
    optimizer.step()

    global running_loss
    running_loss += loss.item()


def main():
    # replay_memory_test()
    env = retro.make(game="BeamRider-Atari2600")
    env = RewardWrapper(env)
    env = FrameSkip(env)
    env = gym.wrappers.GrayScaleObservation(env)  # RGB -> gray
    env = CropObservationWrapper(env, 40, 200, 0, 160)
    env = gym.wrappers.ResizeObservation(env, 84)  # Rescale to 84x84
    env = RemoveDimensionsOfSize1Wrapper(env)
    env = gym.wrappers.FrameStack(env, 4)  # Keep history of 4 frames

    num_actions = env.action_space.shape[0]

    # The parameters from the previous iteration are held fixed when optimising the loss function
    # ==> we need two sets of parameters
    prev_net = Net(num_actions).double()
    trained_net = Net(num_actions).double()
    optimizer = torch.optim.RMSprop(trained_net.parameters())

    # Find latest savepoint:
    latest_epoch_savepoint = -1
    for epoch in range(200):
        if os.path.exists(f"./network_epoch_{epoch}.pt"):
            latest_epoch_savepoint = epoch
    if latest_epoch_savepoint >= 0:
        print(f"Latest checkpoint found: epoch {latest_epoch_savepoint}")
        checkpoint = torch.load(f"./network_epoch_{latest_epoch_savepoint}.pt")
        trained_net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("No checkpoint found, starting from beginning")

    prev_obs = env.reset()
    replay_memory = ReplayMemory(max_len=1000000)
    minibatch_size = 32
    frame_index = 0
    with open("log.csv", "a") as file:
        writer = csv.writer(file)
        for epoch in range(latest_epoch_savepoint + 1, 200):
            net_updates_count = 0
            prev_net.load_state_dict(trained_net.state_dict())
            # One epoch corresponds to 50000 minibatch weight updates
            while net_updates_count < 50000:
                # Choose the action
                # epsilon annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter
                if frame_index < 1000000:
                    epsilon = 0.1 + 0.9 * (1000000 - frame_index) / 1000000
                else:
                    epsilon = 0.1

                if (
                    random.random() < epsilon
                    or prev_obs is None
                    or prev_obs.shape[0] < 4
                ):
                    action = random.randint(0, num_actions - 1)
                else:
                    inp = torch.tensor(prev_obs).double()
                    inp = inp.unsqueeze(0)
                    outp = trained_net(inp)  # TODO or prev_net?
                    action = torch.argmax(outp).item()

                actions = np.zeros(num_actions, dtype=np.int8)
                actions[action] = 1.0
                obs, rew, done, _ = env.step(actions)
                # gym.wrappers.FrameStack returns LazyFrames -- convert to numpy
                obs = np.asarray(obs)

                replay_memory.add(prev_obs, action, rew, obs, done)
                if len(replay_memory) > minibatch_size:
                    minibatch_indices = np.random.choice(
                        len(replay_memory), minibatch_size, replace=False
                    )
                    (
                        prev_obs_minibatch,
                        action_minibatch,
                        rew_minibatch,
                        obs_minibatch,
                        done_minibatch,
                    ) = replay_memory.get_minibatch(minibatch_indices)
                    gradient_descent_step(
                        prev_obs_minibatch,
                        action_minibatch,
                        rew_minibatch,
                        obs_minibatch,
                        done_minibatch,
                        optimizer,
                        prev_net,
                        trained_net,
                    )
                    net_updates_count += 1
                    if net_updates_count % 200 == 199:  # print every 200 mini-batches
                        global running_loss
                        print(
                            f"[{epoch}, {net_updates_count + 1:5d}] loss: {running_loss / 200:.3f}"
                        )
                        writer.writerow([epoch, net_updates_count, running_loss])
                        running_loss = 0.0

                # if rew != 0.0:
                #     print(rew)

                # env.render()
                # show_history(obs)
                frame_index += 1

                if done:
                    obs = env.reset()
                prev_obs = obs
            torch.save(
                {
                    # 'epoch': EPOCH,
                    "model_state_dict": trained_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # 'loss': LOSS,
                },
                f"./network_epoch_{epoch}.pt",
            )
    env.close()


def show_history(history):
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(history[0, :, :], cmap="gray")
    axs[1].imshow(history[1, :, :], cmap="gray")
    axs[2].imshow(history[2, :, :], cmap="gray")
    axs[3].imshow(history[3, :, :], cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
