# Run with
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python dqn.py

# TODO:
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

from typing import Dict

REPLAY_MEMORY_FILE = "replay_memory.hdf5"
REPLAY_MEMORY_TABLE_NAME = "replay_memory_table"


criterion = torch.nn.MSELoss()
gamma = 0.95  # TODO: choose a discount factor
running_loss = 0
num_epochs = 200
replay_memory_size = 1000000


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


class ReplayMemoryRowDescription(tables.IsDescription):
    # pos is needed so that modify_rows works
    prev_obs = tables.UInt8Col(shape=(4, 84, 84), pos=1)
    action = tables.UInt8Col(pos=2)
    reward = tables.Float32Col(pos=3)
    obs = tables.UInt8Col(shape=(4, 84, 84), pos=4)
    done = tables.BoolCol(pos=5)


class ReplayMemory:
    """Store multiple (prev_obs, action, reward, obs, done) in a table using PyTables."""

    def __init__(
        self,
        max_len,
        checkpoint_filename: str = None,
        current_index_filename: str = None,
    ):
        self.initialized = False
        self.max_len = max_len
        self.open_file(checkpoint_filename, current_index_filename)

    def open_file(
        self, checkpoint_filename: str = None, current_index_filename: str = None
    ):
        if checkpoint_filename:
            tables.copy_file(checkpoint_filename, REPLAY_MEMORY_FILE, overwrite=True)
            self.h5file = tables.open_file(REPLAY_MEMORY_FILE, mode="a")
            self.table = self.h5file.get_node(
                self.h5file.root, REPLAY_MEMORY_TABLE_NAME
            )
            self.n_items = self.table.nrows
            with open(current_index_filename, "r") as f:
                self.curr_index = int(f.read())
        else:
            self.h5file = tables.open_file(
                REPLAY_MEMORY_FILE, mode="w", title="replay_memory"
            )
            self.table = self.h5file.create_table(
                self.h5file.root,
                REPLAY_MEMORY_TABLE_NAME,
                ReplayMemoryRowDescription,
                expectedrows=self.max_len,
            )
            self.n_items = 0
            self.curr_index = (
                0  # circular array - if max_len is reached, overwrite index 0
            )

    def close(self):
        self.h5file.close()

    def add(
        self,
        prev_obs: np.ndarray,
        action: int,
        reward: float,
        obs: np.ndarray,
        done: bool,
    ):
        if self.n_items < self.max_len:
            entry = self.table.row
            entry["prev_obs"] = prev_obs
            entry["action"] = action
            entry["reward"] = reward
            entry["obs"] = obs
            entry["done"] = done
            entry.append()
            self.n_items += 1
            self.table.flush()
        else:
            self.table.modify_rows(
                start=self.curr_index,
                stop=self.curr_index + 1,
                rows=[(prev_obs, action, reward, obs, done)],
            )
            self.table.flush()

        # Restart from 0 if got at the end
        self.curr_index += 1
        if self.curr_index >= self.max_len:
            self.curr_index = 0

    def get_minibatch(self, indices: list):
        return self.table.read_coordinates(indices)

    def save(self, filename, curr_index_filename):
        self.h5file.copy_file(dstfilename=filename)
        with open(curr_index_filename, "w") as f:
            f.write(str(self.curr_index))

    def __len__(self):
        return self.n_items


def replay_memory_test():
    print("Testing replay memory")
    replay_memory_fields_test()
    replay_memory_overflow_test()
    replay_memory_checkpoint_test()
    print("Done testing replay memory")


def replay_memory_fields_test():
    r = ReplayMemory(max_len=4)
    prev_obs = np.zeros((4, 84, 84), dtype=np.float32)
    obs = np.zeros((4, 84, 84), dtype=np.float32)
    r.add(prev_obs, 0, 0.0, obs, False)
    r.add(prev_obs + 1, 1, 1.0, obs + 1, False)
    r.add(prev_obs + 2, 2, 2.0, obs + 2, True)
    assert len(r) == 3
    assert r.curr_index == 3
    mb = r.get_minibatch([0, 2])
    assert mb[0]["prev_obs"][0, 0, 0] == 0
    assert mb[0]["action"] == 0
    assert mb[0]["reward"] == 0
    assert mb[0]["obs"][0, 0, 0] == 0
    assert mb[0]["done"] == False
    assert mb[1]["prev_obs"][0, 0, 0] == 2
    assert mb[1]["action"] == 2
    assert mb[1]["reward"] == 2
    assert mb[1]["obs"][0, 0, 0] == 2
    assert mb[1]["done"] == True
    r.close()
    os.remove(REPLAY_MEMORY_FILE)


def replay_memory_overflow_test():
    r = ReplayMemory(max_len=4)
    prev_obs = np.zeros((4, 84, 84), dtype=np.float32)
    obs = np.zeros((4, 84, 84), dtype=np.float32)
    r.add(prev_obs, 0, 0.0, obs, False)
    r.add(prev_obs + 1, 1, 1.0, obs + 1, False)
    r.add(prev_obs + 2, 2, 2.0, obs + 2, True)
    r.add(prev_obs + 3, 3, 3.0, obs + 3, True)
    r.add(prev_obs + 4, 4, 4.0, obs + 4, True)
    r.add(prev_obs + 5, 5, 5.0, obs + 5, True)
    assert len(r) == 4
    assert r.curr_index == 2
    mb = r.get_minibatch([0, 1, 2])
    assert mb[0]["prev_obs"][0, 0, 0] == 4
    assert mb[1]["prev_obs"][0, 0, 0] == 5
    assert mb[2]["prev_obs"][0, 0, 0] == 2
    r.close()
    os.remove(REPLAY_MEMORY_FILE)


def replay_memory_checkpoint_test():
    r = ReplayMemory(max_len=4)
    prev_obs = np.zeros((4, 84, 84), dtype=np.float32)
    obs = np.zeros((4, 84, 84), dtype=np.float32)
    r.add(prev_obs, 0, 0.0, obs, False)
    r.add(prev_obs + 1, 1, 1.0, obs + 1, False)
    r.add(prev_obs + 2, 2, 2.0, obs + 2, True)
    r.add(prev_obs + 3, 3, 3.0, obs + 3, True)
    r.add(prev_obs + 4, 4, 4.0, obs + 4, True)
    r.add(prev_obs + 5, 5, 5.0, obs + 5, True)
    r.save("test_checkpoint.hdf5", "test_checkpoint_index.txt")
    r.close()
    os.remove(REPLAY_MEMORY_FILE)

    r2 = ReplayMemory(4, "test_checkpoint.hdf5", "test_checkpoint_index.txt")
    assert len(r2) == 4
    assert r2.curr_index == 2
    r2.add(prev_obs + 6, 6, 6.0, obs + 6, True)
    mb = r2.get_minibatch([1, 2])
    assert mb[0]["prev_obs"][0, 0, 0] == 5
    assert mb[1]["prev_obs"][0, 0, 0] == 6
    r2.close()

    os.remove(REPLAY_MEMORY_FILE)
    os.remove("test_checkpoint.hdf5")
    os.remove("test_checkpoint_index.txt")


def gradient_descent_step(
    minibatch: Dict[str, np.ndarray],
    optimizer: torch.optim.Optimizer,
    prev_net: nn.Module,
    trained_net: nn.Module,
):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    inp = torch.tensor(minibatch["prev_obs"]).double()
    outputs = trained_net(inp)

    # We only care about the predicted Q-values of actions we've chosen, select those
    actions_torch = torch.unsqueeze(
        torch.tensor(minibatch["action"], dtype=torch.int64), dim=1
    )
    outputs_for_actions = torch.gather(outputs, 1, actions_torch).squeeze()

    with torch.no_grad():
        # Target = reward + gamma * max_a prev_net(s')
        rewards = torch.tensor(np.array(minibatch["reward"]))
        prev_net_outputs = prev_net(torch.tensor(minibatch["obs"]).double())
        # select maximum
        state_values = prev_net_outputs.max(dim=1).values
        # ignore state values for terminal states
        state_values = state_values * (
            1 - torch.tensor(np.array(minibatch["done"]).astype(int))
        )
        # add them
        target_outputs = rewards + gamma * state_values

    loss = criterion(outputs_for_actions, target_outputs)
    loss.backward()
    optimizer.step()

    global running_loss
    running_loss += loss.item()


def model_checkpoint_filename(epoch):
    return f"network_epoch_{epoch}.pt"


def replay_memory_checkpoint_filename(epoch):
    return f"replay_memory_epoch_{epoch}.hdf5"


def replay_memory_index_checkpoint_filename(epoch):
    return f"replay_memory_index_epoch_{epoch}.txt"


def get_last_checkpoint():
    latest_epoch_savepoint = None
    for epoch in range(num_epochs):
        if os.path.exists(model_checkpoint_filename(epoch)):
            latest_epoch_savepoint = epoch
    return latest_epoch_savepoint


def load_checkpoint_model(trained_net, optimizer, epoch):
    checkpoint = torch.load(model_checkpoint_filename(epoch))
    trained_net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def main():
    replay_memory_test()
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
    last_checkpoint_epoch = get_last_checkpoint()
    if last_checkpoint_epoch:
        print(f"Latest checkpoint found: epoch {last_checkpoint_epoch}")
        load_checkpoint_model(trained_net, optimizer, last_checkpoint_epoch)
        replay_memory = ReplayMemory(
            replay_memory_size,
            replay_memory_checkpoint_filename(last_checkpoint_epoch),
            replay_memory_index_checkpoint_filename(last_checkpoint_epoch),
        )
    else:
        print("No checkpoint found, starting from beginning")
        replay_memory = ReplayMemory(replay_memory_size)

    prev_obs = env.reset()
    minibatch_size = 32
    frame_index = 0
    with open("log.csv", "a") as file:
        writer = csv.writer(file)
        start_epoch = (last_checkpoint_epoch + 1) if last_checkpoint_epoch else 0
        for epoch in range(start_epoch, num_epochs):
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
                    minibatch = replay_memory.get_minibatch(minibatch_indices)
                    gradient_descent_step(
                        minibatch,
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
                model_checkpoint_filename(epoch),
            )
            replay_memory.save(
                replay_memory_checkpoint_filename(epoch),
                replay_memory_index_checkpoint_filename(epoch),
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
