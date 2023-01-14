import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


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


def get_action_epsilon_greedy(
    epsilon: float, obs: np.ndarray, net: Net, num_actions: int
):
    if random.random() < epsilon or obs is None or obs.shape[0] < 4:
        action = random.randint(0, num_actions - 1)
    else:
        action = get_action(obs, net)
    return action


def get_action(frames: np.ndarray, net: Net):
    if type(frames) != np.ndarray:
        print(type(frames))
    inp = torch.tensor(frames).double()
    inp = inp.unsqueeze(0)
    outp = net(inp)
    action = torch.argmax(outp).item()
    return action
