import os
import torch
from typing import Union
from matplotlib import pyplot as plt

from .net import Net

max_epochs = 200


def model_checkpoint_filename(epoch):
    return f"network_epoch_{epoch}.pt"


def replay_memory_checkpoint_filename(epoch):
    return f"replay_memory_epoch_{epoch}.hdf5"


def replay_memory_index_checkpoint_filename(epoch):
    return f"replay_memory_index_epoch_{epoch}.txt"


def get_last_checkpoint():
    latest_epoch_savepoint = None
    for epoch in range(max_epochs):
        if os.path.exists(model_checkpoint_filename(epoch)):
            latest_epoch_savepoint = epoch
    return latest_epoch_savepoint


def load_checkpoint_model(
    trained_net: Net, optimizer: Union[torch.optim.Optimizer, None], epoch
):
    checkpoint = torch.load(model_checkpoint_filename(epoch))
    trained_net.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def show_history(history):
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(history[0, :, :], cmap="gray")
    axs[1].imshow(history[1, :, :], cmap="gray")
    axs[2].imshow(history[2, :, :], cmap="gray")
    axs[3].imshow(history[3, :, :], cmap="gray")
    plt.show()
