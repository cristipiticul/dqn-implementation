# Run with
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python train_dqn.py

# TODO:
# - try different discount factors (gamma)
# - try different learning rates

import numpy as np
import torch
import torch.nn as nn
import os
import csv
from time import perf_counter
from typing import Dict

from src.environment import create_env
from src.replay_memory import replay_memory_test, ReplayMemory
from src.net import Net, get_action_epsilon_greedy
from src.util import (
    model_checkpoint_filename,
    replay_memory_checkpoint_filename,
    replay_memory_index_checkpoint_filename,
    get_last_checkpoint,
    load_checkpoint_model,
)

# More parameters in replay_memory.py
REPLAY_MEMORY_DELETE_OLD_CHECKPOINT = True

criterion = torch.nn.MSELoss()
gamma = 0.95  # TODO: choose a discount factor
running_loss = 0
num_epochs = 200
replay_memory_size = 1000000
learning_rate = 0.03  # default is 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gradient_descent_step(
    minibatch: Dict[str, np.ndarray],
    optimizer: torch.optim.Optimizer,
    prev_net: nn.Module,
    trained_net: nn.Module,
):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    inp = torch.tensor(minibatch["prev_obs"], device=device).double()
    outputs = trained_net(inp)

    # We only care about the predicted Q-values of actions we've chosen, select those
    actions_torch = torch.unsqueeze(
        torch.tensor(minibatch["action"], dtype=torch.int64, device=device), dim=1
    )
    outputs_for_actions = torch.gather(outputs, 1, actions_torch).squeeze()

    with torch.no_grad():
        # Target = reward + gamma * max_a prev_net(s')
        rewards = torch.tensor(np.array(minibatch["reward"]), device=device)
        prev_net_outputs = prev_net(
            torch.tensor(minibatch["obs"], device=device).double()
        )
        # select maximum
        state_values = prev_net_outputs.max(dim=1).values
        # ignore state values for terminal states
        state_values = state_values * (
            1 - torch.tensor(np.array(minibatch["done"]).astype(int), device=device)
        )
        # add them
        target_outputs = rewards + gamma * state_values

    loss = criterion(outputs_for_actions, target_outputs)
    loss.backward()
    optimizer.step()

    global running_loss
    running_loss += loss.item()


def main():
    replay_memory_test()
    env = create_env()

    num_actions = env.action_space.shape[0]

    # The parameters from the previous iteration are held fixed when optimising the loss function
    # ==> we need two sets of parameters
    prev_net = Net(num_actions).double().to(device)
    trained_net = Net(num_actions).double().to(device)
    optimizer = torch.optim.RMSprop(trained_net.parameters(), lr=learning_rate)

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
        writer.writerow(
            ["epoch", "net_updates_count", "avg_loss", "time_per_update_sec"]
        )
        last_print_time = perf_counter()
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

                action = get_action_epsilon_greedy(
                    epsilon, prev_obs, trained_net, num_actions
                )  # TODO trained_net or prev_net?

                actions = np.zeros(num_actions, dtype=np.int8)
                actions[action] = 1.0
                obs, rew, done, _ = env.step(actions)

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
                        new_print_time = perf_counter()
                        time_per_update = (new_print_time - last_print_time) / 200
                        last_print_time = new_print_time
                        print(
                            f"[{epoch}, {net_updates_count + 1:5d}] loss: {running_loss / 200:.3f}, time per update (s): {time_per_update}"
                        )
                        writer.writerow(
                            [
                                epoch,
                                net_updates_count,
                                running_loss / 200,
                                time_per_update,
                            ]
                        )
                        file.flush()
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
            prev_replay_mem_checkpoint = replay_memory_checkpoint_filename(epoch - 1)
            if REPLAY_MEMORY_DELETE_OLD_CHECKPOINT and os.path.exists(
                prev_replay_mem_checkpoint
            ):
                os.remove(prev_replay_mem_checkpoint)
                os.remove(replay_memory_index_checkpoint_filename(epoch - 1))
    env.close()


if __name__ == "__main__":
    main()
