# Run with
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python eval_dqn.py
import os
import numpy as np
import csv
import random
import torch

from src.environment import create_env
from src.net import Net, get_action_epsilon_greedy
from src.util import max_epochs, model_checkpoint_filename, load_checkpoint_model

epsilon = 0.1

# We'll get the average score for 10 games for each checkpoint
num_episodes = 10
# We'll get the average q values for these many random states
num_random_states = 10000
# We'll run a random policy agent and select states with this prob
probability_to_pick_state = 0.25


def main():
    env = create_env()
    num_actions = env.action_space.shape[0]
    print("Getting random states...")
    random_states = get_random_states(env, num_actions)
    print("Done!")
    trained_net = Net(num_actions).double()
    with open("eval.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                *[f"score_episode_{i}" for i in range(1, num_episodes + 1)],
                *[f"length_episode_{i}" for i in range(1, num_episodes + 1)],
                "average_q_val",
            ]
        )
        file.flush()
        for epoch in range(max_epochs):
            if os.path.exists(model_checkpoint_filename(epoch)):
                print(f"Evaluating epoch: {epoch}")
                load_checkpoint_model(trained_net, None, epoch)
                episode_scores, episode_lengths = run_episodes(
                    env, trained_net, num_actions
                )
                average_q_values = get_average_q_values(trained_net, random_states)
                print(f"Average Q-values: {average_q_values}")
                writer.writerow(
                    [epoch, *episode_scores, *episode_lengths, average_q_values]
                )
                file.flush()
            else:
                print(f"Checkpoint for epoch {epoch} not found. Stopping.")
                break


def get_random_states(env, num_actions):
    states = []
    env.reset()
    while len(states) < num_random_states:
        actions = np.zeros(num_actions, dtype=np.int8)
        rand_action = random.randint(0, num_actions - 1)
        actions[rand_action] = 1.0
        obs, _, done, _ = env.step(actions)
        if random.random() < probability_to_pick_state:
            states.append(obs)
        if done:
            obs = env.reset()
            states.append(obs)
    return states


def run_episodes(env, trained_net, num_actions):
    episode_scores = []
    episode_lengths = []
    for episode in range(num_episodes):
        score, length = run_episode(env, trained_net, num_actions)
        episode_scores.append(score)
        episode_lengths.append(length)
        print(f"Episode: {episode}, Score: {score}, Length: {length}")
    return episode_scores, episode_lengths


def run_episode(env, trained_net, num_actions):
    prev_obs = env.reset()
    done = False
    total_score = 0
    episode_length = 0
    while not done:
        action = get_action_epsilon_greedy(epsilon, prev_obs, trained_net, num_actions)
        actions = np.zeros(num_actions, dtype=np.int8)
        actions[action] = 1.0
        obs, rew, done, _ = env.step(actions)
        total_score += rew
        episode_length += 1
        prev_obs = obs
        # print(actions)
        # env.render()
    return total_score, episode_length


def get_average_q_values(trained_net, random_states):
    current_sum = 0
    for state in random_states:
        inp = torch.tensor(state).double()
        inp = inp.unsqueeze(0)
        outp = trained_net(inp)
        q_value = torch.max(outp).item()
        current_sum += q_value
    return current_sum / len(random_states)


if __name__ == "__main__":
    main()
