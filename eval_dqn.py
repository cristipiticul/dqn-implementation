# Run with
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python eval_dqn.py
import os
import numpy as np
import csv

from src.environment import create_env
from src.net import Net, get_action_epsilon_greedy
from src.util import max_epochs, model_checkpoint_filename, load_checkpoint_model


def main():
    env = create_env()
    num_actions = env.action_space.shape[0]
    trained_net = Net(num_actions).double()
    prev_obs = env.reset()
    with open("eval.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                *[f"score_episode_{i}" for i in range(1, 11)],
                *[f"length_episode_{i}" for i in range(1, 11)],
            ]
        )
        file.flush()
        for epoch in range(max_epochs):
            if os.path.exists(model_checkpoint_filename(epoch)):
                load_checkpoint_model(trained_net, None, epoch)
                episode_scores = []
                episode_lengths = []
                for episode in range(10):
                    done = False
                    total_score = 0
                    episode_length = 0
                    while not done:
                        epsilon = 0.0
                        action = get_action_epsilon_greedy(
                            epsilon, prev_obs, trained_net, num_actions
                        )  # TODO trained_net or prev_net?

                        actions = np.zeros(num_actions, dtype=np.int8)
                        actions[action] = 1.0
                        obs, rew, done, _ = env.step(actions)
                        total_score += rew
                        episode_length += 1
                        # gym.wrappers.FrameStack returns LazyFrames -- convert to numpy
                        obs = np.asarray(obs)
                        if done:
                            obs = env.reset()
                        prev_obs = obs
                    episode_scores.append(total_score)
                    episode_lengths.append(episode_length)
                    print(
                        f"Epoch: {epoch}, Episode: {episode}, Score: {total_score}, Length: {episode_length}"
                    )
                writer.writerow([epoch, *episode_scores, *episode_lengths])
                file.flush()
            else:
                print(f"Checkpoint for epoch {epoch} not found. Stopping.")
                break


if __name__ == "__main__":
    main()
