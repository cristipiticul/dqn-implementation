# Run with
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python player.py
import numpy as np

from src.environment import create_env


def main():
    env = create_env()
    num_actions = env.action_space.shape[0]
    env.reset()
    while True:
        action = int(input("Action (-1=noop, -2=stop): "))
        repeat_times = int(input("Repeat times: "))
        actions = np.zeros(num_actions, dtype=np.int8)
        if action == -2:
            break
        if action != -1:
            actions[action] = 1.0
        for i in range(repeat_times):
            _, _, done, _ = env.step(actions)
            env.render()
            if done:
                print("Game over! Restarting")
                env.reset()
                break


if __name__ == "__main__":
    main()
