import tables
import os
import numpy as np

REPLAY_MEMORY_FILE = "replay_memory.hdf5"
REPLAY_MEMORY_TABLE_NAME = "replay_memory_table"
REPLAY_MEMORY_COMPRESSION_ENABLED = True
REPLAY_MEMORY_COMPRESSION_LEVEL = 1  # Number between 1-9


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
            filters = None
            if REPLAY_MEMORY_COMPRESSION_ENABLED:
                filters = tables.Filters(
                    complevel=REPLAY_MEMORY_COMPRESSION_LEVEL,
                    complib="blosc",
                    fletcher32=True,
                )

            self.table = self.h5file.create_table(
                self.h5file.root,
                REPLAY_MEMORY_TABLE_NAME,
                ReplayMemoryRowDescription,
                expectedrows=self.max_len,
                filters=filters,
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
