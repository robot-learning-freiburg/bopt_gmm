import logging
from pathlib import Path
import numpy as np
from collections import deque, namedtuple
from tqdm import tqdm

from dataclasses import dataclass

Transition = namedtuple(
    "Transition", ["state", "action", "next_state", "reward", "done"]
)

def _default_processor(data) -> Transition:
    return Transition(data["state"].item(),
                      data["action"],
                      data["next_state"].item(),
                      data["reward"].item(),
                      data["done"].item())

class ReplayBuffer:
    def __init__(self, max_capacity=5000000):
        self.logger = logging.getLogger(__name__)
        self.unsaved_transitions = 0
        self.curr_file_idx = 1
        self.replay_buffer = deque(maxlen=int(max_capacity))

    def __getitem__(self, idx):
        return self.replay_buffer[idx]

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        transition = Transition(state, action, next_state, reward, done)
        self.replay_buffer.append(transition)
        self.unsaved_transitions += 1

    def sample(self, batch_size: int):
        indices = np.random.choice(
            len(self.replay_buffer),
            min(len(self.replay_buffer), batch_size),
            replace=False,
        )
        states, actions, next_states, rewards, dones = zip(
            *[self.replay_buffer[idx] for idx in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
        )

    def save(self, path="./replay_buffer"):
        if path is None:
            return False
        if self.unsaved_transitions > 0:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            final_rb_index = len(self.replay_buffer)
            start_rb_index = len(self.replay_buffer) - self.unsaved_transitions
            for replay_buffer_index in range(start_rb_index, final_rb_index):
                transition = self.replay_buffer[replay_buffer_index]
                file_name = "%s/transition_%09d.npz" % (path, self.curr_file_idx)
                np.savez(
                    file_name,
                    state=transition.state,
                    action=transition.action,
                    next_state=transition.next_state,
                    reward=transition.reward,
                    done=transition.done,
                )
                self.curr_file_idx += 1
            # Logging
            if self.unsaved_transitions == 1:
                self.logger.info(
                    "Saved file with index : %09d" % (self.curr_file_idx - 1)
                )
            else:
                self.logger.info(
                    "Saved files with indices : %09d - %09d"
                    % (
                        self.curr_file_idx - self.unsaved_transitions,
                        self.curr_file_idx - 1,
                    )
                )
            self.unsaved_transitions = 0
            return True
        return False

    def load(self, path="./replay_buffer", data_processor=_default_processor, num_transitions=None):
        if path is None:
            raise RuntimeError(f'Need path to replay buffer')

        p = Path(path)
        if not p.is_dir():
            raise RuntimeError(f'"{path}" is not a directory.')

        x = -1  # Trick to make logging work
        for x, file in enumerate(tqdm([f for f in p.glob("*.npz") if f.is_file()][:num_transitions], desc='Loading replay buffer...')):
            data = np.load(file, allow_pickle=True)
            transition = data_processor(data)
            self.replay_buffer.append(transition)

        self.curr_file_idx = x + 1
        self.logger.info(f'Replay buffer loaded successfully {self.curr_file_idx} files')
