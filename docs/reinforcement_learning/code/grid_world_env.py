from typing import Optional
import torch
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import DiscreteTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase


class GridWorldEnv(EnvBase):
    metadata = {"render_modes": ["human"]}

    def __init__(self, device="cpu", batch_size=None, seed=None):
        super().__init__(device=device, batch_size=batch_size)

        # Define action and observation specs
        self.action_spec = DiscreteTensorSpec(n=4, shape=torch.Size([]), device=device)
        self.observation_spec = DiscreteTensorSpec(n=12, shape=torch.Size([]), device=device)  # 3x4 grid
        self.reward_spec = DiscreteTensorSpec(n=1, shape=torch.Size([]), device=device)
        self.done_spec = DiscreteTensorSpec(n=1, shape=torch.Size([]), device=device)

        self.start_state = (0, 2)
        self.goal_states = {(2, 3): 1.0, (1, 3): -1.0}  # (row, col): reward
        self.wall = (1, 1)
        self.grid_size = (3, 4)

        self._action_to_delta = {
            0: (-1, 0),  # North
            1: (1, 0),  # South
            2: (0, 1),  # East
            3: (0, -1),  # West
        }

        self.stochastic_actions = {
            0: {0: 0.8, 2: 0.1, 3: 0.1},  # North is stochastic
        }

        self.set_seed(seed)

    def _reset(self, tensordict: TensorDictBase = None) -> TensorDict:
        state = self._pos_to_idx(self.start_state)
        done = torch.zeros(*self.batch_size, dtype=torch.bool, device=self.device)
        reward = torch.zeros(*self.batch_size, dtype=torch.float32, device=self.device)
        return TensorDict(
            {
                "observation": state,
                "done": done,
                "reward": reward,
            },
            batch_size=self.batch_size,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        state_idx = tensordict["observation"]
        action = tensordict["action"]

        # Convert index to position
        state_pos = self._idx_to_pos(state_idx)

        # Handle stochastic actions
        if action.item() in self.stochastic_actions:
            dist = self.stochastic_actions[action.item()]
            chosen_action = int(torch.multinomial(torch.tensor(list(dist.values())), 1).item())
        else:
            chosen_action = action.item()

        delta = self._action_to_delta[chosen_action]
        new_row = max(0, min(self.grid_size[0] - 1, state_pos[0] + delta[0]))
        new_col = max(0, min(self.grid_size[1] - 1, state_pos[1] + delta[1]))

        new_pos = (new_row, new_col)

        # Check wall collision
        if new_pos == self.wall:
            new_pos = state_pos

        # Get reward and check termination
        reward = 0.0
        done = False
        if new_pos in self.goal_states:
            reward = self.goal_states[new_pos]
            done = True

        # Convert new position back to index
        new_state_idx = self._pos_to_idx(new_pos)

        return TensorDict(
            {
                "observation": new_state_idx,
                "done": torch.tensor(done, dtype=torch.bool, device=self.device),
                "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
            },
            batch_size=tensordict.batch_size,
        )

    def _pos_to_idx(self, pos):
        return pos[0] * self.grid_size[1] + pos[1]

    def _idx_to_pos(self, idx):
        idx = idx.item() if isinstance(idx, torch.Tensor) else idx
        return divmod(idx, self.grid_size[1])

    def _set_seed(self, seed: Optional[int]):
        if seed:
            torch.manual_seed(seed)


if __name__ == "__main__":
    env = GridWorldEnv()
    obs = env.reset()
