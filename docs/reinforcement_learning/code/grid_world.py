import torch
from torchrl.envs import EnvBase
from torchrl.envs.utils import step_mdp
import pygame
import numpy as np


class StochasticGridWorldEnv(EnvBase):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.grid_size = (3, 4)
        self.start_state = (2, 0)
        self.current_state = self.start_state
        self.wall = (1, 1)
        self.rewards = {(0, 3): 1.0, (1, 3): -1.0}
        self.actions = {0: "North", 1: "South", 2: "East", 3: "West"}
        self.gamma = 0.9
        self.done = False

        # Pygame setup
        pygame.init()
        self.cell_size = 100
        self.screen = pygame.display.set_mode((self.grid_size[1] * self.cell_size, self.grid_size[0] * self.cell_size))
        pygame.display.set_caption("Stochastic Grid World")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.current_state = self.start_state
        self.done = False
        return torch.tensor([self.current_state], dtype=torch.float32).to(self.device)

    def _is_valid_move(self, state, action):
        row, col = state
        if action == 0:  # North
            new_row, new_col = row - 1, col
        elif action == 1:  # South
            new_row, new_col = row + 1, col
        elif action == 2:  # East
            new_row, new_col = row, col + 1
        elif action == 3:  # West
            new_row, new_col = row, col - 1
        else:
            raise ValueError(f"Invalid action: {action}")

        if not (0 <= new_row < self.grid_size[0] and 0 <= new_col < self.grid_size[1]):
            return False
        if (new_row, new_col) == self.wall:
            return False
        return True

    def _move(self, state, action):
        if not self._is_valid_move(state, action):
            return state
        row, col = state
        if action == 0:  # North
            new_row, new_col = row - 1, col
        elif action == 1:  # South
            new_row, new_col = row + 1, col
        elif action == 2:  # East
            new_row, new_col = row, col + 1
        elif action == 3:  # West
            new_row, new_col = row, col - 1
        return (new_row, new_col)

    def step(self, action):
        action = action.item()
        reward = 0.0
        done = False

        if action == 0:  # North
            if np.random.rand() < 0.8:
                self.current_state = self._move(self.current_state, 0)
            elif np.random.rand() < 0.5:
                self.current_state = self._move(self.current_state, 3)
            else:
                self.current_state = self._move(self.current_state, 2)
        else:
            self.current_state = self._move(self.current_state, action)

        if self.current_state in self.rewards:
            reward = self.rewards[self.current_state]
            done = True

        self.done = done
        return step_mdp(
            next_observation=torch.tensor([self.current_state], dtype=torch.float32).to(self.device),
            reward=torch.tensor([reward], dtype=torch.float32).to(self.device),
            done=torch.tensor([done]).to(self.device),
            info={},
        )

    def render(self):
        self.screen.fill((255, 255, 255))  # White background

        # Draw grid lines
        for x in range(self.grid_size[1]):
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.grid_size[0] * self.cell_size),
            )
        for y in range(self.grid_size[0]):
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (0, y * self.cell_size),
                (self.grid_size[1] * self.cell_size, y * self.cell_size),
            )

        # Draw start position
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (
                self.start_state[1] * self.cell_size + 5,
                self.start_state[0] * self.cell_size + 5,
                self.cell_size - 10,
                self.cell_size - 10,
            ),
        )

        # Draw wall
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (
                self.wall[1] * self.cell_size + 5,
                self.wall[0] * self.cell_size + 5,
                self.cell_size - 10,
                self.cell_size - 10,
            ),
        )

        # Draw goal positions
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (3 * self.cell_size + 5, 0 * self.cell_size + 5, self.cell_size - 10, self.cell_size - 10),
        )  # Green
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (3 * self.cell_size + 5, 1 * self.cell_size + 5, self.cell_size - 10, self.cell_size - 10),
        )  # Red

        # Draw agent
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),
            (int((self.current_state[1] + 0.5) * self.cell_size), int((self.current_state[0] + 0.5) * self.cell_size)),
            self.cell_size // 3,
        )

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


# Example usage
if __name__ == "__main__":
    env = StochasticGridWorldEnv()
    obs = env.reset()
    while not env.done:
        env.render()
        action = np.random.randint(4)  # Random action for demonstration
        obs, reward, done, info = env.step(torch.tensor(action))
        print(f"Action: {env.actions[action]}, Reward: {reward.item()}, Done: {done.item()}")
    env.close()
