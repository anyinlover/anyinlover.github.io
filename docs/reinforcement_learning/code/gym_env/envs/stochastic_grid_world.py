from enum import Enum
import random  # Added for stochasticity
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class StochasticGridWorldEnv(gym.Env):
    """
    Custom Environment for a 3x4 Stochastic Grid World.

    The agent starts at (0,0) and needs to navigate to maximize reward.
    - Grid Size: 3 rows, 4 columns (indexed 0-2 for rows, 0-3 for columns)
    - Start State: (0, 0) - Bottom Left
    - Wall: (1, 1) - Agent cannot enter this cell
    - Rewards:
        - +1.0 at (3, 2) - Top Right (Green)
        - -1.0 at (3, 1) - Cell below Green (Red)
    - Actions: 0: North, 1: South, 2: East, 3: West
    - Stochasticity (North Action):
        - 80% chance: Moves North
        - 10% chance: Moves West
        - 10% chance: Moves East
    - Stochasticity (Other Actions): Deterministic (100% chance)
    - Collision: If a move results in hitting the wall or going out of bounds,
                 the agent stays in its current cell.
    - Termination: The episode ends when the agent reaches either reward cell.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()

        self.size_rows = 3
        self.size_cols = 4
        self.grid_shape = (self.size_cols, self.size_rows)
        self.window_size = (512, 384)

        # Define agent start state
        self._start_state = np.array([0, 2], dtype=int)

        # Define wall location (row, col)
        self._wall_location = np.array([1, 1], dtype=int)

        # Define reward locations and values
        self._reward_locations = {
            tuple([3, 0]): 10.0,  # Green cell (top-right)
            tuple([3, 1]): -10.0,  # Red cell (below green)
        }
        self._target_location_green = np.array([3, 0], dtype=int)  # For rendering
        self._target_location_red = np.array([3, 1], dtype=int)  # For rendering

        # Define action space (Discrete: 0=N, 1=S, 2=E, 3=W)
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            Actions.NORTH.value: np.array([0, -1]),  # North (increase row)
            Actions.SOUTH.value: np.array([0, 1]),  # South (increase row)
            Actions.EAST.value: np.array([1, 0]),  # East (increase col)
            Actions.WEST.value: np.array([-1, 0]),  # West (decrease col)
        }

        # Define observation space: Agent's location (col, row)
        # Use MultiDiscrete for [row, col] representation
        self.observation_space = spaces.MultiDiscrete(
            np.array([self.size_cols, self.size_rows]), dtype=int
        )

        # Initialize internal state
        self._agent_location = None

        # --- Rendering ---
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None  # For pygame rendering, if added later
        self.clock = None  # For pygame rendering, if added later

    def _get_obs(self):
        """Returns the current observation (agent's location)."""
        return self._agent_location.copy()  # Return a copy

    def _get_info(self):
        """Returns auxiliary information (e.g., distance)."""
        # You could add more info here if needed, like distance to goal
        return {"agent_location": self._agent_location.copy()}

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set the agent to the starting position
        self._agent_location = self._start_state.copy()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """Executes one step in the environment."""
        current_pos = self._agent_location

        # --- Determine the intended direction based on action and stochasticity ---
        intended_direction_idx = action

        if intended_direction_idx == 0:  # Action is North
            # Sample the outcome
            rand_num = self.np_random.random()  # Use env's RNG
            if rand_num < 0.8:
                actual_direction_idx = 0  # North
            elif rand_num < 0.9:
                actual_direction_idx = 3  # West
            else:
                actual_direction_idx = 2  # East
        else:  # Other actions are deterministic
            actual_direction_idx = intended_direction_idx

        # Get the change in coordinates for the actual direction
        delta = self._action_to_direction[actual_direction_idx]

        # Calculate the potential next position
        potential_next_pos = current_pos + delta

        # --- Check for collisions and boundaries ---
        next_pos_col, next_pos_row = potential_next_pos

        # Assume stay in place initially
        next_state = current_pos.copy()
        valid_move = True

        # Check boundaries
        if not (
            0 <= next_pos_row < self.size_rows and 0 <= next_pos_col < self.size_cols
        ):
            valid_move = False
        # Check wall collision
        elif np.array_equal(potential_next_pos, self._wall_location):
            valid_move = False

        # Update state only if the move is valid
        if valid_move:
            next_state = potential_next_pos

        # Update agent's location
        self._agent_location = next_state

        # --- Calculate Reward ---
        # Reward is based on the state *entered*
        reward = self._reward_locations.get(tuple(self._agent_location), -1.0)

        # --- Check for Termination ---
        # Episode terminates if the agent is in a reward cell
        terminated = tuple(self._agent_location) in self._reward_locations

        # Truncated is typically used for time limits, not applicable here unless added
        truncated = False

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "ansi":
            print(self._render_text())  # Print ANSI string directly

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renders the environment based on the selected mode."""
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "human":
            self._render_frame()
            return None

    def _render_text(self):
        """Helper function to generate a text representation."""
        grid = np.full(self.grid_shape, ".", dtype=str)

        # Mark wall
        wall_c, wall_r = self._wall_location
        grid[wall_c, wall_r] = "W"

        # Mark rewards
        green_c, green_r = self._target_location_green
        grid[green_c, green_r] = "G"
        red_c, red_r = self._target_location_red
        grid[red_c, red_r] = "R"

        # Mark agent
        agent_c, agent_r = self._agent_location
        # Ensure agent marker overwrites reward if agent is on a reward cell
        grid[agent_c, agent_r] = "A"

        # Create output string
        output = ""
        for r in range(self.size_rows):
            output += "".join(grid[:, r]) + "\n"
        return output

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size[0] / self.size_cols

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location_red,
                (pix_square_size, pix_square_size),
            ),
        )

        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._target_location_green,
                (pix_square_size, pix_square_size),
            ),
        )

        pygame.draw.rect(
            canvas,
            (100, 100, 100),
            pygame.Rect(
                pix_square_size * self._wall_location,
                (pix_square_size, pix_square_size),
            ),
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for y in range(self.size_rows):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * y),
                (self.window_size[0], pix_square_size * y),
                width=3,
            )

        for x in range(self.size_cols):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size[1]),
                width=3,
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Cleans up resources (e.g., rendering window)."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


def run_pygame_visualization(env, max_episode_steps=100):
    pygame.init()
    pygame.display.set_caption("Stochastic Grid World")
    CELL_SIZE = 128
    GRID_WIDTH = env.size_cols * CELL_SIZE
    GRID_HEIGHT = env.size_rows * CELL_SIZE
    INFO_HEIGHT = 100
    BUTTON_WIDTH = 80
    BUTTON_HEIGHT = 30
    BUTTON_PADDING = 10
    SCREEN_WIDTH = GRID_WIDTH
    SCREEN_HEIGHT = GRID_HEIGHT + INFO_HEIGHT

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREY = (128, 128, 128)
    DARK_GREY = (64, 64, 64)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BUTTON_COLOR = (70, 130, 180)
    BUTTON_TEXT_COLOR = WHITE

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    button_font = pygame.font.Font(None, 24)
    button_y = GRID_HEIGHT + (INFO_HEIGHT - BUTTON_HEIGHT) // 2
    start_button_rect = pygame.Rect(
        BUTTON_PADDING, button_y, BUTTON_WIDTH, BUTTON_HEIGHT
    )
    stop_button_rect = pygame.Rect(
        start_button_rect.right + BUTTON_PADDING, button_y, BUTTON_WIDTH, BUTTON_HEIGHT
    )
    reset_button_rect = pygame.Rect(
        stop_button_rect.right + BUTTON_PADDING, button_y, BUTTON_WIDTH, BUTTON_HEIGHT
    )

    running_simulation = False  # Controls if env.step() is called
    game_active = True  # Controls the main loop
    cumulative_reward = 0.0
    step_count = 0
    agent_action_policy = (
        lambda: env.action_space.sample()
    )  # Use random actions for now

    # Initialize environment state
    observation, info = env.reset(
        seed=random.randint(0, 10000)
    )  # Use random seed each time
    while game_active:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_active = False  # Exit main loop

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pos = event.pos
                    if (
                        start_button_rect.collidepoint(mouse_pos)
                        and step_count < max_episode_steps
                        and tuple(env._agent_location) not in env._reward_locations
                    ):
                        print("Start button clicked")
                        running_simulation = True
                    elif stop_button_rect.collidepoint(mouse_pos):
                        print("Stop button clicked")
                        running_simulation = False
                    elif reset_button_rect.collidepoint(mouse_pos):
                        print("Reset button clicked")
                        running_simulation = False
                        observation, info = env.reset(seed=random.randint(0, 10000))
                        cumulative_reward = 0.0
                        step_count = 0

        # --- Simulation Step ---
        terminated = tuple(env._agent_location) in env._reward_locations
        truncated = step_count >= max_episode_steps

        if running_simulation and not terminated and not truncated:
            action = agent_action_policy()  # Get action (random for now)
            try:
                observation, reward, term, trunc, info = env.step(action)
                cumulative_reward += reward
                step_count += 1
                terminated = term  # Update terminated status
                truncated = trunc  # Update truncated status (though env doesn't set it)

                # Automatically stop if max steps reached or episode terminates
                if terminated or step_count >= max_episode_steps:
                    running_simulation = False
                    print(
                        f"Simulation stopped. Terminated: {terminated}, Steps: {step_count}"
                    )

            except Exception as e:
                print(f"Error during env.step: {e}")
                running_simulation = False  # Stop on error

        # --- Drawing ---
        screen.fill(WHITE)  # Clear screen with white background

        # Draw Grid Lines
        for r in range(env.size_rows + 1):
            pygame.draw.line(
                screen, GREY, (0, r * CELL_SIZE), (GRID_WIDTH, r * CELL_SIZE)
            )
        for c in range(env.size_cols + 1):
            pygame.draw.line(
                screen, GREY, (c * CELL_SIZE, 0), (c * CELL_SIZE, GRID_HEIGHT)
            )

        # Draw Cells
        # Wall
        pygame.draw.rect(
            screen,
            DARK_GREY,
            pygame.Rect(CELL_SIZE * env._wall_location, (CELL_SIZE, CELL_SIZE)),
        )

        # Rewards
        pygame.draw.rect(
            screen,
            GREEN,
            pygame.Rect(CELL_SIZE * env._target_location_green, (CELL_SIZE, CELL_SIZE)),
        )
        pygame.draw.rect(
            screen,
            RED,
            pygame.Rect(CELL_SIZE * env._target_location_red, (CELL_SIZE, CELL_SIZE)),
        )

        # Agent
        pygame.draw.circle(
            screen, BLUE, (env._agent_location + 0.5) * CELL_SIZE, CELL_SIZE // 3
        )

        # Draw Info Area Background
        pygame.draw.rect(screen, GREY, (0, GRID_HEIGHT, SCREEN_WIDTH, INFO_HEIGHT))

        # Draw Buttons
        pygame.draw.rect(screen, BUTTON_COLOR, start_button_rect)
        pygame.draw.rect(screen, BUTTON_COLOR, stop_button_rect)
        pygame.draw.rect(screen, BUTTON_COLOR, reset_button_rect)

        start_text = button_font.render("Start", True, BUTTON_TEXT_COLOR)
        stop_text = button_font.render("Stop", True, BUTTON_TEXT_COLOR)
        reset_text = button_font.render("Reset", True, BUTTON_TEXT_COLOR)

        screen.blit(
            start_text,
            (
                start_button_rect.x
                + (start_button_rect.width - start_text.get_width()) // 2,
                start_button_rect.y
                + (start_button_rect.height - start_text.get_height()) // 2,
            ),
        )
        screen.blit(
            stop_text,
            (
                stop_button_rect.x
                + (stop_button_rect.width - stop_text.get_width()) // 2,
                stop_button_rect.y
                + (stop_button_rect.height - stop_text.get_height()) // 2,
            ),
        )
        screen.blit(
            reset_text,
            (
                reset_button_rect.x
                + (reset_button_rect.width - reset_text.get_width()) // 2,
                reset_button_rect.y
                + (reset_button_rect.height - reset_text.get_height()) // 2,
            ),
        )

        # Draw Step Count and Reward Text
        info_text_str = (
            f"Step: {step_count}/{max_episode_steps}    Reward: {cumulative_reward}"
        )
        info_text = font.render(info_text_str, True, BLACK)
        info_text_rect = info_text.get_rect(
            center=(SCREEN_WIDTH // 2, GRID_HEIGHT + INFO_HEIGHT // 4)
        )  # Position text top-center in info area
        # Adjust position if overlapping buttons - better to place text above buttons
        text_y = GRID_HEIGHT + INFO_HEIGHT * 0.25
        info_text_rect = info_text.get_rect(center=(SCREEN_WIDTH // 2, text_y))

        buttons_total_width = (BUTTON_WIDTH * 3) + (BUTTON_PADDING * 4)
        text_x_start = buttons_total_width
        info_text_rect = info_text.get_rect(
            midleft=(text_x_start, button_y + BUTTON_HEIGHT // 2)
        )

        screen.blit(info_text, info_text_rect)

        # --- Update Display ---
        pygame.display.flip()

        # --- Control Frame Rate ---
        clock.tick(10)

    pygame.quit()
    print("Pygame closed.")


if __name__ == "__main__":
    env_instance = StochasticGridWorldEnv()
    run_pygame_visualization(env_instance, max_episode_steps=50)
