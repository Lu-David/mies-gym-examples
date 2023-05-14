import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=20):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.float32),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.float32),
                "agent_battery": spaces.Box(0, 100, shape=(1,), dtype=np.float32),
                "time_remaining": spaces.Box(0, 100, shape=(1,), dtype=np.float32),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(2)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # self._action_to_direction = {
        #     0: np.array([1, 0]),
        #     1: np.array([0, 1]),
        #     2: np.array([-1, 0]),
        #     3: np.array([0, -1]),
        # }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, 
                "distance": self._target_location, 
                "agent_battery" : self._agent_battery,
                "time_remaining" : self._time_remaining
                }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        bat_w_x = np.random.rand() * self.size
        bat_w_y = np.random.rand() * self.size
        self._agent_location = np.array([bat_w_x, bat_w_y])
        self._agent_battery = np.random.randint(50, 100)

        moth_w_x = np.random.rand() * self.size
        moth_w_y = np.random.rand() * self.size
        self._target_location = np.array([moth_w_x, moth_w_y])

        self._time_remaining = 5

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        dt = 0.1
        move_drain_factor = 0.5
        rest_drain_factor = 0.1

        bat_vel = self._target_location - self._agent_location
        if action == 1:
            self._agent_location += bat_vel * dt
            self._agent_battery -= move_drain_factor * np.linalg.norm(bat_vel)
        elif action == 0:
            self._agent_battery -= rest_drain_factor

        reward = 0
        terminated = 0
        if np.linalg.norm(bat_vel) < 0.5:
            reward = 1
            terminated = 1
        elif self._agent_battery < 0: 
            reward = -1 
            terminated = 1
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()


        self._time_remaining -= dt

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()

            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        font = pygame.font.SysFont('Comic Sans MS', 30)

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Add in battery state 
        battery_img = font.render(f"Battery Level: {round(self._agent_battery)}", True, (0, 0, 0))
        time_img = font.render(f"Time remaining: {round(self._time_remaining, 2)}", True, (0, 0, 0))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())

            # Add battery level 
            self.window.blit(battery_img, (20, 20))
            self.window.blit(time_img, (20, self.window_size - 20))
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

