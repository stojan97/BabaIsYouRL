# baba_is_you_env.py (Final, Tested Version)

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from PIL import Image
from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, WorldObj, Wall
from minigrid.minigrid_env import MiniGridEnv

ACTION_VECTOR = [
    np.array((0, -1)),  # UP
    np.array((1, 0)),  # RIGHT
    np.array((0, 1)),  # DOWN
    np.array((-1, 0)),  # LEFT
]

ACTION_TO_DIR = {0: 3, 1: 0, 2: 1, 3: 2}


# We must wrap the environment to have a discrete observation space for the Q-table
class PositionalWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Discrete(env.width * env.height)

    def observation(self, obs):
        # Convert (x, y) position to a single integer. Direction is no longer needed.
        agent_pos = self.unwrapped.agent_pos
        width = self.unwrapped.width

        return agent_pos[1] * width + agent_pos[0]


class FullStateWrapper(gym.ObservationWrapper):
    """
    Encodes the full state of the game (agent position and all text block positions)
    into a single, unique integer. This makes the environment fully observable.

    More on the encoding on the wiki page:
    https://en.wikipedia.org/wiki/Row-_and_column-major_order
    """

    def __init__(self, env):
        super().__init__(env)

        # The number of possible locations for any single object.
        self.num_cells = env.width * env.height

        # The total state space size is (num_cells)^(number of dynamic objects).
        # We have 4 dynamic objects: Agent, 'FLAG' text, 'IS' text, 'WIN' text.
        # WARNING: This number can get astronomically large!
        observation_space_size = self.num_cells ** 4

        print("=" * 50)
        print(f"USING FULL STATE REPRESENTATION")
        print(f"Grid size: {env.width}x{env.height} ({self.num_cells} cells)")
        print(f"Number of dynamic objects: 4")
        print(f"Total state space size: {self.num_cells}^4 = {observation_space_size}")
        print("=" * 50)

        self.observation_space = gym.spaces.Discrete(observation_space_size)

    def observation(self, obs):
        """
        Calculates the unique state index based on all object positions.
        """
        # Get the current positions of all dynamic objects.
        agent_pos = self.unwrapped.agent_pos
        flag_text_pos = self.unwrapped.text_objects["FLAG"].pos
        is_text_pos = self.unwrapped.text_objects["IS"].pos
        win_text_pos = self.unwrapped.text_objects["WIN"].pos

        # Convert each (x,y) tuple to a single integer index from 0 to num_cells-1.
        agent_idx = agent_pos[1] * self.unwrapped.width + agent_pos[0]
        flag_idx = flag_text_pos[1] * self.unwrapped.width + flag_text_pos[0]
        is_idx = is_text_pos[1] * self.unwrapped.width + is_text_pos[0]
        win_idx = win_text_pos[1] * self.unwrapped.width + win_text_pos[0]

        # Combine these four indices into a single unique state ID.
        # This works like converting a number to a different base, where the base is `num_cells`.
        # This guarantees that every unique combination of positions maps to a unique integer.
        state_idx = agent_idx
        state_idx = state_idx * self.num_cells + flag_idx
        state_idx = state_idx * self.num_cells + is_idx
        state_idx = state_idx * self.num_cells + win_idx

        return state_idx


TEXT_TO_IMAGE = {
    'BABA': Image.open('assets/baba.png').convert('RGBA'),
    'FLAG': Image.open('assets/flag.png').convert('RGBA'),
    'IS': Image.open('assets/is.png').convert('RGBA'),
    'WIN': Image.open('assets/win.png').convert('RGBA')
}


# Enhanced Text class with pushing capability
class Text(WorldObj):
    def __init__(self, text_label: str):
        if text_label not in OBJECT_TO_IDX:
            OBJECT_TO_IDX[text_label] = max(OBJECT_TO_IDX.values()) + 1

        super().__init__(text_label, 'red')

        self.text_label = text_label
        self.image = TEXT_TO_IMAGE[self.text_label]
        self._pos = None

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def render(self, img):
        png_pil = self.image.resize((img.shape[1], img.shape[0]), resample=Image.Resampling.LANCZOS)
        # print(f'RENDER...... Text Label={self.text_label}, image_dir={png_pil}')
        tile_pil = Image.fromarray(img)
        tile_pil.paste(png_pil, (0, 0), mask=png_pil)
        # Convert the modified PIL image back to a NumPy array and update the original tile
        img[:] = np.array(tile_pil)


class BabaIsYouGridEnv(MiniGridEnv):
    def __init__(self, level_map: list[str], **kwargs):
        # The level map now defines the size.
        self.level_map = level_map
        height = len(level_map)
        width = len(level_map[0]) if height > 0 else 0

        self.rule_is_win_active = False
        self.text_objects: Dict[str, Text] = {}
        self.goal_pos = None
        mission_space = MissionSpace(mission_func=lambda: "form 'FLAG IS WIN' and go to the flag")

        # The parent constructor now gets its size from the map.
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=4 * width * height,
            **kwargs,
        )

        self.action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._check_rules()
        return obs, info

    def _gen_grid(self, width: int, height: int):
        # This method is now entirely driven by self.level_map.
        self.grid = Grid(width, height)

        # Map lowercase characters to their text label.
        char_to_text = {
            'b': "BABA", 'f': "FLAG", 'i': "IS", 'w': "WIN",
            'r': "ROCK", 'p': "PUSH", 's': "STOP", 'y': "YOU"
        }

        # W = Wall, B = Baba Agent, F = Flag (Goal)
        # f = 'FLAG' text, i = 'IS' text, w = 'WIN' text
        # '.' = empty space
        #
        # pos[y, x] = y is col, x is row
        self.text_objects.clear()

        for y, row in enumerate(self.level_map):
            for x, char in enumerate(row):
                obj = None
                if char == 'B':
                    self.agent_pos = (x, y)
                    self.agent_dir = 0
                elif char == 'F':
                    obj = Goal()
                    self.goal_pos = (x, y)
                elif char == 'W':
                    obj = Wall()
                elif char in char_to_text:
                    text_label = char_to_text[char]
                    obj = Text(text_label)
                    obj.pos = (x, y)
                    self.text_objects[text_label] = obj
                if obj:
                    self.put_obj(obj, x, y)

        if self.agent_pos is None:
            raise ValueError("Level map must contain an agent 'B'.")

        self.rule_is_win_active = False

    def step(self, action):
        self.step_count += 1
        reward = -0.01
        terminated = False

        self.agent_dir = ACTION_TO_DIR[action]
        move_vec = ACTION_VECTOR[action]

        target_pos = tuple(self.agent_pos + move_vec)

        objects_to_push = []
        can_move = False

        # Check the first cell in front of the agent
        target_cell = self.grid.get(*target_pos)

        if target_cell is None or target_cell.can_overlap():
            can_move = True
        elif isinstance(target_cell, Text):
            # Potential push. Check the entire chain.
            check_pos = target_pos

            while True:
                # Boundary Check: Prevents crashing at the edge of the grid
                if not (0 <= check_pos[0] < self.width and 0 <= check_pos[1] < self.height):
                    can_move = False
                    break

                cell = self.grid.get(*check_pos)
                if cell is None: break  # Chain can be pushed
                if isinstance(cell, Text):
                    objects_to_push.append(cell)
                    check_pos = tuple(np.array(check_pos) + move_vec)
                else:
                    can_move = False  # Chain is blocked
                    break

        if can_move:
            for obj in reversed(objects_to_push):
                old_obj_pos = obj.pos
                new_obj_pos = tuple(old_obj_pos + move_vec)

                self.grid.set(*new_obj_pos, obj)
                self.grid.set(*old_obj_pos, None)

                obj.pos = new_obj_pos

            old_agent_pos = self.agent_pos
            self.grid.set(*old_agent_pos, None)
            self.agent_pos = target_pos

        # --- Post-Action Checks (unchanged) ---
        self._check_rules()

        current_cell = self.grid.get(*self.agent_pos)
        if current_cell is not None and current_cell.type == "goal" and self.rule_is_win_active:
            terminated = True
            reward = 50.0

        truncated = self.step_count >= self.max_steps
        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}

    def _check_rules(self):
        # Check if all objects have been placed with positions
        if (self.text_objects["FLAG"].pos is None or
                self.text_objects["IS"].pos is None or
                self.text_objects["WIN"].pos is None):
            self.rule_is_win_active = False
            return

        fx, fy = self.text_objects["FLAG"].pos
        ix, iy = self.text_objects["IS"].pos
        wx, wy = self.text_objects["WIN"].pos

        is_horizontal = (fy == iy == wy) and (fx + 1 == ix) and (ix + 1 == wx)
        is_vertical = (fx == ix == wx) and (fy + 1 == iy) and (iy + 1 == wy)

        self.rule_is_win_active = is_horizontal or is_vertical
