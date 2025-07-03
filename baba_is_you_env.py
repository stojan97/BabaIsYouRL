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

    # def observation(self, obs):
    #     """
    #     Calculates a unique integer for each (position, direction) pair.
    #     """
    #
    #     agent_pos = self.unwrapped.agent_pos
    #     agent_dir = self.unwrapped.agent_dir
    #
    #     pos_idx = agent_pos[1] * self.unwrapped.width + agent_pos[0]
    #
    #     print(agent_pos, agent_dir, pos_idx)
    #
    #     # Calculate the final unique index.
    #     # Formula ensures every (pos_idx, agent_dir) pair maps to a unique number.
    #     # Example: For a 7x7 grid (49 positions):
    #     # - (pos 0, dir 0) -> 0*4 + 0 = 0
    #     # - (pos 0, dir 1) -> 0*4 + 1 = 1
    #     # - (pos 1, dir 0) -> 1*4 + 0 = 4
    #     # - (pos 1, dir 1) -> 1*4 + 1 = 5
    #     state_idx = pos_idx * 4 + agent_dir
    #
    #     return state_idx


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

        self.text_objects.clear()  # Reset from previous episodes
        # pos[y, x] = y is col, x is row

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

        # Update agent direction for rendering purposes
        self.agent_dir = ACTION_TO_DIR[action]

        # Get the desired new position
        move_vec = ACTION_VECTOR[action]
        target_pos = self.agent_pos + move_vec
        target_cell = self.grid.get(*target_pos)

        # Move if the cell is empty or can be overlapped
        if target_cell is None or target_cell.can_overlap():
            self.agent_pos = tuple(target_pos)

        # Handle pushing a text block
        elif isinstance(target_cell, Text):
            push_pos = tuple(target_pos + move_vec)
            if self.grid.get(*push_pos) is None:  # Check if space behind is empty
                # Move the text block
                self.grid.set(*push_pos, target_cell)
                target_cell.pos = push_pos
                self.grid.set(*target_pos, None)
                # Move the agent
                self.agent_pos = tuple(target_pos)

        # --- Post-Action Checks (run every step) ---

        self._check_rules()

        current_cell = self.grid.get(*self.agent_pos)
        if current_cell is not None and current_cell.type == "goal" and self.rule_is_win_active:
            terminated = True
            reward = 1

        # 3. Check for truncation due to the step limit
        truncated = self.step_count >= self.max_steps

        # Generate observation and return
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
