# baba_is_you_env.py (Final, Tested Version)

from typing import Any, Dict, Optional, List, Set

import gymnasium as gym
import numpy as np
from PIL import Image
from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, WorldObj, Wall
from minigrid.minigrid_env import MiniGridEnv


class FullStateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_cells = env.width * env.height

        # DYNAMIC OBJECTS: Agent, FLAG text, IS text, WIN text, PUSH text, STOP text, DEFEAT text,
        # KEY text, DOOR text, BALL text, Key obj, Door obj, Ball obj
        # This is getting very large! We must be selective.
        # Let's track the agent and the NOUN objects.
        self.tracked_objects = ["BABA", "KEY", "DOOR", "BALL", "FLAG", "IS", "WIN"]  # simplified for now
        self.num_dynamic_objects = len(self.tracked_objects)

        observation_space_size = self.num_cells ** self.num_dynamic_objects
        # ... (print statements) ...
        self.observation_space = gym.spaces.Discrete(observation_space_size)

    def observation(self, obs):
        # This needs a robust way to get the primary instance of each object
        # This is a simplification and might need adjustment for levels with multiple keys, etc.
        def get_pos_idx(obj_list):
            if obj_list and obj_list[0].pos:
                return obj_list[0].pos[1] * self.unwrapped.width + obj_list[0].pos[0]
            return 0  # Default to 0 if object doesn't exist

        indices = [
            self.unwrapped.agent_pos[1] * self.unwrapped.width + self.unwrapped.agent_pos[0],
            get_pos_idx(self.unwrapped.noun_objects.get("KEY")),
            get_pos_idx(self.unwrapped.noun_objects.get("DOOR")),
            get_pos_idx(self.unwrapped.noun_objects.get("BALL")),
            get_pos_idx(self.unwrapped.text_blocks.get("FLAG")),
            get_pos_idx(self.unwrapped.text_blocks.get("IS")),
            get_pos_idx(self.unwrapped.text_blocks.get("WIN"))
        ]

        state_idx = 0
        for idx in indices:
            state_idx = state_idx * self.num_cells + idx
        return state_idx


ACTION_VECTOR = [
    np.array((0, -1)),  # UP
    np.array((1, 0)),  # RIGHT
    np.array((0, 1)),  # DOWN
    np.array((-1, 0)),  # LEFT
]

ACTION_TO_DIR = {0: 3, 1: 0, 2: 1, 3: 2}

TEXT_TO_IMAGE = {
    'FLAG': Image.open('assets/flag.png').convert('RGBA'),
    'IS': Image.open('assets/is.png').convert('RGBA'),
    'WIN': Image.open('assets/win.png').convert('RGBA'),
    'DOOR': Image.open('assets/door.png').convert('RGBA'),
    'KEY': Image.open('assets/key.png').convert('RGBA'),
    'BALL': Image.open('assets/ball_object.png').convert('RGBA'),
    'STOP': Image.open('assets/stop.png').convert('RGBA'),
    'PUSH': Image.open('assets/push.png').convert('RGBA'),
    'DEFEAT': Image.open('assets/defeat.png').convert('RGBA'),
}

OBJ_TO_IMAGE = {

    'KEY': Image.open('assets/key_object.png').convert('RGBA'),
    'DOOR': Image.open('assets/door_object.png').convert('RGBA'),
    'BALL': Image.open('assets/ball_object.png').convert('RGBA'),
}


# --- World Object classes remain the same ---
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
        if not self.image: return
        png_pil = self.image.resize((img.shape[1], img.shape[0]), resample=Image.Resampling.LANCZOS)
        tile_pil = Image.fromarray(img)
        tile_pil.paste(png_pil, (0, 0), mask=png_pil)
        img[:] = np.array(tile_pil)


class Key(WorldObj):
    def __init__(self):
        super().__init__('key', 'yellow')
        self.image = OBJ_TO_IMAGE['KEY']
        self._pos = None

    @property
    def pos(self): return self._pos

    @pos.setter
    def pos(self, value): self._pos = value

    def render(self, img):
        if not self.image: return
        png_pil = self.image.resize((img.shape[1], img.shape[0]), resample=Image.Resampling.LANCZOS)
        tile_pil = Image.fromarray(img)
        tile_pil.paste(png_pil, (0, 0), mask=png_pil)
        img[:] = np.array(tile_pil)


class Door(WorldObj):
    def __init__(self):
        super().__init__('door', 'blue')
        self.image = OBJ_TO_IMAGE['DOOR']
        self._pos = None

    @property
    def pos(self): return self._pos

    @pos.setter
    def pos(self, value): self._pos = value

    def render(self, img):
        if not self.image: return
        png_pil = self.image.resize((img.shape[1], img.shape[0]), resample=Image.Resampling.LANCZOS)
        tile_pil = Image.fromarray(img)
        tile_pil.paste(png_pil, (0, 0), mask=png_pil)
        img[:] = np.array(tile_pil)


class Ball(WorldObj):
    def __init__(self):
        super().__init__('ball', 'grey')
        self.image = OBJ_TO_IMAGE['BALL']
        self._pos = None

    @property
    def pos(self): return self._pos

    @pos.setter
    def pos(self, value): self._pos = value

    def render(self, img):
        if not self.image: return
        png_pil = self.image.resize((img.shape[1], img.shape[0]), resample=Image.Resampling.LANCZOS)
        tile_pil = Image.fromarray(img)
        tile_pil.paste(png_pil, (0, 0), mask=png_pil)
        img[:] = np.array(tile_pil)


class BabaIsYouGridEnv(MiniGridEnv):
    def __init__(self, level_map: list[str], **kwargs):
        self.level_map = level_map
        height = len(level_map)
        width = len(level_map[0]) if height > 0 else 0

        self.text_blocks: Dict[str, List[Text]] = {}
        self.noun_objects: Dict[str, List[WorldObj]] = {}
        self.active_rules: Dict[str, Set[str]] = {}
        self._pre_move_obj = None
        mission_space = MissionSpace(mission_func=lambda: "achieve the win condition")
        super().__init__(mission_space=mission_space, width=width, height=height, **kwargs)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._check_rules()
        return obs, info

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.text_blocks.clear()
        self.noun_objects.clear()

        # --- SIMPLIFIED: No 'B' or 'b' characters ---
        char_to_noun = {'F': 'FLAG', 'D': 'DOOR', 'K': 'KEY', 'O': 'BALL'}
        char_to_text = {
            'f': "FLAG", 'i': "IS", 'w': "WIN", 'd': "DOOR", 'k': "KEY",
            's': "STOP", 'p': "PUSH", 'e': "DEFEAT", 'o': "BALL"
        }

        for y, row in enumerate(self.level_map):
            for x, char in enumerate(row):
                obj, obj_type = None, None
                if char == 'A':
                    self.agent_pos = (x, y);
                    self.agent_dir = 0
                elif char == 'W':
                    obj = Wall()
                elif char in char_to_noun:
                    noun_type = char_to_noun[char]

                    # --- THIS IS THE KEY LOGIC ---
                    # Create the native object
                    if noun_type == "FLAG":
                        obj = Goal()
                        # Allow the agent to walk onto it
                        obj.can_overlap = lambda: True
                    elif noun_type == "DOOR":
                        obj = Door()
                    elif noun_type == "KEY":
                        obj = Key()
                    elif noun_type == "BALL":
                        obj = Ball()

                    # ** DYNAMICALLY ATTACH THE NOUN IDENTITY **
                    if obj:
                        obj.noun_type = noun_type
                    # --- END OF KEY LOGIC ---

                    obj_type = noun_type

                elif char in char_to_text:
                    text_label = char_to_text[char]
                    obj = Text(text_label)
                    if text_label not in self.text_blocks: self.text_blocks[text_label] = []
                    self.text_blocks[text_label].append(obj)

                if obj:
                    obj.pos = (x, y)
                    self.put_obj(obj, x, y)
                    if obj_type:
                        if obj_type not in self.noun_objects: self.noun_objects[obj_type] = []
                        self.noun_objects[obj_type].append(obj)

        if self.agent_pos is None: raise ValueError("Level map must contain an agent 'A'.")

    def _check_rules(self):
        VALID_NOUNS = {"FLAG", "DOOR", "KEY", "BALL", "WALL"}
        VALID_PROPERTIES = {"PUSH", "STOP", "WIN", "DEFEAT"}

        # --- FIX for Identity Crisis ---
        # 1. Set default properties for NOUNS (the physical objects)
        self.active_rules = {"WALL": {"STOP"}}  # Physical WALL is STOP

        # 2. Set default properties for all TEXT blocks (they are always PUSH)
        # We store them with a 'TXT_' prefix to distinguish them
        for text_type in VALID_NOUNS.union(VALID_PROPERTIES).union({"IS"}):
            self.active_rules[f"TXT_{text_type}"] = {"PUSH"}

        if "IS" not in self.text_blocks: return

        for is_block in self.text_blocks["IS"]:
            ix, iy = is_block.pos

            # Horizontal Check
            # --- FIX for AssertionError ---
            if ix > 0 and ix < self.width - 1:
                noun_block = self.grid.get(ix - 1, iy)
                prop_block = self.grid.get(ix + 1, iy)
                if (noun_block and prop_block and
                        isinstance(noun_block, Text) and isinstance(prop_block, Text) and
                        noun_block.text_label in VALID_NOUNS and prop_block.text_label in VALID_PROPERTIES):
                    noun, prop = noun_block.text_label, prop_block.text_label
                    if noun not in self.active_rules: self.active_rules[noun] = set()
                    self.active_rules[noun].add(prop)

            # Vertical Check
            # --- FIX for AssertionError ---
            if iy > 0 and iy < self.height - 1:
                noun_block = self.grid.get(ix, iy - 1)
                prop_block = self.grid.get(ix, iy + 1)
                if (noun_block and prop_block and
                        isinstance(noun_block, Text) and isinstance(prop_block, Text) and
                        noun_block.text_label in VALID_NOUNS and prop_block.text_label in VALID_PROPERTIES):
                    noun, prop = noun_block.text_label, prop_block.text_label
                    if noun not in self.active_rules: self.active_rules[noun] = set()
                    self.active_rules[noun].add(prop)

    def _has_property(self, obj, prop):
        if obj is None: return False

        # --- FIX for Identity Crisis ---
        obj_type = ""
        if isinstance(obj, Text):
            # It's a text block, check for its text-specific rule
            obj_type = f"TXT_{obj.text_label}"
        elif hasattr(obj, 'noun_type'):
            # It's a physical object, use its noun identity
            obj_type = obj.noun_type
        else:
            # It's a basic object like a Wall
            obj_type = obj.type.upper()

        return obj_type in self.active_rules and prop in self.active_rules[obj_type]

    def step(self, action):
        self._check_rules()
        self.step_count += 1
        reward = -0.01
        terminated, truncated = False, False

        self.agent_dir = ACTION_TO_DIR[action]
        move_vec = ACTION_VECTOR[action]
        target_pos = tuple(self.agent_pos + move_vec)
        target_cell = self.grid.get(*target_pos)

        # --- CONSISTENT "SOLID BY DEFAULT" LOGIC ---

        # Case 1: The target cell is empty. The agent can always move.
        if target_cell is None:
            # We still need to handle not erasing the Goal if we are moving OFF of it.
            object_at_agent_pos = self.grid.get(*self.agent_pos)
            if not isinstance(object_at_agent_pos, Goal):
                self.grid.set(*self.agent_pos, None)
            self.agent_pos = target_pos

        # Case 2: The target cell has an object. Check its properties to see if we can interact.
        else:
            # Sub-case 2a: The object has the WIN property, allowing the agent to move onto it.
            if self._has_property(target_cell, "WIN"):
                object_at_agent_pos = self.grid.get(*self.agent_pos)
                if not isinstance(object_at_agent_pos, Goal):
                    self.grid.set(*self.agent_pos, None)
                self.agent_pos = target_pos

            # Sub-case 2b: The object has the PUSH property.
            elif self._has_property(target_cell, "PUSH"):
                # This is the robust multi-push logic we built.
                objects_to_push = []
                is_chain_pushable = True
                check_pos = target_pos
                while True:
                    if not (0 <= check_pos[0] < self.width and 0 <= check_pos[1] < self.height):
                        is_chain_pushable = False; break
                    cell = self.grid.get(*check_pos)
                    if cell is None: break
                    if self._has_property(cell, "PUSH"):
                        objects_to_push.append(cell)
                        check_pos = tuple(np.array(check_pos) + move_vec)
                    else:
                        is_chain_pushable = False; break

                if is_chain_pushable:
                    is_special_move = False
                    if len(objects_to_push) > 0 and isinstance(objects_to_push[0], Key):
                        door_pos = tuple(objects_to_push[0].pos + move_vec)
                        if isinstance(self.grid.get(*door_pos), Door):
                            self.grid.set(*objects_to_push[0].pos, None);
                            self.grid.set(*door_pos, None)
                            self.grid.set(*self.agent_pos, None);
                            self.agent_pos = target_pos
                            is_special_move = True

                    if not is_special_move:
                        for obj in reversed(objects_to_push):
                            old_obj_pos, new_obj_pos = obj.pos, tuple(obj.pos + move_vec)
                            self.grid.set(*new_obj_pos, obj);
                            self.grid.set(*old_obj_pos, None)
                            obj.pos = new_obj_pos
                        self.grid.set(*self.agent_pos, None);
                        self.agent_pos = target_pos

            # Sub-case 2c (Implicit): The object is not WIN and not PUSH. It is therefore STOP.
            # No 'else' block needed. The move is blocked.

        # --- POST-ACTION CHECKS ---
        agent_overlaps_cell = self.grid.get(*self.agent_pos)

        if self._has_property(agent_overlaps_cell, "DEFEAT"):
            reward = -50.0; truncated = True
        elif self._has_property(agent_overlaps_cell, "WIN"):
            reward = 50.0; terminated = True

        if not truncated and self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}