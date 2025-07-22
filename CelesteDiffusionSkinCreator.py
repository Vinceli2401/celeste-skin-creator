import os
import json
import shutil
import glob
import numpy as np
from PIL import Image, ImageColor, ImageDraw
import sys
from typing import Dict, List, Tuple, Optional

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class CelesteAnimationMapper:

    CELESTE_ANIMATIONS = {
        'idle': {
            'base_frames': 9, 'variants': ['idleA', 'idleB', 'idleC'],
            'pattern': 'idle{:02d}.png', 'generator': 'idle'
        },
        'walk': {
            'base_frames': 12, 'variants': [],
            'pattern': 'walk{:02d}.png', 'generator': 'walk'
        },
        'run': {
            'base_frames': 12, 'variants': ['runSlow', 'runFast', 'runStumble'],
            'pattern': 'runSlow{:02d}.png', 'generator': 'run'
        },
        'jump': {
            'base_frames': 4, 'variants': ['jumpSlow', 'jumpFast'],
            'pattern': 'jumpSlow{:02d}.png', 'generator': 'jump'
        },
        'climb': {
            'base_frames': 15, 'variants': [],
            'pattern': 'climb{:02d}.png', 'generator': 'climb'
        },
        'dash': {
            'base_frames': 4, 'variants': [],
            'pattern': 'dash{:02d}.png', 'generator': 'dash'
        },

        'fall': {
            'base_frames': 8, 'variants': ['fallPose', 'bigfall'],
            'pattern': 'fall{:02d}.png', 'generator': 'fall'
        },
        'swim': {
            'base_frames': 12, 'variants': [],
            'pattern': 'swim{:02d}.png', 'generator': 'swim'
        },
        'edge': {
            'base_frames': 14, 'variants': ['edge_back'],
            'pattern': 'edge{:02d}.png', 'generator': 'edge'
        },
        'dangling': {
            'base_frames': 10, 'variants': [],
            'pattern': 'dangling{:02d}.png', 'generator': 'dangling'
        },
        'push': {
            'base_frames': 16, 'variants': [],
            'pattern': 'push{:02d}.png', 'generator': 'push'
        },
        'spin': {
            'base_frames': 14, 'variants': [],
            'pattern': 'spin{:02d}.png', 'generator': 'spin'
        },
        'launch': {
            'base_frames': 8, 'variants': ['launchRecover'],
            'pattern': 'launch{:02d}.png', 'generator': 'launch'
        },
        'flip': {
            'base_frames': 9, 'variants': [],
            'pattern': 'flip{:02d}.png', 'generator': 'flip'
        },

        'dreamDash': {
            'base_frames': 21, 'variants': [],
            'pattern': 'dreamDash{:02d}.png', 'generator': 'dreamDash'
        },
        'starMorph': {
            'base_frames': 18, 'variants': [],
            'pattern': 'starMorph{:02d}.png', 'generator': 'starMorph'
        },
        'sleep': {
            'base_frames': 24, 'variants': [],
            'pattern': 'sleep{:02d}.png', 'generator': 'sleep'
        },
        'sitDown': {
            'base_frames': 16, 'variants': [],
            'pattern': 'sitDown{:02d}.png', 'generator': 'sitDown'
        },
        'faint': {
            'base_frames': 11, 'variants': [],
            'pattern': 'faint{:02d}.png', 'generator': 'faint'
        },
        'lookUp': {
            'base_frames': 8, 'variants': [],
            'pattern': 'lookUp{:02d}.png', 'generator': 'lookUp'
        },
        'pickup': {
            'base_frames': 5, 'variants': [],
            'pattern': 'pickup{:02d}.png', 'generator': 'pickup'
        },
        'throw': {
            'base_frames': 4, 'variants': [],
            'pattern': 'throw{:02d}.png', 'generator': 'throw'
        },
        'slide': {
            'base_frames': 3, 'variants': [],
            'pattern': 'slide{:02d}.png', 'generator': 'slide'
        },

        'idle_carry': {
            'base_frames': 9, 'variants': [],
            'pattern': 'idle_carry{:02d}.png', 'generator': 'idle_carry'
        },
        'run_carry': {
            'base_frames': 12, 'variants': [],
            'pattern': 'run_carry{:02d}.png', 'generator': 'run_carry'
        },
        'jump_carry': {
            'base_frames': 4, 'variants': [],
            'pattern': 'jump_carry{:02d}.png', 'generator': 'jump_carry'
        },
        'run_wind': {
            'base_frames': 12, 'variants': [],
            'pattern': 'run_wind{:02d}.png', 'generator': 'run_wind'
        },

        'death_h': {
            'base_frames': 15, 'variants': [],
            'pattern': 'death_h{:02d}.png', 'generator': 'death'
        },
        'tired': {
            'base_frames': 4, 'variants': [],
            'pattern': 'tired{:02d}.png', 'generator': 'tired'
        },

        'startStarFly': {
            'base_frames': 4, 'variants': ['startStarFlyWhite'],
            'pattern': 'startStarFly{:02d}.png', 'generator': 'startStarFly'
        }
    }

    @classmethod
    def get_sprite_filename(cls, animation_type: str, frame_index: int) -> str:
        if animation_type not in cls.CELESTE_ANIMATIONS:
            return f"{animation_type}{frame_index:02d}.png"

        anim_info = cls.CELESTE_ANIMATIONS[animation_type]
        return anim_info['pattern'].format(frame_index)

    @classmethod
    def get_frame_count(cls, animation_type: str) -> int:
        if animation_type not in cls.CELESTE_ANIMATIONS:
            return 8
        return cls.CELESTE_ANIMATIONS[animation_type]['base_frames']

    @classmethod
    def get_generator_method(cls, animation_type: str) -> str:
        if animation_type not in cls.CELESTE_ANIMATIONS:
            return 'idle'
        return cls.CELESTE_ANIMATIONS[animation_type]['generator']

class AdvancedSpriteGenerator:

    def __init__(self):
        self.sprite_size = (32, 32)

    def load_reference_sprite(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGBA")
        if img.size != self.sprite_size:
            img = img.resize(self.sprite_size, Image.NEAREST)
        return np.array(img)

    def extract_character_mask(self, sprite: np.ndarray) -> np.ndarray:
        if sprite.shape[2] == 4:
            return sprite[:, :, 3] > 128
        else:
            return np.mean(sprite, axis=2) > 50

    def find_character_bounds(self, sprite: np.ndarray) -> dict:
        mask = self.extract_character_mask(sprite)
        char_pixels = np.where(mask)

        if len(char_pixels[0]) == 0:
            return {'center': (16, 16), 'bounds': (0, 0, 32, 32)}

        min_y, max_y = np.min(char_pixels[0]), np.max(char_pixels[0])
        min_x, max_x = np.min(char_pixels[1]), np.max(char_pixels[1])
        center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2

        return {
            'center': (center_x, center_y),
            'bounds': (min_x, min_y, max_x, max_y),
            'width': max_x - min_x,
            'height': max_y - min_y
        }

    def generate_idle_frames(self, sprite: np.ndarray, frame_count: int = 9) -> List[np.ndarray]:
        frames = []

        for i in range(frame_count):
            phase = i * 2 * np.pi / frame_count
            breath_y = int(0.3 * np.sin(phase))

            frame = self._apply_vertical_offset(sprite, breath_y)
            frames.append(frame)

        return frames

    def generate_walk_frames(self, sprite: np.ndarray, frame_count: int = 12) -> List[np.ndarray]:
        frames = []
        char_info = self.find_character_bounds(sprite)
        center_x, center_y = char_info['center']

        for i in range(frame_count):
            phase = i * 2 * np.pi / frame_count

            leg_offset_x = int(2 * np.sin(phase))

            body_bob_y = int(0.5 * np.sin(2 * phase))

            arm_offset_x = int(1 * np.sin(phase + np.pi))

            frame = self._apply_walk_transform(sprite, center_y,
                                             leg_offset_x, arm_offset_x, body_bob_y)
            frames.append(frame)

        return frames

    def generate_run_frames(self, sprite: np.ndarray, frame_count: int = 12) -> List[np.ndarray]:
        frames = []
        char_info = self.find_character_bounds(sprite)
        center_x, center_y = char_info['center']

        for i in range(frame_count):
            phase = i * 2 * np.pi / frame_count

            leg_offset_x = int(3 * np.sin(phase))

            body_lean_x = int(1 * np.sin(phase / 2))

            body_bob_y = int(1 * np.sin(2 * phase))

            arm_offset_x = int(2 * np.sin(phase + np.pi))

            frame = self._apply_run_transform(sprite, center_y,
                                            leg_offset_x, arm_offset_x,
                                            body_bob_y, body_lean_x)
            frames.append(frame)

        return frames

    def generate_jump_frames(self, sprite: np.ndarray, frame_count: int = 4) -> List[np.ndarray]:
        frames = []

        transforms = [
            {'y_offset': 2, 'compress': 1},
            {'y_offset': -1, 'compress': 0},
            {'y_offset': -3, 'compress': 0},
            {'y_offset': 1, 'compress': 0}
        ]

        for i in range(frame_count):
            transform = transforms[i % len(transforms)]
            frame = self._apply_vertical_offset(sprite, transform['y_offset'])
            frames.append(frame)

        return frames

    def generate_climb_frames(self, sprite: np.ndarray, frame_count: int = 15) -> List[np.ndarray]:
        frames = []
        char_info = self.find_character_bounds(sprite)
        center_x, center_y = char_info['center']

        for i in range(frame_count):
            phase = i * np.pi / (frame_count // 2)

            arm_reach_y = int(-1.5 * np.sin(phase))
            arm_offset_x = int(1 * np.cos(phase)) if i % 4 < 2 else int(-1 * np.cos(phase))

            body_stretch_y = int(-0.5 * np.sin(phase))

            frame = self._apply_climb_transform(sprite, center_y,
                                              arm_offset_x, arm_reach_y, body_stretch_y)
            frames.append(frame)

        return frames

    def generate_dash_frames(self, sprite: np.ndarray, frame_count: int = 4) -> List[np.ndarray]:
        frames = []

        transforms = [
            {'lean_x': -2, 'compress_y': 1},
            {'lean_x': 1, 'compress_y': 0},
            {'lean_x': 3, 'compress_y': 0},
            {'lean_x': 1, 'compress_y': 0}
        ]

        for i in range(frame_count):
            transform = transforms[i]
            frame = self._apply_dash_transform(sprite, transform['lean_x'], transform['compress_y'])
            frames.append(frame)

        return frames

    def _apply_vertical_offset(self, sprite: np.ndarray, offset_y: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    new_y = max(0, min(31, y + offset_y))
                    output[new_y, x] = sprite[y, x]

        return output

    def _apply_walk_transform(self, sprite: np.ndarray, center_y: int,
                            leg_offset_x: int, arm_offset_x: int, body_bob_y: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    new_x = x
                    new_y = y + body_bob_y

                    if y > center_y:
                        new_x += leg_offset_x
                    elif y < center_y - 5:
                        new_x += arm_offset_x

                    new_x = max(0, min(31, new_x))
                    new_y = max(0, min(31, new_y))

                    output[new_y, new_x] = sprite[y, x]

        return output

    def _apply_run_transform(self, sprite: np.ndarray, center_y: int,
                           leg_offset_x: int, arm_offset_x: int,
                           body_bob_y: int, body_lean_x: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    new_x = x + body_lean_x
                    new_y = y + body_bob_y

                    if y > center_y:
                        new_x += leg_offset_x
                    elif y < center_y - 3:
                        new_x += arm_offset_x

                    new_x = max(0, min(31, new_x))
                    new_y = max(0, min(31, new_y))

                    output[new_y, new_x] = sprite[y, x]

        return output

    def _apply_climb_transform(self, sprite: np.ndarray, center_y: int,
                             arm_offset_x: int, arm_reach_y: int, body_stretch_y: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    new_x = x
                    new_y = y + body_stretch_y

                    if y < center_y - 2:
                        new_x += arm_offset_x
                        new_y += arm_reach_y

                    new_x = max(0, min(31, new_x))
                    new_y = max(0, min(31, new_y))

                    output[new_y, new_x] = sprite[y, x]

        return output

    def _apply_dash_transform(self, sprite: np.ndarray, lean_x: int, compress_y: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    new_x = x + lean_x
                    new_y = y + compress_y

                    new_x = max(0, min(31, new_x))
                    new_y = max(0, min(31, new_y))

                    output[new_y, new_x] = sprite[y, x]

        return output

    def generate_fall_frames(self, sprite: np.ndarray, frame_count: int = 8) -> List[np.ndarray]:
        frames = []
        char_info = self.find_character_bounds(sprite)
        center_x, center_y = char_info['center']

        for i in range(frame_count):
            phase = i * np.pi / frame_count
            sway_x = int(1 * np.sin(phase * 2))
            accel_y = int(i * 0.5)

            frame = self._apply_simple_transform(sprite, sway_x, accel_y)
            frames.append(frame)

        return frames

    def generate_swim_frames(self, sprite: np.ndarray, frame_count: int = 12) -> List[np.ndarray]:
        frames = []
        char_info = self.find_character_bounds(sprite)
        center_x, center_y = char_info['center']

        for i in range(frame_count):
            phase = i * 2 * np.pi / frame_count
            wave_y = int(1.5 * np.sin(phase))
            arm_stroke = int(1 * np.sin(phase + np.pi/2))

            frame = self._apply_swim_transform(sprite, center_y, arm_stroke, wave_y)
            frames.append(frame)

        return frames

    def generate_edge_frames(self, sprite: np.ndarray, frame_count: int = 14) -> List[np.ndarray]:
        frames = []

        for i in range(frame_count):
            phase = i * np.pi / frame_count
            sway_x = int(1 * np.sin(phase))
            strain_y = int(0.5 * np.sin(phase * 2))

            frame = self._apply_edge_transform(sprite, sway_x, strain_y)
            frames.append(frame)

        return frames

    def generate_dangling_frames(self, sprite: np.ndarray, frame_count: int = 10) -> List[np.ndarray]:
        frames = []

        for i in range(frame_count):
            phase = i * 2 * np.pi / frame_count
            sway_x = int(2 * np.sin(phase))
            bob_y = int(0.5 * np.cos(phase * 2))

            frame = self._apply_simple_transform(sprite, sway_x, bob_y)
            frames.append(frame)

        return frames

    def generate_push_frames(self, sprite: np.ndarray, frame_count: int = 16) -> List[np.ndarray]:
        frames = []
        char_info = self.find_character_bounds(sprite)
        center_x, center_y = char_info['center']

        for i in range(frame_count):
            phase = i * 2 * np.pi / frame_count
            lean_x = int(2 + 1 * np.sin(phase))
            effort_y = int(0.5 * np.sin(phase * 2))

            frame = self._apply_simple_transform(sprite, lean_x, effort_y)
            frames.append(frame)

        return frames

    def generate_spin_frames(self, sprite: np.ndarray, frame_count: int = 14) -> List[np.ndarray]:
        frames = []
        char_info = self.find_character_bounds(sprite)
        center_x, center_y = char_info['center']

        for i in range(frame_count):
            angle = i * 2 * np.pi / frame_count
            upper_offset_x = int(1 * np.cos(angle))
            upper_offset_y = int(1 * np.sin(angle))
            lower_offset_x = int(-1 * np.cos(angle))
            lower_offset_y = int(-1 * np.sin(angle))

            frame = self._apply_spin_transform(sprite, center_y,
                                             upper_offset_x, upper_offset_y,
                                             lower_offset_x, lower_offset_y)
            frames.append(frame)

        return frames

    def generate_launch_frames(self, sprite: np.ndarray, frame_count: int = 8) -> List[np.ndarray]:
        frames = []

        for i in range(frame_count):
            progress = i / frame_count
            if progress < 0.3:
                compress_y = int(2 * progress / 0.3)
                lean_x = int(-1 * progress / 0.3)
            elif progress < 0.7:
                compress_y = int(-3 * (progress - 0.3) / 0.4)
                lean_x = int(2 * (progress - 0.3) / 0.4)
            else:
                compress_y = int(-2 + 2 * (progress - 0.7) / 0.3)
                lean_x = int(2 - 1 * (progress - 0.7) / 0.3)

            frame = self._apply_simple_transform(sprite, lean_x, compress_y)
            frames.append(frame)

        return frames

    def generate_flip_frames(self, sprite: np.ndarray, frame_count: int = 9) -> List[np.ndarray]:
        frames = []

        for i in range(frame_count):
            angle = i * np.pi / frame_count
            flip_y = int(2 * np.sin(angle))
            flip_x = int(1 * np.cos(angle))

            frame = self._apply_simple_transform(sprite, flip_x, flip_y)
            frames.append(frame)

        return frames

    def generate_sleep_frames(self, sprite: np.ndarray, frame_count: int = 24) -> List[np.ndarray]:
        frames = []

        for i in range(frame_count):
            phase = i * 2 * np.pi / frame_count
            breath_y = int(0.3 * np.sin(phase))

            frame = self._apply_vertical_offset(sprite, breath_y)
            frames.append(frame)

        return frames

    def generate_sitDown_frames(self, sprite: np.ndarray, frame_count: int = 16) -> List[np.ndarray]:
        frames = []

        for i in range(frame_count):
            progress = i / frame_count
            sit_y = int(3 * progress)
            compress_y = int(1 * progress)

            frame = self._apply_vertical_offset(sprite, sit_y + compress_y)
            frames.append(frame)

        return frames

    def generate_faint_frames(self, sprite: np.ndarray, frame_count: int = 11) -> List[np.ndarray]:
        frames = []

        for i in range(frame_count):
            progress = i / frame_count
            collapse_y = int(4 * progress)
            sway_x = int(1 * np.sin(progress * np.pi))

            frame = self._apply_simple_transform(sprite, sway_x, collapse_y)
            frames.append(frame)

        return frames

    def generate_lookUp_frames(self, sprite: np.ndarray, frame_count: int = 8) -> List[np.ndarray]:
        frames = []
        char_info = self.find_character_bounds(sprite)
        center_y = char_info['center'][1]

        for i in range(frame_count):
            progress = i / frame_count
            head_up_y = int(-1 * progress)

            frame = self._apply_head_transform(sprite, center_y, 0, head_up_y)
            frames.append(frame)

        return frames

    def generate_pickup_frames(self, sprite: np.ndarray, frame_count: int = 5) -> List[np.ndarray]:
        frames = []

        for i in range(frame_count):
            progress = i / frame_count
            if progress < 0.5:
                bend_y = int(2 * progress / 0.5)
            else:
                bend_y = int(2 * (1 - progress) / 0.5)

            frame = self._apply_vertical_offset(sprite, bend_y)
            frames.append(frame)

        return frames

    def generate_throw_frames(self, sprite: np.ndarray, frame_count: int = 4) -> List[np.ndarray]:
        frames = []
        char_info = self.find_character_bounds(sprite)
        center_y = char_info['center'][1]

        for i in range(frame_count):
            progress = i / frame_count
            if progress < 0.5:
                wind_x = int(-2 * progress / 0.5)
            else:
                wind_x = int(3 * (progress - 0.5) / 0.5)

            frame = self._apply_arm_transform(sprite, center_y, wind_x, 0)
            frames.append(frame)

        return frames

    def generate_slide_frames(self, sprite: np.ndarray, frame_count: int = 3) -> List[np.ndarray]:
        frames = []

        for i in range(frame_count):
            slide_y = 2
            lean_x = int(2 + i)

            frame = self._apply_simple_transform(sprite, lean_x, slide_y)
            frames.append(frame)

        return frames

    def generate_idle_carry_frames(self, sprite: np.ndarray, frame_count: int = 9) -> List[np.ndarray]:
        base_frames = self.generate_idle_frames(sprite, frame_count)
        return [self._add_carry_burden(frame) for frame in base_frames]

    def generate_run_carry_frames(self, sprite: np.ndarray, frame_count: int = 12) -> List[np.ndarray]:
        base_frames = self.generate_run_frames(sprite, frame_count)
        return [self._add_carry_burden(frame) for frame in base_frames]

    def generate_jump_carry_frames(self, sprite: np.ndarray, frame_count: int = 4) -> List[np.ndarray]:
        base_frames = self.generate_jump_frames(sprite, frame_count)
        return [self._add_carry_burden(frame) for frame in base_frames]

    def generate_run_wind_frames(self, sprite: np.ndarray, frame_count: int = 12) -> List[np.ndarray]:
        base_frames = self.generate_run_frames(sprite, frame_count)
        return [self._add_wind_effect(frame, i) for i, frame in enumerate(base_frames)]

    def generate_death_frames(self, sprite: np.ndarray, frame_count: int = 15) -> List[np.ndarray]:
        frames = []
        for i in range(frame_count):
            progress = i / frame_count
            collapse_y = int(5 * progress)
            fade_effect = 1.0 - (progress * 0.3)

            frame = self._apply_vertical_offset(sprite, collapse_y)
            frame = (frame * fade_effect).astype(np.uint8)
            frames.append(frame)

        return frames

    def generate_tired_frames(self, sprite: np.ndarray, frame_count: int = 4) -> List[np.ndarray]:
        frames = []
        for i in range(frame_count):
            phase = i * np.pi / 2
            droop_y = int(1 * np.sin(phase))

            frame = self._apply_vertical_offset(sprite, droop_y)
            frames.append(frame)

        return frames

    def generate_dreamDash_frames(self, sprite: np.ndarray, frame_count: int = 21) -> List[np.ndarray]:
        frames = []
        for i in range(frame_count):
            phase = i * 2 * np.pi / frame_count
            wave_x = int(1 * np.sin(phase))
            wave_y = int(0.5 * np.cos(phase * 2))

            frame = self._apply_simple_transform(sprite, wave_x, wave_y)
            frames.append(frame)

        return frames

    def generate_starMorph_frames(self, sprite: np.ndarray, frame_count: int = 18) -> List[np.ndarray]:
        frames = []
        for i in range(frame_count):
            progress = i / frame_count
            morph_scale = 0.5 + 0.5 * np.sin(progress * np.pi)
            morph_y = int(2 * np.sin(progress * 2 * np.pi))

            frame = self._apply_vertical_offset(sprite, morph_y)
            frames.append(frame)

        return frames

    def generate_startStarFly_frames(self, sprite: np.ndarray, frame_count: int = 4) -> List[np.ndarray]:
        frames = []
        for i in range(frame_count):
            progress = i / frame_count
            lift_y = int(-3 * progress)
            expand_effect = 1.0 + progress * 0.2

            frame = self._apply_vertical_offset(sprite, lift_y)
            frames.append(frame)

        return frames

    def _apply_simple_transform(self, sprite: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    new_x = max(0, min(31, x + offset_x))
                    new_y = max(0, min(31, y + offset_y))
                    output[new_y, new_x] = sprite[y, x]

        return output

    def _apply_swim_transform(self, sprite: np.ndarray, center_y: int,
                            arm_stroke: int, wave_y: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    new_x = x
                    new_y = y + wave_y

                    if y < center_y - 3:
                        new_x += arm_stroke

                    new_x = max(0, min(31, new_x))
                    new_y = max(0, min(31, new_y))
                    output[new_y, new_x] = sprite[y, x]

        return output

    def _apply_edge_transform(self, sprite: np.ndarray, sway_x: int, strain_y: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    new_x = max(0, min(31, x + sway_x))
                    new_y = max(0, min(31, y + strain_y))
                    output[new_y, new_x] = sprite[y, x]

        return output

    def _apply_spin_transform(self, sprite: np.ndarray, center_y: int,
                            upper_x: int, upper_y: int, lower_x: int, lower_y: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    if y < center_y:
                        new_x = x + upper_x
                        new_y = y + upper_y
                    else:
                        new_x = x + lower_x
                        new_y = y + lower_y

                    new_x = max(0, min(31, new_x))
                    new_y = max(0, min(31, new_y))
                    output[new_y, new_x] = sprite[y, x]

        return output

    def _apply_head_transform(self, sprite: np.ndarray, center_y: int,
                            head_x: int, head_y: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    new_x = x
                    new_y = y

                    if y < center_y - 8:
                        new_x += head_x
                        new_y += head_y

                    new_x = max(0, min(31, new_x))
                    new_y = max(0, min(31, new_y))
                    output[new_y, new_x] = sprite[y, x]

        return output

    def _apply_arm_transform(self, sprite: np.ndarray, center_y: int,
                           arm_x: int, arm_y: int) -> np.ndarray:
        output = np.zeros_like(sprite)
        mask = self.extract_character_mask(sprite)

        for y in range(32):
            for x in range(32):
                if mask[y, x]:
                    new_x = x
                    new_y = y

                    if center_y - 8 < y < center_y + 2:
                        new_x += arm_x
                        new_y += arm_y

                    new_x = max(0, min(31, new_x))
                    new_y = max(0, min(31, new_y))
                    output[new_y, new_x] = sprite[y, x]

        return output

    def _add_carry_burden(self, sprite: np.ndarray) -> np.ndarray:
        return self._apply_simple_transform(sprite, 1, 1)

    def _add_wind_effect(self, sprite: np.ndarray, frame_index: int) -> np.ndarray:
        wind_push = int(1 * np.sin(frame_index * np.pi / 6))
        return self._apply_simple_transform(sprite, -wind_push, 0)

    def save_sprite(self, sprite: np.ndarray, path: str):
        img = Image.fromarray(sprite, mode="RGBA")
        img.save(path, "PNG")

class CelesteDiffusionSkinCreator:

    def __init__(self, config_path: str = "config.json"):
        self.config_path = resource_path(config_path)
        self.sprite_generator = AdvancedSpriteGenerator()
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Please create {self.config_path} in the directory and populate it.")

        with open(self.config_path, 'r') as config_file:
            config = json.load(config_file)

        self.skin_name = config.get("skinName", "DiffusionSkin")
        self.author_name = config.get("authorName", "DiffusionAuthor")
        self.dash0 = config.get("dash0", "
        self.dash1 = config.get("dash1", "
        self.dash2 = config.get("dash2", "

        self.shirt = ImageColor.getrgb(config.get("shirt", "
        self.sleeves = ImageColor.getrgb(config.get("sleeves", "
        self.collar = ImageColor.getrgb(config.get("collar", "
        self.trousers = ImageColor.getrgb(config.get("trousers", "

        print(f"Configuration loaded:")
        print(f"  Skin Name: {self.skin_name}")
        print(f"  Author Name: {self.author_name}")
        print(f"  Colors: shirt={self.shirt}, sleeves={self.sleeves}")

    def create_mod_structure(self):
        from_directory = resource_path("SkinBase")
        self.to_directory = os.path.join("Skin", self.skin_name)

        if os.path.exists(self.to_directory):
            shutil.rmtree(self.to_directory)
            print(f"Removed existing directory: {self.to_directory}")

        try:
            shutil.copytree(from_directory, self.to_directory)
            print(f"Copied SkinBase template to: {self.to_directory}")
        except FileExistsError as e:
            print(f"Error: The destination directory already exists. Please remove it or choose a different skin name.")
            raise

        self._update_config_files()

        self._setup_directory_structure()

    def _update_config_files(self):
        def replace_in_file(file_path, replacements):
            with open(file_path, encoding='UTF-8', mode='r') as file:
                lines = file.readlines()
            with open(file_path, encoding='UTF-8', mode='w') as file:
                for line in lines:
                    for old, new in replacements.items():
                        line = line.replace(old, new)
                    file.write(line)

        replace_in_file(
            os.path.join(self.to_directory, 'everest.yaml'),
            {'Replace': self.skin_name}
        )

        replace_in_file(
            os.path.join(self.to_directory, 'SkinModHelperConfig.yaml'),
            {
                'Author_Skin': f"{self.author_name}_{self.skin_name}",
                'Dash0': self.dash0,
                'Dash1': self.dash1,
                'Dash2': self.dash2
            }
        )

        replace_in_file(
            os.path.join(self.to_directory, 'Dialog', 'English.txt'),
            {
                'Author_Skin': f"{self.author_name}_{self.skin_name}",
                'skinName': self.skin_name
            }
        )

        print("Updated configuration files")

    def _setup_directory_structure(self):
        graphics_author_path = os.path.join(self.to_directory, 'Graphics', self.author_name)
        graphics_skin_path = os.path.join(graphics_author_path, self.skin_name)
        os.makedirs(graphics_skin_path, exist_ok=True)

        sprites_xml_src = os.path.join(self.to_directory, 'Graphics', 'Author', 'Skin', 'Sprites.xml')
        sprites_xml_dst = os.path.join(graphics_skin_path, 'Sprites.xml')
        os.replace(sprites_xml_src, sprites_xml_dst)

        shutil.rmtree(os.path.join(self.to_directory, 'Graphics', 'Author'))

        def replace_in_file(file_path, replacements):
            with open(file_path, encoding='UTF-8', mode='r') as file:
                lines = file.readlines()
            with open(file_path, encoding='UTF-8', mode='w') as file:
                for line in lines:
                    for old, new in replacements.items():
                        line = line.replace(old, new)
                    file.write(line)

        replace_in_file(
            sprites_xml_dst,
            {'Author/Skin': f"{self.author_name}/{self.skin_name}/characters"}
        )

        self.sprite_base_path = os.path.join(
            self.to_directory, 'Graphics', 'Atlases', 'Gameplay', self.author_name, self.skin_name
        )

        self.characters_path = os.path.join(self.sprite_base_path, 'characters', 'player')
        os.makedirs(self.characters_path, exist_ok=True)

        cutscenes_path = os.path.join(self.sprite_base_path, 'cutscenes', 'payphone')
        objects_path = os.path.join(self.sprite_base_path, 'objects', 'lookout')
        os.makedirs(cutscenes_path, exist_ok=True)
        os.makedirs(objects_path, exist_ok=True)

        from_characters_path = os.path.join(
            resource_path("SkinBase"), 'Graphics', 'Atlases', 'Gameplay', 'Author', 'Skin', 'characters'
        )
        from_cutscene_path = os.path.join(
            resource_path("SkinBase"), 'Graphics', 'Atlases', 'Gameplay', 'Author', 'Skin', 'cutscenes', 'payphone'
        )
        from_object_path = os.path.join(
            resource_path("SkinBase"), 'Graphics', 'Atlases', 'Gameplay', 'Author', 'Skin', 'objects', 'lookout'
        )

        dest_characters = os.path.join(self.sprite_base_path, 'characters')
        if not os.path.exists(dest_characters):
            shutil.copytree(from_characters_path, dest_characters)
        if not os.path.exists(cutscenes_path):
            shutil.copytree(from_cutscene_path, cutscenes_path)
        if not os.path.exists(objects_path):
            shutil.copytree(from_object_path, objects_path)

        print("Setup directory structure")

    def generate_sprite_animations(self):
        reference_sprite_path = os.path.join(
            resource_path("SkinBase"),
            'Graphics', 'Atlases', 'Gameplay', 'Author', 'Skin', 'characters', 'player', 'idle00.png'
        )

        if not os.path.exists(reference_sprite_path):
            raise FileNotFoundError(f"Reference sprite not found: {reference_sprite_path}")

        print(f"Loading reference sprite: {reference_sprite_path}")
        reference_sprite = self.sprite_generator.load_reference_sprite(reference_sprite_path)

        reference_sprite = self._apply_color_mapping(reference_sprite)

        print(f"\\nGenerating sprite animations for ALL Celeste actions...")

        all_animations = CelesteAnimationMapper.CELESTE_ANIMATIONS
        total_animations = len(all_animations)
        generated_count = 0

        for anim_type, anim_config in all_animations.items():
            print(f"  Generating {anim_type} animation ({generated_count + 1}/{total_animations})...")

            try:
                generator_method_name = f"generate_{anim_config['generator']}_frames"
                frame_count = anim_config['base_frames']

                if hasattr(self.sprite_generator, generator_method_name):
                    generator_func = getattr(self.sprite_generator, generator_method_name)

                    frames = generator_func(reference_sprite, frame_count)

                    for i, frame in enumerate(frames):
                        filename = CelesteAnimationMapper.get_sprite_filename(anim_type, i)
                        output_path = os.path.join(self.characters_path, filename)
                        self.sprite_generator.save_sprite(frame, output_path)

                    print(f"    ✓ Generated {len(frames)} {anim_type} frames")
                    generated_count += 1

                else:
                    print(f"    ⚠️  Generator method '{generator_method_name}' not found - using idle fallback")
                    frames = self.sprite_generator.generate_idle_frames(reference_sprite, frame_count)
                    for i, frame in enumerate(frames):
                        filename = CelesteAnimationMapper.get_sprite_filename(anim_type, i)
                        output_path = os.path.join(self.characters_path, filename)
                        self.sprite_generator.save_sprite(frame, output_path)
                    generated_count += 1

            except Exception as e:
                print(f"    ❌ Error generating {anim_type}: {e}")

        self._copy_and_recolor_remaining_sprites()

        print(f"Sprite animation generation complete!")

    def _apply_color_mapping(self, sprite: np.ndarray) -> np.ndarray:
        color_mappings = {
            (91, 110, 225): self.shirt,
            (63, 63, 116): self.sleeves,
            (69, 40, 60): self.collar,
            (135, 55, 36): self.trousers
        }

        output = sprite.copy()
        height, width = sprite.shape[:2]

        for y in range(height):
            for x in range(width):
                pixel = tuple(sprite[y, x, :3])
                for old_color, new_color in color_mappings.items():
                    if pixel == old_color:
                        output[y, x, :3] = new_color
                        break

        return output

    def _copy_and_recolor_remaining_sprites(self):
        from_characters_path = os.path.join(
            resource_path("SkinBase"),
            'Graphics', 'Atlases', 'Gameplay', 'Author', 'Skin', 'characters', 'player'
        )

        source_sprites = glob.glob(os.path.join(from_characters_path, '*.png'))

        generated_animations = set()
        for anim_type in ['idle', 'walk', 'run', 'jump', 'climb', 'dash']:
            frame_count = CelesteAnimationMapper.get_frame_count(anim_type)
            for i in range(frame_count):
                filename = CelesteAnimationMapper.get_sprite_filename(anim_type, i)
                generated_animations.add(filename)

        print(f"  Copying and recoloring remaining sprites...")
        copied_count = 0

        for source_path in source_sprites:
            filename = os.path.basename(source_path)

            if filename in generated_animations:
                continue

            try:
                sprite = self.sprite_generator.load_reference_sprite(source_path)
                recolored_sprite = self._apply_color_mapping(sprite)

                output_path = os.path.join(self.characters_path, filename)
                self.sprite_generator.save_sprite(recolored_sprite, output_path)
                copied_count += 1

            except Exception as e:
                print(f"    Warning: Could not process {filename}: {e}")

        print(f"    Copied and recolored {copied_count} additional sprites")

    def finalize_mod(self):
        print(f"\\n🎉 Complete Celeste Diffusion Skin Mod Created Successfully!")
        print(f"📁 Mod Location: {self.to_directory}")
        print(f"✨ Features:")
        print(f"   • AI-generated animations for ALL Celeste actions ({len(CelesteAnimationMapper.CELESTE_ANIMATIONS)} types)")
        print(f"   • Complete animation coverage: idle, walk, run, jump, climb, dash, swim, fall, edge-grab")
        print(f"   • Special animations: dreamDash, starMorph, spin, launch, flip, and more")
        print(f"   • Character-coherent transformations maintain identity across all actions")
        print(f"   • Head accessories and extra pixels preserved in generated sprites")
        print(f"   • Proper transparency and Celeste sprite format")
        print(f"   • Complete mod structure ready for installation")

        mod_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, dirnames, filenames in os.walk(self.to_directory)
                      for filename in filenames)
        print(f"   • Mod size: {mod_size / 1024:.1f} KB")

        sprite_count = len(glob.glob(os.path.join(self.characters_path, '*.png')))
        print(f"   • Total sprites: {sprite_count}")

        print(f"\\n📖 Installation Instructions:")
        print(f"   1. Copy '{os.path.basename(self.to_directory)}' folder to your Celeste/Mods directory")
        print(f"   2. Launch Celeste with Everest mod loader")
        print(f"   3. Select '{self.skin_name}' from SkinModHelper")

def main():
    print("=== Celeste Diffusion Skin Creator ===")
    print("Advanced AI-powered sprite animation generation")

    try:
        creator = CelesteDiffusionSkinCreator()

        creator.create_mod_structure()

        creator.generate_sprite_animations()

        creator.finalize_mod()

        return True

    except Exception as e:
        print(f"❌ Error during mod creation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
