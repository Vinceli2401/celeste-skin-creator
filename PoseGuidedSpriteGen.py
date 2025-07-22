import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from typing import Dict, List, Tuple
import json

class CharacterSegmenter:

    def __init__(self):
        self.part_colors = {
            'head': (255, 100, 100),
            'torso': (100, 255, 100),
            'left_arm': (100, 100, 255),
            'right_arm': (255, 255, 100),
            'left_leg': (255, 100, 255),
            'right_leg': (100, 255, 255),
            'background': (0, 0, 0)
        }

    def segment_character(self, sprite: np.ndarray) -> Dict[str, np.ndarray]:
        if sprite.shape[2] == 4:
            alpha = sprite[:, :, 3]
            rgb = sprite[:, :, :3]
            char_mask = alpha > 128
        else:
            char_mask = np.mean(sprite, axis=2) > 50

        h, w = char_mask.shape
        segments = {}

        char_pixels = np.where(char_mask)
        if len(char_pixels[0]) == 0:
            return {part: np.zeros_like(char_mask) for part in self.part_colors.keys()}

        min_y, max_y = np.min(char_pixels[0]), np.max(char_pixels[0])
        min_x, max_x = np.min(char_pixels[1]), np.max(char_pixels[1])
        center_x = (min_x + max_x) // 2
        char_height = max_y - min_y

        head_mask = np.zeros_like(char_mask)
        head_top = min_y
        head_bottom = min_y + int(char_height * 0.3)
        head_mask[head_top:head_bottom, min_x:max_x] = char_mask[head_top:head_bottom, min_x:max_x]
        segments['head'] = head_mask

        torso_mask = np.zeros_like(char_mask)
        torso_top = head_bottom
        torso_bottom = min_y + int(char_height * 0.7)
        torso_mask[torso_top:torso_bottom, min_x:max_x] = char_mask[torso_top:torso_bottom, min_x:max_x]
        segments['torso'] = torso_mask

        arm_mask_left = torso_mask.copy()
        arm_mask_left[:, center_x:] = False
        segments['left_arm'] = arm_mask_left

        arm_mask_right = torso_mask.copy()
        arm_mask_right[:, :center_x] = False
        segments['right_arm'] = arm_mask_right

        leg_top = torso_bottom
        leg_bottom = max_y

        leg_mask_left = np.zeros_like(char_mask)
        leg_mask_left[leg_top:leg_bottom, min_x:center_x] = char_mask[leg_top:leg_bottom, min_x:center_x]
        segments['left_leg'] = leg_mask_left

        leg_mask_right = np.zeros_like(char_mask)
        leg_mask_right[leg_top:leg_bottom, center_x:max_x] = char_mask[leg_top:leg_bottom, center_x:max_x]
        segments['right_leg'] = leg_mask_right

        return segments

    def visualize_segmentation(self, sprite: np.ndarray, segments: Dict[str, np.ndarray], output_path: str):
        h, w = sprite.shape[:2]
        vis_image = np.zeros((h, w, 3), dtype=np.uint8)

        for part_name, mask in segments.items():
            if part_name in self.part_colors:
                color = self.part_colors[part_name]
                vis_image[mask] = color

        Image.fromarray(vis_image).save(output_path)

class PoseTransformer:

    def __init__(self):
        self.segmenter = CharacterSegmenter()

    def apply_pose_transformation(self, sprite: np.ndarray, source_pose: Dict, target_pose: Dict) -> np.ndarray:

        segments = self.segmenter.segment_character(sprite)

        if sprite.shape[2] == 4:
            output = np.zeros_like(sprite)
        else:
            output = np.zeros((32, 32, 4), dtype=np.uint8)

        for part_name, mask in segments.items():
            if part_name == 'background' or not np.any(mask):
                continue

            if part_name in source_pose and part_name in target_pose:
                src_pos = source_pose[part_name]
                tgt_pos = target_pose[part_name]

                dx = tgt_pos[0] - src_pos[0]
                dy = tgt_pos[1] - src_pos[1]

                transformed_part = self._transform_part(sprite, mask, dx, dy)

                part_mask = transformed_part[:, :, 3] > 0
                output[part_mask] = transformed_part[part_mask]

        return output

    def _transform_part(self, sprite: np.ndarray, mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
        h, w = mask.shape

        transform_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        if sprite.shape[2] == 4:
            part_sprite = sprite.copy()
            part_sprite[~mask] = [0, 0, 0, 0]
        else:
            part_sprite = np.zeros((h, w, 4), dtype=np.uint8)
            part_sprite[:, :, :3] = sprite
            part_sprite[:, :, 3] = (mask * 255).astype(np.uint8)

        transformed = cv2.warpAffine(part_sprite, transform_matrix, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))

        return transformed

class SpriteAnimationGenerator:

    def __init__(self):
        self.pose_transformer = PoseTransformer()

    def generate_animation_sequence(self, reference_sprite_path: str,
                                  animation_type: str,
                                  output_dir: str) -> List[str]:

        sprite = self.load_sprite(reference_sprite_path)

        base_pose = self._extract_simple_pose(sprite)

        from SmallSpriteDiffusion import PoseKeypoints
        pose_sequence = PoseKeypoints.generate_pose_sequence(base_pose, animation_type)

        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for i, target_pose in enumerate(pose_sequence):
            transformed_sprite = self.pose_transformer.apply_pose_transformation(
                sprite, base_pose, target_pose
            )

            output_path = os.path.join(output_dir, f"{animation_type}_pose_{i:02d}.png")
            self.save_sprite(transformed_sprite, output_path)
            output_paths.append(output_path)

        return output_paths

    def load_sprite(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGBA")
        if img.size != (32, 32):
            img = img.resize((32, 32), Image.NEAREST)
        return np.array(img)

    def save_sprite(self, sprite: np.ndarray, path: str):
        if sprite.shape[2] == 4:
            img = Image.fromarray(sprite, mode="RGBA")
        else:
            img = Image.fromarray(sprite, mode="RGB")
        img.save(path, "PNG")

    def _extract_simple_pose(self, sprite: np.ndarray) -> Dict:
        from SmallSpriteDiffusion import PoseKeypoints
        return PoseKeypoints.extract_from_sprite(sprite)

    def create_sprite_sheet(self, animation_frames: List[str], output_path: str,
                          frames_per_row: int = 8):

        if not animation_frames:
            return

        first_frame = Image.open(animation_frames[0]).convert("RGBA")
        frame_w, frame_h = first_frame.size

        num_frames = len(animation_frames)
        rows = (num_frames + frames_per_row - 1) // frames_per_row
        sheet_w = frames_per_row * frame_w
        sheet_h = rows * frame_h

        sprite_sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

        for i, frame_path in enumerate(animation_frames):
            frame = Image.open(frame_path).convert("RGBA")

            row = i // frames_per_row
            col = i % frames_per_row

            x = col * frame_w
            y = row * frame_h

            sprite_sheet.paste(frame, (x, y))

        sprite_sheet.save(output_path, "PNG")
        print(f"Sprite sheet saved: {output_path}")

def main():

    reference_sprite = "SkinBase/Graphics/Atlases/Gameplay/Author/Skin/characters/player/idle00.png"

    if not os.path.exists(reference_sprite):
        print(f"Reference sprite not found: {reference_sprite}")
        print("Please run this from the celeste-skin-creator directory")
        return

    generator = SpriteAnimationGenerator()

    sprite = generator.load_sprite(reference_sprite)
    segments = generator.pose_transformer.segmenter.segment_character(sprite)

    os.makedirs("PoseGuidedOutput", exist_ok=True)
    generator.pose_transformer.segmenter.visualize_segmentation(
        sprite, segments, "PoseGuidedOutput/segmentation_test.png"
    )
    print("Character segmentation visualization saved: PoseGuidedOutput/segmentation_test.png")

    print("Generating walk animation...")
    animation_frames = generator.generate_animation_sequence(
        reference_sprite, "walk", "PoseGuidedOutput/walk"
    )

    generator.create_sprite_sheet(
        animation_frames, "PoseGuidedOutput/walk_sprite_sheet.png"
    )

    print("Pose-guided sprite generation complete!")
    print("Check PoseGuidedOutput/ directory for results")

if __name__ == "__main__":
    main()
