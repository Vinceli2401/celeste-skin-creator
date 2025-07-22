import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.models import UNet2DModel
import numpy as np
from PIL import Image
import os
import argparse
import json
from typing import List, Tuple, Optional
import cv2

class SpriteCharacterEncoder(nn.Module):

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class PoseKeypoints:

    CELESTE_KEYPOINTS = {
        'head': (16, 8),
        'torso': (16, 16),
        'left_arm': (12, 14),
        'right_arm': (20, 14),
        'left_leg': (14, 24),
        'right_leg': (18, 24),
    }

    @staticmethod
    def extract_from_sprite(sprite_array: np.ndarray) -> dict:
        if sprite_array.shape[2] == 4:
            alpha = sprite_array[:, :, 3]
            gray = np.mean(sprite_array[:, :, :3], axis=2) * (alpha / 255.0)
        else:
            gray = np.mean(sprite_array, axis=2)

        char_pixels = np.where(gray > 0.1)

        if len(char_pixels[0]) == 0:
            return PoseKeypoints.CELESTE_KEYPOINTS.copy()

        min_y, max_y = np.min(char_pixels[0]), np.max(char_pixels[0])
        min_x, max_x = np.min(char_pixels[1]), np.max(char_pixels[1])
        center_x = (min_x + max_x) // 2

        head_y = min_y + (max_y - min_y) * 0.2
        torso_y = min_y + (max_y - min_y) * 0.5
        legs_y = min_y + (max_y - min_y) * 0.8

        return {
            'head': (center_x, int(head_y)),
            'torso': (center_x, int(torso_y)),
            'left_arm': (center_x - 4, int(torso_y)),
            'right_arm': (center_x + 4, int(torso_y)),
            'left_leg': (center_x - 2, int(legs_y)),
            'right_leg': (center_x + 2, int(legs_y)),
        }

    @staticmethod
    def generate_pose_sequence(base_pose: dict, animation_type: str) -> List[dict]:
        sequences = {
            'walk': PoseKeypoints._generate_walk_sequence(base_pose),
            'run': PoseKeypoints._generate_run_sequence(base_pose),
            'jump': PoseKeypoints._generate_jump_sequence(base_pose),
            'idle': PoseKeypoints._generate_idle_sequence(base_pose),
        }

        return sequences.get(animation_type, [base_pose])

    @staticmethod
    def _generate_walk_sequence(base_pose: dict) -> List[dict]:
        frames = []
        for i in range(8):
            frame_pose = base_pose.copy()

            leg_offset = int(3 * np.sin(i * np.pi / 4))
            frame_pose['left_leg'] = (base_pose['left_leg'][0] + leg_offset,
                                    base_pose['left_leg'][1])
            frame_pose['right_leg'] = (base_pose['right_leg'][0] - leg_offset,
                                     base_pose['right_leg'][1])

            arm_offset = int(2 * np.sin(i * np.pi / 4))
            frame_pose['left_arm'] = (base_pose['left_arm'][0] + arm_offset,
                                    base_pose['left_arm'][1])
            frame_pose['right_arm'] = (base_pose['right_arm'][0] - arm_offset,
                                     base_pose['right_arm'][1])

            frames.append(frame_pose)
        return frames

    @staticmethod
    def _generate_run_sequence(base_pose: dict) -> List[dict]:
        frames = []
        for i in range(6):
            frame_pose = base_pose.copy()

            leg_offset = int(5 * np.sin(i * np.pi / 3))
            frame_pose['left_leg'] = (base_pose['left_leg'][0] + leg_offset,
                                    base_pose['left_leg'][1])
            frame_pose['right_leg'] = (base_pose['right_leg'][0] - leg_offset,
                                     base_pose['right_leg'][1])

            lean_offset = int(1 * np.sin(i * np.pi / 3))
            frame_pose['head'] = (base_pose['head'][0] + lean_offset, base_pose['head'][1])
            frame_pose['torso'] = (base_pose['torso'][0] + lean_offset, base_pose['torso'][1])

            frames.append(frame_pose)
        return frames

    @staticmethod
    def _generate_jump_sequence(base_pose: dict) -> List[dict]:
        frames = []

        crouch_pose = base_pose.copy()
        crouch_pose['torso'] = (base_pose['torso'][0], base_pose['torso'][1] + 2)
        crouch_pose['left_leg'] = (base_pose['left_leg'][0], base_pose['left_leg'][1] + 1)
        crouch_pose['right_leg'] = (base_pose['right_leg'][0], base_pose['right_leg'][1] + 1)
        frames.append(crouch_pose)

        jump_pose = base_pose.copy()
        jump_pose['head'] = (base_pose['head'][0], base_pose['head'][1] - 3)
        jump_pose['torso'] = (base_pose['torso'][0], base_pose['torso'][1] - 2)
        jump_pose['left_arm'] = (base_pose['left_arm'][0] - 2, base_pose['left_arm'][1] - 3)
        jump_pose['right_arm'] = (base_pose['right_arm'][0] + 2, base_pose['right_arm'][1] - 3)
        frames.append(jump_pose)

        frames.append(crouch_pose)
        frames.append(base_pose)

        return frames

    @staticmethod
    def _generate_idle_sequence(base_pose: dict) -> List[dict]:
        frames = []
        for i in range(16):
            frame_pose = base_pose.copy()

            breath_offset = int(0.5 * np.sin(i * np.pi / 8))
            frame_pose['torso'] = (base_pose['torso'][0], base_pose['torso'][1] + breath_offset)

            frames.append(frame_pose)
        return frames

class SmallSpriteDiffusionModel(nn.Module):

    def __init__(self, character_dim=64, pose_dim=12):
        super().__init__()

        self.character_encoder = SpriteCharacterEncoder(character_dim)

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32 * 32)
        )

        self.unet = UNet2DModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"
            ),
            class_embed_dim=character_dim,
        )

        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"
        )

    def encode_pose(self, keypoints: dict) -> torch.Tensor:
        pose_values = []
        for key in ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']:
            x, y = keypoints.get(key, (16, 16))
            pose_values.extend([x / 16.0 - 1.0, y / 16.0 - 1.0])
        return torch.tensor(pose_values, dtype=torch.float32)

    def forward(self, reference_sprite, target_pose_keypoints, num_inference_steps=50):
        device = next(self.parameters()).device

        if isinstance(reference_sprite, np.ndarray):
            reference_sprite = torch.from_numpy(reference_sprite).float()
        if len(reference_sprite.shape) == 3:
            reference_sprite = reference_sprite.unsqueeze(0)
        if reference_sprite.shape[1] != 4:
            reference_sprite = reference_sprite.permute(0, 3, 1, 2)

        reference_sprite = reference_sprite.to(device) / 255.0
        character_embedding = self.character_encoder(reference_sprite)

        pose_tensor = self.encode_pose(target_pose_keypoints).unsqueeze(0).to(device)

        self.scheduler.set_timesteps(num_inference_steps)

        shape = (1, 4, 32, 32)
        image = torch.randn(shape, device=device)

        for t in self.scheduler.timesteps:
            model_output = self.unet(
                sample=image,
                timestep=t,
                class_labels=character_embedding,
                return_dict=False
            )[0]

            image = self.scheduler.step(model_output, t, image, return_dict=False)[0]

        image = torch.clamp(image * 255, 0, 255).cpu().numpy().astype(np.uint8)
        return image[0].transpose(1, 2, 0)

def load_reference_sprite(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.NEAREST)
    return np.array(img)

def save_sprite(sprite_array: np.ndarray, path: str):
    img = Image.fromarray(sprite_array, mode="RGBA")
    img.save(path, "PNG")

def main():
    parser = argparse.ArgumentParser(description="Generate sprite animations using diffusion")
    parser.add_argument("--reference", type=str, required=True, help="Reference sprite path")
    parser.add_argument("--poses", type=str, default="walk", help="Comma-separated animation types")
    parser.add_argument("--output_dir", type=str, default="GeneratedSprites", help="Output directory")
    parser.add_argument("--model_path", type=str, help="Pretrained model path (optional)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading reference sprite: {args.reference}")
    reference_sprite = load_reference_sprite(args.reference)

    base_pose = PoseKeypoints.extract_from_sprite(reference_sprite)
    print(f"Extracted base pose: {base_pose}")

    print("Initializing diffusion model...")
    model = SmallSpriteDiffusionModel()

    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pretrained model: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    else:
        print("⚠️  No pretrained model found. Using randomly initialized model.")
        print("   For best results, train the model first or download pretrained weights.")

    model.eval()

    pose_types = [p.strip() for p in args.poses.split(",")]

    for pose_type in pose_types:
        print(f"\nGenerating {pose_type} animation...")

        pose_sequence = PoseKeypoints.generate_pose_sequence(base_pose, pose_type)

        for i, target_pose in enumerate(pose_sequence):
            output_path = os.path.join(args.output_dir, f"{pose_type}_{i:02d}.png")

            print(f"  Generating frame {i+1}/{len(pose_sequence)}: {output_path}")

            with torch.no_grad():
                generated_sprite = model(reference_sprite, target_pose)
                save_sprite(generated_sprite, output_path)

    print(f"\nGeneration complete! Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
