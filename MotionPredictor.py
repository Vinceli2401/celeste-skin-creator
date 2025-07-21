import numpy as np
import cv2
import os
import glob
import json

def resource_path(relative_path):
    return os.path.join(os.path.abspath("."), relative_path)

from_directory = resource_path("SkinBase")
from_characters_path = os.path.join(
    from_directory, 'Graphics', 'Atlases', 'Gameplay', 'Author', 'Skin', 'characters', 'player'
)
base_image_path = os.path.join(from_characters_path, "idle00.png")

base_image = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)
if base_image is None:
    raise FileNotFoundError(f"Base image {base_image_path} not found.")

all_images = glob.glob(os.path.join(from_characters_path, '**', '*.png'), recursive=True)
mappings = {}

for target_image_path in all_images:
    if os.path.abspath(target_image_path) == os.path.abspath(base_image_path):
        continue

    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if target_image is None or base_image.shape != target_image.shape:
        continue

    feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=5, blockSize=5)
    base_points = cv2.goodFeaturesToTrack(base_image, mask=None, **feature_params)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    target_points, st, err = cv2.calcOpticalFlowPyrLK(base_image, target_image, base_points, None, **lk_params)

    valid_base_points = base_points[st == 1]
    valid_target_points = target_points[st == 1]

    if len(valid_base_points) < 3 or len(valid_target_points) < 3:
        print(f"Insufficient feature points for {target_image_path}. Skipping.")
        continue

    base_points = np.array(valid_base_points, dtype=np.float32)
    target_points = np.array(valid_target_points, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(base_points, target_points)

    if matrix is None:
        print(f"Affine transformation failed for {target_image_path}. Skipping.")
        continue

    matrix = np.array(matrix, dtype=np.float32)
    warped_base = cv2.warpAffine(base_image, matrix, (target_image.shape[1], target_image.shape[0]))

    flow = cv2.calcOpticalFlowFarneback(warped_base, target_image, None, 0.5, 3, 21, 3, 5, 1.2, 0)

    pixel_mappings = []
    difference = cv2.absdiff(warped_base, target_image)
    changed_pixels = np.argwhere(difference > 0)

    for changed_pixel_position in changed_pixels:
        y1, x1 = map(int, changed_pixel_position)
        dx, dy = flow[y1, x1]
        x2 = max(0, min(target_image.shape[1] - 1, int(x1 + dx)))
        y2 = max(0, min(target_image.shape[0] - 1, int(y1 + dy)))
        pixel_mappings.append({
            "base_pixel": [x1, y1],
            "mapped_pixel": [x2, y2]
        })

    mappings[os.path.relpath(target_image_path, from_directory)] = {
        "pixel_mappings": pixel_mappings,
        "num_pixels": len(pixel_mappings)
    }

output_path = resource_path("pixel_mappings.json")
with open(output_path, 'w') as f:
    json.dump(mappings, f, indent=4)

print(f"Pixel mappings saved to {output_path}")
