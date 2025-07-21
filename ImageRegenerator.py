import os
import sys
import cv2
import json

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

base_image_path = os.path.join("./idle00.png")
json_path = resource_path("pixel_mappings.json")
output_folder = resource_path("GeneratedImages")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(json_path, 'r') as f:
    mappings = json.load(f)

base_image = cv2.imread(base_image_path, cv2.IMREAD_COLOR)

if base_image is None:
    raise FileNotFoundError(f"Base image {base_image_path} not found.")

for target_image, data in mappings.items():
    pixel_mappings = data["pixel_mappings"]  # Access the pixel mappings list
    generated_image = base_image.copy()
    for mapping in pixel_mappings:
        x1, y1 = mapping["base_pixel"]
        x2, y2 = mapping["mapped_pixel"]
        if 0 <= x2 < generated_image.shape[1] and 0 <= y2 < generated_image.shape[0]:
            generated_image[y2, x2] = base_image[y1, x1]
    target_filename = os.path.basename(target_image)
    output_path = os.path.join(output_folder, target_filename)
    cv2.imwrite(output_path, generated_image)
