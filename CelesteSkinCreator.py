import glob
import os
import json
import shutil
from PIL import Image, ImageColor
import sys

# Helper function to handle PyInstaller's dynamic paths
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Load config.json
config_file_path = resource_path('config.json')
if not os.path.exists(config_file_path):
    raise FileNotFoundError(f"Please create config.json in the directory and populate it.")

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

skinName = config.get("skinName", "default_skin")
authorName = config.get("authorName", "default_author")
dash0 = config.get("dash0", "#FFFFFF")
dash1 = config.get("dash1", "#FFFFFF")
dash2 = config.get("dash2", "#FFFFFF")
shirt = ImageColor.getrgb(config.get("shirt", "#FFFFFF"))
sleeves = ImageColor.getrgb(config.get("sleeves", "#FFFFFF"))
collar = ImageColor.getrgb(config.get("collar", "#FFFFFF"))
trousers = ImageColor.getrgb(config.get("trousers", "#FFFFFF"))

# Debug
print(f"Skin Name: {skinName}")
print(f"Author Name: {authorName}")
print(f"Dash Colors: {dash0}, {dash1}, {dash2}")
print(f"Shirt Color: {shirt}")
print(f"Sleeves Color: {sleeves}")
print(f"Collar Color: {collar}")
print(f"Trousers Color: {trousers}")

# From skinbase to skin
from_directory = resource_path("SkinBase")
to_directory = os.path.join("Skin", skinName)

# If you run it default too many times this will happen. Change config.json.
try:
    shutil.copytree(from_directory, to_directory)
except FileExistsError as e:
    print(f"Error: The destination directory already exists. Please remove it or choose a different skin name.")
    raise

def replace_in_file(file_path, replacements):
    with open(file_path, encoding='UTF-8', mode='r') as file:
        lines = file.readlines()
    with open(file_path, encoding='UTF-8', mode='w') as file:
        for line in lines:
            for old, new in replacements.items():
                line = line.replace(old, new)
            file.write(line)

replace_in_file(
    os.path.join(to_directory, 'everest.yaml'),
    {'Replace': skinName}
)

replace_in_file(
    os.path.join(to_directory, 'SkinModHelperConfig.yaml'),
    {
        'Author_Skin': f"{authorName}_{skinName}",
        'Dash0': dash0,
        'Dash1': dash1,
        'Dash2': dash2
    }
)

graphics_author_path = os.path.join(to_directory, 'Graphics', authorName)
graphics_skin_path = os.path.join(graphics_author_path, skinName)
os.makedirs(graphics_skin_path, exist_ok=True)

os.replace(
    os.path.join(to_directory, 'Graphics', 'Author', 'Skin', 'Sprites.xml'),
    os.path.join(graphics_skin_path, 'Sprites.xml')
)
shutil.rmtree(os.path.join(to_directory, 'Graphics', 'Author'))

replace_in_file(
    os.path.join(graphics_skin_path, 'Sprites.xml'),
    {'Author/Skin': f"{authorName}/{skinName}/characters"}
)

replace_in_file(
    os.path.join(to_directory, 'Dialog', 'English.txt'),
    {
        'Author_Skin': f"{authorName}_{skinName}",
        'skinName': skinName
    }
)

from_characters_path = os.path.join(
    from_directory, 'Graphics', 'Atlases', 'Gameplay', 'Author', 'Skin', 'characters'
)
to_characters_path = os.path.join(
    to_directory, 'Graphics', 'Atlases', 'Gameplay', authorName, skinName, 'characters'
)

from_cutscene_path = os.path.join(
    from_directory, 'Graphics', 'Atlases', 'Gameplay', 'Author', 'Skin', 'cutscenes', 'payphone'
)
to_cutscene_path = os.path.join(
    to_directory, 'Graphics', 'Atlases', 'Gameplay', authorName, skinName, 'cutscenes', 'payphone'
)

from_object_path = os.path.join(
    from_directory, 'Graphics', 'Atlases', 'Gameplay', 'Author', 'Skin', 'objects', 'lookout'
)
to_object_path = os.path.join(
    to_directory, 'Graphics', 'Atlases', 'Gameplay', authorName, skinName, 'objects', 'lookout'
)

shutil.copytree(from_characters_path, to_characters_path)
shutil.copytree(from_cutscene_path, to_cutscene_path)
shutil.copytree(from_object_path, to_object_path)
shutil.rmtree(os.path.join(to_directory, 'Graphics', 'Atlases', 'Gameplay', 'Author'))

images = (
    glob.glob(os.path.join(to_characters_path, '**', '*.png'), recursive=True) +
    glob.glob(os.path.join(to_cutscene_path, '**', '*.png'), recursive=True) +
    glob.glob(os.path.join(to_object_path, '**', '*.png'), recursive=True)
)

def update_image_colors(image_path, color_mappings):
    img = Image.open(image_path)
    width, height = img.size
    for i in range(width):
        for j in range(height):
            data = img.getpixel((i, j))
            for old_color, new_color in color_mappings.items():
                if data[:3] == old_color:
                    img.putpixel((i, j), new_color)
    img.save(image_path)

# Default color mappings.
color_mappings = {
    (91, 110, 225): shirt,
    (63, 63, 116): sleeves,
    (69, 40, 60): collar,
    (135, 55, 36): trousers
}

for image in images:
    update_image_colors(image, color_mappings)
