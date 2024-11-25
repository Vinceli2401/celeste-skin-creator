import glob
import os
import json
import shutil

from PIL import Image
from PIL import ImageColor

# install all the libraries before using the program!!!

config_file_path = os.path.join(os.getcwd(), 'config.json')

if not os.path.exists(config_file_path):
    raise FileNotFoundError(f"Configuration file 'config.json' not found in the root directory.")

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

skinName = config.get("skinName", "default_skin")
authorName = config.get("authorName", "default_author")
dash0 = config.get("dash0", "#FFFFFF")  # Default to white
dash1 = config.get("dash1", "#FFFFFF")
dash2 = config.get("dash2", "#FFFFFF")

shirt = ImageColor.getrgb(config.get("shirt", "#FFFFFF"))
sleeves = ImageColor.getrgb(config.get("sleeves", "#FFFFFF"))
collar = ImageColor.getrgb(config.get("collar", "#FFFFFF"))
trousers = ImageColor.getrgb(config.get("trousers", "#FFFFFF"))

# Debug output (optional)
print(f"Skin Name: {skinName}")
print(f"Author Name: {authorName}")
print(f"Dash Colors: {dash0}, {dash1}, {dash2}")
print(f"Shirt Color: {shirt}")
print(f"Sleeves Color: {sleeves}")
print(f"Collar Color: {collar}")
print(f"Trousers Color: {trousers}")

from_directory = "./SkinBase"
to_directory = "./Skin/" + skinName
shutil.copytree(from_directory, to_directory)

with open(to_directory + '/everest.yaml', encoding='UTF-8', mode='r') as file:
    newlines = []
    for line in file:
        newlines.append(line.replace('Replace', skinName))

with open(to_directory + '/everest.yaml', encoding='UTF-8', mode='w') as file:
    for line in newlines:
        file.write(line)

with open(to_directory + '/SkinModHelperConfig.yaml', encoding='UTF-8', mode='r') as file:
    newlines = []
    for line in file:
        newlines.append(
            line.replace('Author_Skin', authorName + "_" + skinName).replace("Dash0", dash0).replace("Dash1",
                                                                                                     dash1).replace(
                "Dash2", dash2))

with open(to_directory + '/SkinModHelperConfig.yaml', encoding='UTF-8', mode='w') as file:
    for line in newlines:
        file.write(line)

if not os.path.exists(to_directory + '/Graphics/' + authorName):
    os.mkdir(to_directory + '/Graphics/' + authorName)
    os.mkdir(to_directory + '/Graphics/' + authorName + '/' + skinName)
os.replace(to_directory + '/Graphics/Author/Skin/Sprites.xml',
           to_directory + '/Graphics/' + authorName + '/' + skinName + '/Sprites.xml')
shutil.rmtree(to_directory + '/Graphics/Author')

with open(to_directory + '/Graphics/' + authorName + '/' + skinName + '/Sprites.xml', encoding='UTF-8',
          mode='r') as file:
    newlines = []
    for line in file:
        newlines.append(line.replace('Author/Skin', authorName + "/" + skinName + '/characters'))

with open(to_directory + '/Graphics/' + authorName + '/' + skinName + '/Sprites.xml', encoding='UTF-8',
          mode='w') as file:
    for line in newlines:
        file.write(line)

with open(to_directory + '/Dialog/English.txt', encoding='UTF-8', mode='r') as file:
    newlines = []
    for line in file:
        newlines.append(line.replace('Author_Skin', authorName + "_" + skinName).replace('skinName', skinName))

with open(to_directory + '/Dialog/English.txt', encoding='UTF-8', mode='w') as file:
    for line in newlines:
        file.write(line)

from_directory = to_directory + '/Graphics/Atlases/Gameplay/Author/Skin/characters'
to_directory = to_directory + '/Graphics/Atlases/Gameplay/' + authorName + '/' + skinName + '/characters'
shutil.copytree(from_directory, to_directory)
shutil.rmtree(to_directory + '/Graphics/Atlases/Gameplay/Author')

images = glob.glob(to_directory + '/Graphics/Atlases/Gameplay/' + authorName + '/' + skinName + '/characters/**/*.png')

for image in images:
    img = Image.open(image)
    width = img.size[0]
    height = img.size[1]
    for i in range(0, width):
        for j in range(0, height):
            data = img.getpixel((i, j))
            if data[0] == 91 and data[1] == 110 and data[2] == 225:  # shirt
                img.putpixel((i, j), shirt)  # rgb color code for color you want to change
            if data[0] == 63 and data[1] == 63 and data[2] == 116:  # sleeves
                img.putpixel((i, j), sleeves)
            if data[0] == 69 and data[1] == 40 and data[2] == 60:  # collar
                img.putpixel((i, j), collar)
            if data[0] == 135 and data[1] == 55 and data[2] == 36:  # trousers
                img.putpixel((i, j), trousers)
    img.save(image)
