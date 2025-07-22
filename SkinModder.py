import glob
from PIL import Image
import numpy as np

images=glob.glob("*.png")

for image in images:
    img = Image.open(image)
    width = img.size[0]
    height = img.size[1]
    for i in range(0,width):
        for j in range(0,height):
            data = img.getpixel((i,j))

            if (data[0]==91 and data[1]==110 and data[2]==225):
                img.putpixel((i,j),(255, 170, 187))
            if (data[0]==63 and data[1]==63 and data[2]==116):
                img.putpixel((i,j),(255, 136, 153))
            if (data[0] == 69 and data[1] == 40 and data[2] == 60):
                img.putpixel((i, j), (136, 68, 85))
            if (data[0] == 135 and data[1] == 55 and data[2] == 36):
                img.putpixel((i, j), (187, 85, 119))
    img.save(image)
