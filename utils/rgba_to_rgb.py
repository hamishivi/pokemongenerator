"""
Util for converting all png images in a folder to RGB. This was required as many pokemon
images were RGBA, requiring explicit conversion.
"""
import sys
import os
from PIL import Image

folder = sys.argv[1]
fill_color = 'white'

# from https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
for filename in os.listdir(folder):
    filename = os.path.join(folder, filename)
    if 'png' not in filename:
        continue
    new_name = filename[:-4]
    image = Image.open(filename).convert('RGBA')
    background = Image.new(image.mode[:-1], image.size, fill_color)
    background.paste(image, image.split()[-1])
    background.save(new_name + '.jpg', 'JPEG', quality=95)
    os.remove(filename)