"""
This script is written for this dataset:
   https://github.com/kensanata/numbers/tree/master/0002_CH5M

Download the folder and rename it to "numbers".
"""


import PIL.Image as Image
import os
import json

def convert_to_black_white(file_path):
    image = Image.open(file_path)

    if f"{image.width}x{image.height}" != "84x96":
        print(f"Resizing {file_path}...")
        image = image.resize((84, 96))
    
    pixels = [(x, y) for y in range(0, image.height) for x in range(0, image.width)]
    for pixel in pixels:
        image.putpixel(pixel, (255, 255, 255) if image.getpixel(pixel)[0] >= 200 else 0)
    
    image.save(file_path)
    image.close()

dataset = []

for i in range(0, 10):
    folder = f"numbers/{i}/"
    print(f"Files of {i}")

    for file in os.scandir(folder):
        file_path = folder + file.name

        convert_to_black_white(file_path)

        image = Image.open(file_path)

        pixels = list([0 if x[0] >= 200 else 1 for x in image.getdata()])

        data = {
            "num": i,
            "pixels": pixels
        }

        dataset.append(data)




open("dataset.json", "w").write(json.dumps(dataset))