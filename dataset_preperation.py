"""
This script is written for this dataset:
   https://github.com/kensanata/numbers/tree/master/0002_CH5M

Download the folder and rename it to "data".
"""


import PIL.Image as Image
import os
import json


def load_data_batch(image, num):
    pixels = list([0 if x[0] >= 200 else 1 for x in image.getdata()])

    return {
        "num": num,
        "pixels": pixels
    }

dataset = []

for d in range(1, 30):
    for i in range(10):
        folder = f"data/{d}/{i}/"
        print(f"dir: {folder}")

        for file in os.scandir(folder):
            file = file.name
            file = folder + file

            print("Image: \t", file)

            image = Image.open(file).convert("RGB")
            image = image.resize((84, 96))

            data = load_data_batch(image, i)

            dataset.append(data)


open("dataset.json", "w").write(json.dumps(dataset))
