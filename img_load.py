import PIL.Image as Image
from model import prompt_model

image = Image.open('image.png')

# Rescale image if necessary
image = image.resize((84, 96), Image.ANTIALIAS)


pixels = [1 if x[0] <= 200 else 0 for x in Image.open('image.png').resize((84, 96), Image.ANTIALIAS).getdata()]

prop, prop_map, pred = prompt_model(pixels)

print("Propabillities: ", prop_map, "\t", "Prediction: ", pred)