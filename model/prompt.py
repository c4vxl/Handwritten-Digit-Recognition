import torch
import os
from model.model import SimpleNeuralNetwork

def prompt_model(pixels, model_file = "model.pth"):
    # load model
    model = SimpleNeuralNetwork(n_inp=len(pixels)*len(pixels[0]))

    # load pretrained
    if os.path.exists(model_file): model.load_state_dict(torch.load(model_file))
    else: print("NO MODEL FOUND! USING RANDOM INITIALIZED WEIGHTS!!!")


    pixels = [pixel for row in pixels for pixel in row] # flatten pixels

    # convert pixel array to tensor
    pixels = torch.Tensor(pixels)

    # predict
    prediction = model(pixels)

    probabilities = list(prediction.detach().numpy())

    probabilities_map = {i: probabilities[i] for i in range(0, len(probabilities))}

    prediction = max(probabilities_map, key=lambda k: probabilities_map[k])

    return probabilities, probabilities_map, prediction