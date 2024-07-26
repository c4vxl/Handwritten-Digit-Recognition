# Handwritten Digit Recognition

This repository contains a simple machine learning model for recognizing handwritten digits. The model is built using PyTorch and is capable of classifying digits from 0 to 9 based on input images drawn on a canvas.

## Project Structure

- **/model/model.py**: Defines the architecture of the neural network.
- **/model/prompt.py**: Contains the function to load the model and make predictions.
- **/app.py**: Provides a Tkinter-based GUI for drawing digits and displaying predictions.
- **/dataset_preperation.py**: Prepares the dataset for training.
- **/train.py**: Contains the training script for the neural network.
- **/dataset.json**: The dataset file containing preprocessed images and their corresponding labels.
- **/model.pth**: A model file I trained myself.

## Setup
1. Install libraries:
    ```
    pip3 install torch pillow
    ```

2. You might need to install tkinter:
    <details>
      <summary>Debian/Ubuntu</summary>
    
      ```bash
      sudo apt-get install python3-tk
      ```
    </details>
    <details>
      <summary>Fedora</summary>
    
      ```bash
      sudo dnf install python3-tkinter
      ```
    </details>
    <details>
      <summary>Arch</summary>
    
      ```bash
      sudo pacman -S tk
      ```
    </details>
    <details>
      <summary>Mac</summary>
    
      ```bash
      brew install python-tk
      ```
    </details>

3. Train / Finetune

4. Use the application

## Usage
1. Run `app.py` to start the Tkinter application
2. Draw a digit on the canvas
3. The application will display the predicted digit and probabilities for each digit.

OR

1. Implement the `model.py` in your own code

## Model Architecture
The neural network model consists of:

- An input layer
- Multiple hidden layers (with ReLU activation)
- An output layer

The model takes a flattened 84x96 image (8064 pixels) as input and outputs probabilities for each digit (0-9).

## Notes
- The pretrained model (model.pth) provided is a quick and preliminary model. For better performance, consider training the model for more epochs or using a more complex architecture.

## Acknowledgments
The dataset I used for training my model and used for the dataset.json file comes from [Kensanata](https://github.com/kensanata/numbers/tree/master/0002_CH5M) in github. I downloaded it, tweaked it and then converted it into the dataset.json file. Therefore the `dataset_preperation.py`-Script is written for this exact dataset, so you might need to change it in case you want to use another dataset.

==> Link to this dataset: https://github.com/kensanata/numbers/tree/master/0002_CH5M

---
## Licence
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.