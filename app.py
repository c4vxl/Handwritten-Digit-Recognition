import tkinter as tk
from PIL import Image
import os
from model.prompt import prompt_model

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten number recognition")
        self.root.geometry("300x400")
        # Set up the canvas
        self.canvas_width = 84
        self.canvas_height = 96
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        # Add labels for probabilities
        label = tk.Label(root, text="Last prediction:")
        label.pack()
        self.prob_labels = []
        self.prob_label_texts = [tk.StringVar() for _ in range(11)]
        for i in range(11):
            label = tk.Label(root, textvariable=self.prob_label_texts[i])
            label.pack()
            self.prob_labels.append(label)

        # Add a "Clear Canvas" button
        self.clear_button = tk.Button(root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack()

        # Variables to track mouse position and drawing state
        self.drawing = False
        self.last_x = 0
        self.last_y = 0

        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)

    def on_button_press(self, event):
        # Start drawing
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def on_mouse_drag(self, event):
        if self.drawing:
            # Draw on the canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill='black')
            # Update the last position
            self.last_x = event.x
            self.last_y = event.y

    def on_button_release(self, event):
        # Stop drawing
        self.drawing = False
        # Call custom function when mouse is released
        self.on_draw_complete()
    
    def on_draw_complete(self):
        self.canvas.postscript(file='canvas_image.ps', colormode='mono')
        image = Image.open('canvas_image.ps')
        os.remove("canvas_image.ps")
        
        # Rescale image if necessary
        image = image.resize((self.canvas_width, self.canvas_height), Image.ANTIALIAS)
        
        pixel_cds = [(x, y) for y in range(image.height) for x in range(image.width)]
        # Get pixel values
        pixels = [image.getpixel(x) for x in pixel_cds]
        # Convert pixel values to binary (1 if R value <= 200, otherwise 0)
        pixels = [1 if x[0] <= 200 else 0 for x in pixels]
        # Split pixels into 96 rows of 84 pixels each
        pixels = [pixels[84*i:84*(i+1)] for i in range(96)]
        
        prop, prop_map, pred = prompt_model(pixels)

        # Update the probability labels
        for i in range(10):
            self.prob_label_texts[i].set(f"{i}: {prop_map[i]:.2f}")
        self.prob_label_texts[10].set(f"Final Prediction: {pred}")

        print("Propabillities: ", prop_map, "\t", "Prediction: ", pred)

    def clear_canvas(self):
        self.canvas.delete("all")

# Create the main window
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()