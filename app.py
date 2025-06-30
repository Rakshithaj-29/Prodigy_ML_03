import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import pickle
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

with open(r"D:\DAA\rakshitha\prodigy_infotech\CatvsDog\catvsdog.pkl", "rb") as f:
    model = pickle.load(f)
with open(r"D:\DAA\rakshitha\prodigy_infotech\CatvsDog\hog_config.pkl", "rb") as f:
    hog_params = pickle.load(f)

def prepare_image(image_path):
    img = cv.imread(image_path)
    if img is None or len(img.shape) != 3:
        return None
    img = cv.resize(img, (64, 64))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    features, _ = hog(
        gray,
        orientations=hog_params["orientations"],
        pixels_per_cell=hog_params["pixels_per_cell"],
        cells_per_block=hog_params["cells_per_block"],
        block_norm=hog_params["block_norm"],
        visualize=True
    )
    features = features / 255.0
    return features.reshape(1, -1)

class CatDogClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat vs Dog Classifier")

        self.label = tk.Label(root, text="Predict Dog or Cat")
        self.label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.result_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )

        if not file_path:
            return

        pil_img = Image.open(file_path)
        pil_img = pil_img.resize((200, 200))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

        features = prepare_image(file_path)
        if features is not None:
            prediction = model.predict(features)[0]
            result = "Dog 🐶" if prediction == 1 else "Cat 🐱"
            self.result_label.configure(text=f"Prediction: {result}")
        else:
            self.result_label.configure(text="Invalid image!")

if __name__ == "__main__":
    root = tk.Tk()
    app = CatDogClassifierApp(root)
    root.mainloop()