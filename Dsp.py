import os
import random
import tkinter as tk
from glob import glob
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("1000x600")  # Adjusted initial window size

        self.cat_files = glob('./input/cat-and-dog/training_set/training_set/cats/*.jpg')
        self.dog_files = glob('./input/cat-and-dog/training_set/training_set/dogs/*.jpg')

        self.process_types = ["2D Convolution", "Blur", "Gaussian Blur", "Median Blur",
                              "Bilateral Filter", "Edge Detection", "Image Gradients",
                              "Image Resizing", "Image Rotation", "Cartooning", "Sharpness"]

        self.process_type_var = tk.StringVar()
        self.process_type_var.set(self.process_types[0])  # Default value

        self.num_images_var = tk.IntVar()
        self.num_images_var.set(1)  # Default value

        self.save_path_var = tk.StringVar()
        self.save_path_var.set("")

        self.selected_images = []
        self.processed_images = []

        self.setup_ui()

    def setup_ui(self):
        self.frame_original_images = tk.Frame(self.root)
        self.frame_original_images.pack(side="top", padx=10, pady=10)

        self.frame_processed_images = tk.Frame(self.root)
        self.frame_processed_images.pack(side="bottom", padx=10, pady=10)

        tk.Label(self.frame_original_images, text="Original Images").pack()

        tk.Label(self.frame_processed_images, text="Processed Images").pack()

        process_type_dropdown = ttk.Combobox(self.root, textvariable=self.process_type_var, values=self.process_types)
        process_type_dropdown.pack()

        tk.Label(self.root, text="Choose image location:").pack()
        self.source_var = tk.StringVar()
        tk.Radiobutton(self.root, text="Cat", variable=self.source_var, value="cat").pack()
        tk.Radiobutton(self.root, text="Dog", variable=self.source_var, value="dog").pack()

        tk.Label(self.root, text="Number of images to process:").pack()
        tk.Spinbox(self.root, from_=1, to=10, textvariable=self.num_images_var).pack()

        tk.Button(self.root, text="Select Images", command=self.select_images).pack()
        tk.Button(self.root, text="Process Images", command=self.process_images).pack()
        tk.Button(self.root, text="Reset", command=self.reset).pack()
        tk.Button(self.root, text="Save Processed Images", command=self.save_processed_images).pack()
        tk.Entry(self.root, textvariable=self.save_path_var).pack()
        tk.Button(self.root, text="Browse", command=self.browse_save_path).pack()

    def browse_save_path(self):
        save_path = filedialog.askdirectory(title="Select Directory to Save Images")
        if save_path:
            self.save_path_var.set(save_path)

    def process_images(self):
        process_type = self.process_type_var.get()
        num_images = self.num_images_var.get()
        save_path = self.save_path_var.get()

        if not process_type:
            messagebox.showerror("Error", "Please select a process type.")
            return

        if not self.selected_images:
            messagebox.showerror("Error", "Please select images to process.")
            return

        selected_files = self.cat_files if 'cat' in self.selected_images[0] else self.dog_files

        if len(selected_files) == 0:
            messagebox.showerror("Error", "No image files found.")
            return

        if num_images > len(selected_files):
            messagebox.showerror("Error", f"Number of images exceeds available images ({len(selected_files)}).")
            return

        self.clear_processed_images()

        for file_path in random.sample(self.selected_images, min(len(self.selected_images), num_images)):
            img = cv2.imread(file_path)
            processed_image = self.process_image(img, process_type)
            if processed_image is not None:
                self.processed_images.append(processed_image)

        self.show_processed_images()

    def process_image(self, img, process_type):
        # Convert image to uint8 format if necessary
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        if process_type == "2D Convolution":
            kernel = np.ones((5, 5), np.float32) / 25
            processed_image = cv2.filter2D(img, -1, kernel)
        elif process_type == "Blur":
            processed_image = cv2.blur(img, (5, 5))
        elif process_type == "Gaussian Blur":
            processed_image = cv2.GaussianBlur(img, (5, 5), 0)
        elif process_type == "Median Blur":
            processed_image = cv2.medianBlur(img, 5)
        elif process_type == "Bilateral Filter":
            processed_image = cv2.bilateralFilter(img, 9, 75, 75)
        elif process_type == "Edge Detection":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.Canny(gray, 100, 200)
        elif process_type == "Image Gradients":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            processed_image = np.sqrt(sobelx ** 2 + sobely ** 2)
        elif process_type == "Image Resizing":
            processed_image = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
        elif process_type == "Image Rotation":
            rows, cols, _ = img.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
            processed_image = cv2.warpAffine(img, rotation_matrix, (cols, rows))
        elif process_type == "Cartooning":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(img, 9, 300, 300)
            processed_image = cv2.bitwise_and(color, color, mask=edges)
        elif process_type == "Sharpness":
            processed_image = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        else:
            messagebox.showerror("Error", "Invalid process type.")
            return None

        return processed_image

    def select_image_location(self):
        location = self.source_var.get()
        if location == "cat":
            self.selected_files = self.cat_files
        elif location == "dog":
            self.selected_files = self.dog_files
        else:
            messagebox.showerror("Error", "Please select image location (Cat/Dog).")

    def select_images(self):
        self.select_image_location()
        num_images = self.num_images_var.get()
        if self.selected_files:
            file_paths = self.selected_files[:num_images]
            self.selected_images = file_paths  # Store file paths instead of loading images
            self.show_original_images()  # Display original images

    def show_original_images(self):
        for file_path in self.selected_images:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((100, 100), Image.BILINEAR)  # Resize image for display
            img = ImageTk.PhotoImage(image=img)
            img_label = tk.Label(self.frame_original_images, image=img)
            img_label.image = img
            img_label.pack(side="left", padx=5, pady=5)

    def show_processed_images(self):
        for processed_image in self.processed_images:
            # Convert processed image to uint8 if necessary
            if processed_image.dtype != np.uint8:
                processed_image = (processed_image * 255).astype(np.uint8)

            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_image = Image.fromarray(processed_image)
            processed_image = processed_image.resize((100, 100), Image.BILINEAR)  # Resize image for display
            processed_image = ImageTk.PhotoImage(image=processed_image)
            img_label = tk.Label(self.frame_processed_images, image=processed_image)
            img_label.image = processed_image
            img_label.pack(side="left", padx=5, pady=5)

    def save_processed_images(self):
        save_path = self.save_path_var.get()
        if not save_path:
            messagebox.showerror("Error", "Please select a directory to save the images.")
            return

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, processed_image in enumerate(self.processed_images):
            filename = os.path.join(save_path, f"processed_image_{i}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Info", "Processed images saved successfully.")

    def clear_processed_images(self):
        for widget in self.frame_processed_images.winfo_children():
            widget.destroy()
        self.processed_images.clear()

    def clear_images(self):
        for widget in self.frame_original_images.winfo_children():
            widget.destroy()
        self.selected_images.clear()

    def reset(self):
        self.clear_processed_images()
        self.clear_images()
        self.process_type_var.set(self.process_types[0])
        self.num_images_var.set(1)
        self.save_path_var.set("")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
