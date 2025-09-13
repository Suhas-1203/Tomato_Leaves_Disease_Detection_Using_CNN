import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load your trained model (update the path if needed)
model = tf.keras.models.load_model('tomato_leaf_final_model.h5')

# Define the class labels as per your training dataset
class_names = [
    'Bacterial spot',
    'Early blight',
    'Late blight',
    'Leaf Mold',
    'Septoria leaf spot',
    'Spider Mites',
    'Target Spot',
    'Yellow Leaf Curl Virus',
    'Mosaic virus',
    'Healthy'
]
IMG_SIZE = (128, 128)

def predict_image(file_path):
    """Predict disease class of an uploaded image."""
    try:
        img = Image.open(file_path).resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        return predicted_class
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error during prediction: {e}")
        return None

def upload_image():
    """Handle image upload and display predicted disease."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")]
    )
    if file_path:
        try:
            img = Image.open(file_path).resize((250, 250))
            img_tk = ImageTk.PhotoImage(img)
            panel.config(image=img_tk)
            panel.image = img_tk
            pred_class = predict_image(file_path)
            if pred_class:
                label_result.config(text=f"Prediction: {pred_class}", fg="blue")
        except Exception as e:
            messagebox.showerror("Error", f"Could not process image: {e}")

def clear_all():
    """Reset GUI components."""
    panel.config(image='')
    panel.image = None
    label_result.config(text='')

def show_about():
    """Pop-up with project info."""
    messagebox.showinfo("About", 
        "Tomato Leaf Disease Detector\nDesigned for educational use.\n"
        "Upload an image to see the predicted disease.")

# Build GUI
app = tk.Tk()
app.title("Tomato Disease Detector - Academic Version")
app.geometry("700x500")
app.configure(bg="#f0f4f9")

# Header
header = tk.Label(app, text="Tomato Leaf Disease Detector", font=("Arial Rounded MT Bold", 20),
                  bg="#34495e", fg="white", pady=12)
header.pack(fill=tk.X)

# Description
desc = tk.Label(app, text="Upload an image of tomato leaf to classify its disease.",
                font=("Arial", 11), bg="#f0f4f9")
desc.pack(pady=8)

# Buttons Frame
btn_frame = tk.Frame(app, bg="#f0f4f9")
btn_frame.pack(pady=6)

upload_btn = tk.Button(btn_frame, text="Upload Image", command=upload_image,
                       bg="#2980b9", fg="white", width=15, padx=4, pady=4)
upload_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(btn_frame, text="Reset", command=clear_all,
                      bg="#c0392b", fg="white", width=12, padx=4, pady=4)
clear_btn.grid(row=0, column=1, padx=10)

about_btn = tk.Button(btn_frame, text="About", command=show_about,
                      bg="#16a085", fg="white", width=12, padx=4, pady=4)
about_btn.grid(row=0, column=2, padx=10)

# Image display panel
panel = tk.Label(app, bg="#ecf0f1", width=250, height=250, relief=tk.RIDGE, borderwidth=2)
panel.pack(pady=8)

# Result Label
label_result = tk.Label(app, text="", font=("Arial", 16, "bold"), fg="#2c3e50", bg="#f0f4f9")
label_result.pack(pady=6)

# Legend / Info
legend_text = "Class Labels:\n" + "\n".join([f"{name}" for name in class_names])
legend = tk.Label(app, text=legend_text, font=("Arial", 9), bg="#f0f4f9", fg="#7f8c8d", justify=tk.LEFT)
legend.pack(pady=4)

# Run the GUI event loop
app.mainloop()
