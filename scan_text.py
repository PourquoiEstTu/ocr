"""
Main Application for OCR Text Scanning
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
import joblib
import sys
import os
from preprocessing import potential_segmentation_columns

# Paths
MODEL_PATH = "/u50/chandd9/al3/ocr_model.pth"
LABEL_ENCODER_PATH = "/u50/chandd9/al3/label_encoder.joblib"
device = "cuda" if torch.cuda.is_available() else "cpu"

# make debug directory
if not os.path.exists("debug"):
    os.makedirs("debug")

# clear debug directory
for f in os.listdir("debug"):
    os.remove(os.path.join("debug", f))

# should prob change this to import from other files but we ball
class CNNModel(nn.Module):
    def __init__(self, input_size=(64, 64), out_size=10):
        super().__init__()

        self.input_h, self.input_w = input_size

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop2d = nn.Dropout2d(0.2)
           
        fc_input_dim = 128 * 8 * 8

        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, out_size)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B,1,H,W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop2d(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop2d(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop2d(x)

        x = torch.flatten(x, 1)     # flatten
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

def preprocess_image(img, img_size=(64, 64), count_debug=0):
    # Load grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if img is None:
        raise ValueError(f"Could not read image: {img}")

    # Convert to uint8 if not
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # Crop image to just non-white pixels
    mask = img < 200  # treat pixels < 200 as non-white

    # Crop to bounding box of non-white pixels
    if np.any(mask):
        coords = np.column_stack(np.where(mask))
        print(coords.shape)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img = img[y_min:y_max + 1, x_min:x_max + 1]
    else:
        # Image is empty, just return a blank 64x64 tensor
        img = np.ones(img_size, dtype=np.float32)

    # Convert to float [0,1]
    img = img.astype(np.float32) / 255.0

    # DEBUGING LINES: write cropped preview image
    # cv2.imwrite("preview_cropped_image.png", (img * 255).clip(0,255).astype(np.uint8))

    # ---- Resize while preserving aspect ratio ----
    target_h, target_w = img_size
    h, w = img.shape

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ---- Center on white 64x64 canvas ----
    canvas = np.ones((target_h, target_w), dtype=np.float32)  # white background

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Add channel + batch dims → (1,1,64,64)
    tensor = torch.tensor(canvas).unsqueeze(0).unsqueeze(0)

    # DEBUGING LINES: Save preview image
    preview = tensor.squeeze().cpu().numpy()
    preview_uint8 = (preview * 255).clip(0, 255).astype(np.uint8)

    output_path = f"debug/preview_processed_image_{count_debug}.png"
    if not cv2.imwrite(output_path, preview_uint8):
        raise ValueError("Could not write preview image.")

    print(f"Saved preview image to: {output_path}")
    # print(f"Processed tensor shape: {tensor.shape}")

    return tensor

def load_label_encoder():
    """Load the label encoder safely."""
    if not os.path.exists(LABEL_ENCODER_PATH):
        print("label_encoder.joblib not found.")
        sys.exit(1)

    return joblib.load(LABEL_ENCODER_PATH)


def load_model(img_size):
    """Load the trained CNN model."""
    model = CNNModel(input_size=img_size, out_size=62)

    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        sys.exit(1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def predict_word(image_path, model, label_encoder, img_size=(64,64)):
    """
    Predict text from a word image by segmenting characters and predicting each.
    """
    characters = potential_segmentation_columns(image_path)
    predicted_text = ""
    count_debug = 0

    for char_img in characters:
        count_debug += 1
        tensor = preprocess_image(char_img, img_size, count_debug)
        tensor = tensor.to(device)
        with torch.no_grad():
            outputs = model(tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
            label = label_encoder.inverse_transform([pred_idx])[0]
            predicted_text += label

    return predicted_text


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scan_text.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    # Preprocess image
    img_size = (64, 64)

    # Load model and label encoder
    model = load_model(img_size)
    model.to(device)
    label_encoder = load_label_encoder()

    predicted_words = predict_word(image_path, model, label_encoder, img_size)

    # print(f"Predicted class index: {pred_idx}")
    print(f"Predicted label: {predicted_words}")
