"""
Test a trained CNN model on one image.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import joblib
from model_cnn import CNNModel


IMAGE_PATH = "/u50/chandd9/downloads/letters_numbers/h.png"
MODEL_PATH = "/u50/chandd9/al3/ocr_model.pth" 
LABEL_ENCODER_PATH = "/u50/chandd9/al3/label_encoder.joblib" 
IMG_SIZE = (64, 64)
DEVICE = "cuda" 
OUT_SIZE = 62    # number of classes
# -----------------------------

# def preprocess_image(image_path, img_size=(64, 64)):
#     # ---- Load grayscale ----
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError(f"Could not read image: {image_path}")

#     # Convert to uint8 if not
#     if img.dtype != np.uint8:
#         img = (img * 255).clip(0, 255).astype(np.uint8)

#     # Crop image to just non-white pixels
#     mask = img < 200  # treat pixels < 200 as non-white

#     # Crop to bounding box of non-white pixels
#     if np.any(mask):
#         coords = np.column_stack(np.where(mask))
#         y_min, x_min = coords.min(axis=0)
#         y_max, x_max = coords.max(axis=0)
#         img = img[y_min:y_max + 1, x_min:x_max + 1]

#     # Convert to float [0,1]
#     img = img.astype(np.float32) / 255.0

#     # write cropped preview image
#     # cv2.imwrite("preview_cropped_image.png", (img * 255).clip(0,255).astype(np.uint8))

#     # ---- Resize while preserving aspect ratio ----
#     target_h, target_w = img_size
#     h, w = img.shape

#     scale = min(target_w / w, target_h / h)
#     new_w = int(w * scale)
#     new_h = int(h * scale)

#     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

#     # ---- Center on white 64x64 canvas ----
#     canvas = np.ones((target_h, target_w), dtype=np.float32)  # white background

#     y_offset = (target_h - new_h) // 2
#     x_offset = (target_w - new_w) // 2

#     canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

#     # Add channel + batch dims → (1,1,64,64)
#     tensor = torch.tensor(canvas).unsqueeze(0).unsqueeze(0)

#     # ---- Save preview image ----
#     preview = tensor.squeeze().cpu().numpy()
#     preview_uint8 = (preview * 255).clip(0, 255).astype(np.uint8)

#     output_path = "preview_processed_image.png"
#     if not cv2.imwrite(output_path, preview_uint8):
#         raise ValueError("Could not write preview image.")

#     print(f"Saved preview image to: {output_path}")
#     print(f"Processed tensor shape: {tensor.shape}")

#     return tensor

def preprocess_image(img, img_size=(64, 64), count_debug=0):
    # ---- Load grayscale ----
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img}")

    # Ensure uint8
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # median blur to reduce noise
    img = cv2.medianBlur(img, 3)

    # ---- OTSU Thresholding to detect foreground ----
    # OTSU returns: ret (threshold_value), binary_image
    _, otsu_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Convert mask to boolean (True = foreground)
    mask = otsu_mask > 0

    # Crop to bounding box
    if np.any(mask):
        coords = np.column_stack(np.where(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img = img[y_min:y_max + 1, x_min:x_max + 1]

    # Normalize to [0,1] 
    img = img.astype(np.float32) / 255.0

    # ---- Resize with aspect ratio preserved ----
    target_h, target_w = img_size
    h, w = img.shape

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ---- Center on white canvas ----
    canvas = np.ones((target_h, target_w), dtype=np.float32)  # white background

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Add channel + batch dims → (1,1,64,64)
    tensor = torch.tensor(canvas).unsqueeze(0).unsqueeze(0)

    # DEBUGING LINES: Save preview image
    # preview = tensor.squeeze().cpu().numpy()
    # preview_uint8 = (preview * 255).clip(0, 255).astype(np.uint8)

    # output_path = f"debug/preview_processed_image_{count_debug}.png"
    # if not cv2.imwrite(output_path, preview_uint8):
    #     raise ValueError("Could not write preview image.")

    # print(f"Saved preview image to: {output_path}")
    # print(f"Processed tensor shape: {tensor.shape}")

    return tensor


def load_model(model_path, out_size=62, device="cuda"):
    """
    Load trained CNN model from disk.
    """
    model = CNNModel(out_size=out_size)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def predict_image(model, image_tensor, device="cuda", label_encoder=None):
    """
    Run forward pass on one image and get predicted class.
    """
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1) 
        pred_idx = logits.argmax(dim=1).cpu().item()

    if label_encoder is not None:
        return label_encoder.inverse_transform([pred_idx])[0]

    return pred_idx

def predict_image_V2(model, image_tensor, device="cuda", label_encoder=None, top_k=1):
    """
    Returns:
        - predicted label
        - confidence (0–1)
        - optionally top-k predictions
    """
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)               # convert logits → probabilities
        top_probs, top_idxs = probs.topk(top_k, dim=1)

    # Move to CPU + flatten
    top_probs = top_probs.squeeze().cpu().numpy()
    top_idxs = top_idxs.squeeze().cpu().numpy()

    # Decode labels if encoder exists
    if label_encoder is not None:
        top_labels = label_encoder.inverse_transform(top_idxs)
    else:
        top_labels = top_idxs

    # If user only wants the best prediction
    if top_k == 1:
        return top_labels.item(), float(top_probs.item())

    # For multiple predictions (top-k)
    return list(zip(top_labels, top_probs.astype(float)))


# MAIN FUNCTION
if __name__ == "__main__":

    # Load optional label encoder
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except:
        label_encoder = None

    # Load model
    model = load_model(MODEL_PATH, out_size=OUT_SIZE, device=DEVICE)

    # Preprocess image
    img_tensor = preprocess_image(IMAGE_PATH, img_size=IMG_SIZE)

    # Predict
    pred = predict_image(model, img_tensor, device=DEVICE, label_encoder=label_encoder)
    # output = predict_image_V2(model, img_tensor, device=DEVICE, label_encoder=label_encoder, top_k=3)

    # print("Top predictions (label, confidence):")
    # for label, confidence in output:
    #     print(f"  {label}: {confidence:.4f}")
    print(f"Predicted class: {pred}")

