"""
Preprocessing module for OCR using neural networks.
Includes functions to load images and extract pixel data as features.
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np
import torch

DIR = r"/u50/chandd9/al3/"
FEATURE_DIR = f"{DIR}/ocr-pixel"
DATA = f"{DIR}/ocr-repo-files/Img"  # set directory path

# Old Helper Function
# def preprocess_image(image_path, img_size=(64, 64)):
#     # ---- Load grayscale ----
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError(f"Could not read image: {image_path}")

#     # Ensure uint8
#     if img.dtype != np.uint8:
#         img = (img * 255).clip(0, 255).astype(np.uint8)

#     # median blur to reduce noise
#     img = cv2.medianBlur(img, 3)

#     # ---- OTSU Thresholding to detect foreground ----
#     # OTSU returns: ret (threshold_value), binary_image
#     _, otsu_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Convert mask to boolean (True = foreground)
#     mask = otsu_mask > 0

#     # Crop to bounding box
#     if np.any(mask):
#         coords = np.column_stack(np.where(mask))
#         y_min, x_min = coords.min(axis=0)
#         y_max, x_max = coords.max(axis=0)
#         img = img[y_min:y_max + 1, x_min:x_max + 1]

#     # Normalize to [0,1] 
#     img = img.astype(np.float32) / 255.0

#     # ---- Resize with aspect ratio preserved ----
#     target_h, target_w = img_size
#     h, w = img.shape

#     scale = min(target_w / w, target_h / h)
#     new_w = int(w * scale)
#     new_h = int(h * scale)

#     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

#     # ---- Center on white canvas ----
#     canvas = np.ones((target_h, target_w), dtype=np.float32)  # white background

#     y_offset = (target_h - new_h) // 2
#     x_offset = (target_w - new_w) // 2

#     canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

#     # ---- Convert to tensor (1,1,64,64) ----
#     # tensor = torch.tensor(canvas).unsqueeze(0).unsqueeze(0)

#     # # ---- Debug preview ----
#     # preview = (canvas * 255).astype(np.uint8)
#     # cv2.imwrite("preview_processed_image.png", preview)

#     # print("Saved preview to preview_processed_image.png")
#     # print("Processed tensor shape:", tensor.shape)

#     return canvas


def preprocess_image(image_path, img_size=(64, 64)):
    # ---- Load grayscale ----
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

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

    # ---- Convert to tensor (1,1,64,64) ----
    # tensor = torch.tensor(canvas).unsqueeze(0).unsqueeze(0)

    # # ---- Debug preview ----
    # preview = (canvas * 255).astype(np.uint8)
    # cv2.imwrite("preview_processed_image.png", preview)

    # print("Saved preview to preview_processed_image.png")
    # print("Processed tensor shape:", tensor.shape)

    return canvas

# Functions used for dataset 1 (3410 images) (not in use anymore) -----------------------------------------------------------------------
def gen_pixel_features(input_dir:str, output_dir:str, overwrite_prev_files: bool=False) -> None :
    """Generates pixel features for all images in input_dir and saves them
       as .npy files in output_dir."""
    for file in os.scandir(input_dir) :
        if file.is_file() :
            print(f"Processing file: {file.name}")
            npy_path = os.path.join(output_dir, f"{file.name.strip('.png')}.npy")
            if not overwrite_prev_files and os.path.exists(npy_path):
                continue
            img = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (64, 64))  # Resize to 32x32 pixels
            # features = img_resized.flatten()  # Flatten to 1D array
            img_resized = img_resized.astype(np.float32) / 255.0  # Normalize pixel values
            features = np.expand_dims(img_resized, axis=0)
            np.save(npy_path, features)

def gen_pixel_labels(input_dir:str, output_file:str, overwrite_prev_file=False) -> None :
    """Generates labels for all images in input_dir and saves them
       as a .npy file."""
    labels = []
    # npy_path = os.path.join(output_dir, "ordered_labels.npy")
    if not overwrite_prev_file and os.path.exists(output_file) :
        print(f"Labels already exist. Set overwrite_prev_file to True to re-generate labels.")
        return
    # df = pd.read_csv(csv_file, sep=",")
    df = pd.read_csv(f"{DIR}/ocr-repo-files/english.csv", sep=",")
    dic = {}
    i = 0
    for x in df["image"]:
        dic[x[4:]] = df["label"][i]
        i += 1
    for file in sorted(os.listdir(input_dir)) :
        label = dic.get(file, None)
        labels.append(label)
        # print(label)
        # exit()
    np.save(output_file, np.array(labels))

# Functions used for dataset 2 (210k images) (currently used) -------------------------------------------------
def old_gen_pixel_features_nested(input_dir: str, output_dir: str, overwrite_prev_files: bool = False) -> None:
    """
    Scans input_dir for subfolders (class names), loads images inside them,
    preprocesses them into CNN-ready pixel tensors, and saves .npy files with
    names like 'u_L_1.npy', 'u_L_2.npy', '1_1.npy', etc.

    not in use anymore, replaced by gen_pixel_features_nested_fixed
    """

    os.makedirs(output_dir, exist_ok=True)

    # Loop through subdirectories like 1/, u_L/, U_U/
    for folder in os.scandir(input_dir):
        if not folder.is_dir():
            continue  # skip files
        
        class_name = folder.name  # e.g., "1", "u_L", "U_U"
        print(f"Processing class folder: {class_name}")

        image_counter = 1

        # Loop over every file in the class folder
        for file in os.scandir(folder.path):
            if not file.is_file():
                continue

            # Build output name: className_index.npy
            out_name = f"{class_name}_{image_counter}.npy"
            out_path = os.path.join(output_dir, out_name)

            # Skip if already exists
            if not overwrite_prev_files and os.path.exists(out_path):
                image_counter += 1
                continue

            # Load grayscale
            img = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping unreadable file: {file.path}")
                continue

            # Resize to 64×64
            # img = cv2.resize(img, (64, 64))

            # Normalize to [0,1]
            img = img.astype(np.float32) / 255.0

            # resize while preserving aspect ratio
            h, w = img.shape
            target_h, target_w = 64, 64
            # trying 96x96
            # target_h, target_w = 96, 96
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))

            # Pad to center the image
            canvas = np.zeros((target_h, target_w), dtype=np.float32)  # white background
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

            # Add channel dimension → (1, 64, 64)
            img = np.expand_dims(canvas, axis=0)

            # Save as .npy
            np.save(out_path, img)
            print(f"Saved: {out_name}")

            image_counter += 1

def gen_pixel_features_nested_otsu(input_dir: str, output_dir: str, overwrite_prev_files: bool = False) -> None:
    """
    Uses the global preprocess_image() to convert images to the
    normalized (1,1,64,64) tensor, then saves them as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for folder in os.scandir(input_dir):
        if not folder.is_dir():
            continue

        class_name = folder.name
        print(f"Processing class folder: {class_name}")
        image_counter = 1

        for file in os.scandir(folder.path):
            if not file.is_file():
                continue

            out_name = f"{class_name}_{image_counter}.npy"
            out_path = os.path.join(output_dir, out_name)

            # Skip if exists
            if not overwrite_prev_files and os.path.exists(out_path):
                image_counter += 1
                continue

            # Load grayscale as numpy array
            img = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping unreadable file: {file.path}")
                continue

            # Call preprocess_image
            try:
                arr = preprocess_image(file.path)   # returns shape (1,1,64,64)
            except Exception as e:
                print(f"Error processing {file.path}: {e}")
                continue

            # Save .npy
            np.save(out_path, arr)
            print(f"Saved: {out_name}")

            image_counter += 1


# def gen_pixel_features_nested_fixed(input_dir: str, output_dir: str, overwrite_prev_files: bool = False) -> None:
#     """
#     Loads images, crops text using brightness threshold, resizes while preserving
#     aspect ratio, centers on 64x64 white background, and saves as .npy.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     for folder in os.scandir(input_dir):
#         if not folder.is_dir():
#             continue
        
#         class_name = folder.name
#         print(f"Processing class folder: {class_name}")
#         image_counter = 1

#         for file in os.scandir(folder.path):
#             if not file.is_file():
#                 continue

#             out_name = f"{class_name}_{image_counter}.npy"
#             out_path = os.path.join(output_dir, out_name)

#             if not overwrite_prev_files and os.path.exists(out_path):
#                 image_counter += 1
#                 continue

#             # Load grayscale
#             img = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 print(f"Skipping unreadable file: {file.path}")
#                 continue

#             # Convert to uint8 if needed
#             if img.dtype != np.uint8:
#                 img = img.astype(np.uint8)

#             # CROP IMAGE
#             # Mask of all pixels that are NOT near-white
#             mask = img < 200  # treat pixels < 200 as non-white

#             if np.any(mask):
#                 coords = np.column_stack(np.where(mask))
#                 y_min, x_min = coords.min(axis=0)
#                 y_max, x_max = coords.max(axis=0)
#                 img = img[y_min:y_max+1, x_min:x_max+1]
#             # else: full white image → leave as-is

#             # Convert to float32 [0,1] AFTER cropping
#             img = img.astype(np.float32) / 255.0

#             # RESIZE WITH ASPECT RATIO PRESERVATION
#             target_h, target_w = 64, 64
#             h, w = img.shape

#             scale = min(target_w / w, target_h / h)
#             new_w = int(w * scale)
#             new_h = int(h * scale)

#             resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

#             # CENTER ON WHITE 64×64 CANVAS
#             canvas = np.ones((target_h, target_w), dtype=np.float32)  # white background

#             y_offset = (target_h - new_h) // 2
#             x_offset = (target_w - new_w) // 2

#             canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

#             # Add channel dimension → (1, 64, 64)
#             tensor = np.expand_dims(canvas, axis=0)

#             # Save
#             np.save(out_path, tensor)
#             print(f"Saved: {out_name}")

#             image_counter += 1

def gen_pixel_labels_nested(input_dir: str, output_file: str, overwrite_prev_file=False) -> None:
    """
    Generates labels for .npy feature files by taking the first token in the
    file name (before the first underscore) and saves all labels to output_file.
    
    Example:
        u_L_1.npy -> label "u"
        1_3.npy   -> label "1"
    """
    if not overwrite_prev_file and os.path.exists(output_file):
        print("Labels file already exists. Use overwrite_prev_file=True to regenerate.")
        return

    labels = []

    for file in sorted(os.listdir(input_dir)):
        if not file.endswith(".npy"):
            continue

        # remove ".npy"
        base = file[:-4]  

        # take first token
        label = file[0]

        labels.append(label)

    labels = np.array(labels)
    np.save(output_file, labels)
    print(f"Saved {len(labels)} labels to {output_file}")


# functions used for both datasets (combining both datasets, not currently using) -------------------------------------
def get_pixels_and_labels(feature_dir:str, label_file:str) -> tuple[np.ndarray, np.ndarray] :
    """Loads pixel features and labels from specified directory and file."""
    feature_list = []
    file_names = []
    if not os.path.exists(label_file) :
        print(f"Label file {label_file} does not exist.")
        sys.exit(1)
    for file in sorted(os.listdir(feature_dir)) :
        if file == "ordered_labels.npy":
            continue
        features = np.load(f"{feature_dir}/{file}")
        # print(features.shape)
        feature_list.append(features)
        file_names.append(file)
    X = np.array(feature_list)
    y = np.load(label_file)
    return X, y, file_names

def gen_pixel_labels_combined(input_dir: str, output_file: str, csv_file: str = None, overwrite_prev_file=False) -> None:
    """
    Generates labels for all .npy feature files in input_dir.
    - First tries to map file names using csv_file (if provided)
    - If not found in CSV, uses the first character of the filename as the label
    Saves the labels as a .npy file.
    """
    if not overwrite_prev_file and os.path.exists(output_file):
        print(f"Labels file already exists: {output_file}")
        return

    df = pd.read_csv(f"{DIR}/ocr-repo-files/english.csv", sep=",")
    dic = {}
    i = 0
    for x in df["image"]:
        print(x[4:-4])
        dic[x[4:-4]] = df["label"][i]
        i += 1

    labels = []

    for file in sorted(os.listdir(input_dir)):
        if not file.endswith(".npy") or file == "ordered_labels.npy":
            continue

        # Try CSV mapping
        # print(f"Processing file: {file[:-4]}")
        # print(f"Looking for label for file: {file[:-4]}")
        # print(dic.keys())
        label = dic.get(file[:-4], None)
        # print(f"Label from CSV: {label}")

        # Fallback to first character of filename
        if label is None:
            label = file[0]

        labels.append(label)

    labels = np.array(labels)
    np.save(output_file, labels)
    print(f"Saved {len(labels)} labels to {output_file}")

# Exectuion Calls and Testing ------------------------------------------------------------------------------------

# gen_pixel_features(DATA, FEATURE_DIR, True)
# gen_pixel_features_nested("/u50/chandd9/al3/ocr-repo-files-2", "/u50/chandd9/al3/ocr-pixel-nested-V2", True)
# gen_pixel_features_nested_fixed("/u50/chandd9/al3/ocr-repo-files-2", "/u50/chandd9/al3/ocr-pixel-nested-V2-fixed", True)
# gen_pixel_labels_nested("/u50/chandd9/al3/ocr-pixel-nested-V2-fixed/", os.path.join("/u50/chandd9/al3/ocr-pixel-nested-V2-fixed", "ordered_labels.npy"), True)
# npy_path = os.path.join(DATA, "english.csv")
# df = pd.read_csv(f"{DIR}/ocr-repo-files/english.csv", sep=",")
# dic = {}
# i = 0
# for x in df["image"]:
#     dic[x] = df["label"][i]
#     i += 1
# # print(dic)

# gen_pixel_features_nested_otsu("/u50/chandd9/al3/ocr-repo-files-2", "/u50/chandd9/al3/ocr-pixel-nested-V2-otsu", True)
# gen_pixel_labels_nested("/u50/chandd9/al3/ocr-pixel-nested-V2-otsu/", os.path.join("/u50/chandd9/al3/ocr-pixel-nested-V2-otsu", "ordered_labels.npy"), True)

# gen_pixel_labels(DATA, os.path.join(FEATURE_DIR, "ordered_labels.npy"), True)
# get_pixels_and_labels(FEATURE_DIR, os.path.join(FEATURE_DIR, "ordered_labels.npy"))
# gen_pixel_labels_combined("/u50/chandd9/al3/ocr-pixel-combined", "/u50/chandd9/al3/ocr-pixel-combined/ordered_labels.npy", f"{DIR}/ocr-repo-files/english.csv", True)


# print("labels in combined directory:")
# labels = np.load("/u50/chandd9/al3/ocr-pixel-combined/ordered_labels.npy")
# # print(labels)
# print(f"Total labels: {len(labels)}")
# print(f"Unique labels: {set(labels)}")
# print(f"Unique labels: {len(set(labels))}")
# print(f"labels count:")
# from collections import Counter
# label_counts = Counter(labels)
# for label, count in label_counts.items():
#     print(f"Label: {label}, Count: {count}")

# helper functions 
def visualize_npy_image(npy_path, output_path=None, scale_to_255=True):
    """
    Visualize a single preprocessed image stored as a .npy file.

    Args:
        npy_path (str): Path to the .npy file (shape should be (1, H, W) or (H, W))
        output_path (str, optional): Path to save the image as PNG. If None, displays the image.
        scale_to_255 (bool): If True, converts values [0,1] → [0,255] uint8.

    Returns:
        img (np.ndarray): The image array (H, W) uint8
    """
    # Load .npy
    data = np.load(npy_path)

    # Remove channel dim if exists
    if data.ndim == 3 and data.shape[0] == 1:
        img = data[0]
    else:
        img = data

    # Scale to 0-255
    if scale_to_255:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved visual image to: {output_path}")
    else:
        # Display using OpenCV
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img

def crop_to_text(img: np.ndarray):
    """
    Crops a grayscale image so that only the black text remains
    (removes all surrounding white padding).

    Args:
        img : np.ndarray
            Grayscale image (0=black, 255=white)

    Returns:
        Cropped grayscale image (np.ndarray)
    """

    # Ensure image is grayscale uint8
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # Create a mask of non-white pixels
    # threshold < 255 ensures any slightly dark pixel is counted
    mask = img < 250       # adjustable if you want more/less strict

    if not np.any(mask):
        # No text found (empty image)
        return img

    coords = np.column_stack(np.where(mask))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img[y_min:y_max + 1, x_min:x_max + 1]

    return cropped


# a = visualize_npy_image("/u50/chandd9/al3/ocr-pixel-nested-V2-fixed/A_U_23.npy", "preview_npy_image.png")