"""
Preprocessing module for OCR using neural networks.
Includes functions to load images and extract pixel data as features.
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np

DIR = r"/u50/chandd9/al3/"
FEATURE_DIR = f"{DIR}/ocr-pixel"
DATA = f"{DIR}/ocr-repo-files/Img"  # set directory path

# Functions used for dataset 1 (3410 images) -----------------------------------------------------------------------
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

# Functions used for dataset 2 (210k images) (not handwritten tho) -------------------------------------------------
def gen_pixel_features_nested(input_dir: str, output_dir: str, overwrite_prev_files: bool = False) -> None:
    """
    Scans input_dir for subfolders (class names), loads images inside them,
    preprocesses them into CNN-ready pixel tensors, and saves .npy files with
    names like 'u_L_1.npy', 'u_L_2.npy', '1_1.npy', etc.
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
            img = cv2.resize(img, (64, 64))

            # Normalize to [0,1]
            img = img.astype(np.float32) / 255.0

            # Add channel dimension → (1, 64, 64)
            img = np.expand_dims(img, axis=0)

            # Save as .npy
            np.save(out_path, img)
            print(f"Saved: {out_name}")

            image_counter += 1

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

# functions used for both datasets ----------------------------------------------------------------------------
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
# gen_pixel_features_nested("/u50/chandd9/al3/ocr-repo-files-2", "/u50/chandd9/al3/ocr-pixel-nested", True)
# gen_pixel_labels_nested("/u50/chandd9/al3/ocr-pixel-nested/", os.path.join("/u50/chandd9/al3/ocr-pixel-nested", "ordered_labels.npy"), True)
# npy_path = os.path.join(DATA, "english.csv")
# df = pd.read_csv(f"{DIR}/ocr-repo-files/english.csv", sep=",")
# dic = {}
# i = 0
# for x in df["image"]:
#     dic[x] = df["label"][i]
#     i += 1
# # print(dic)

# gen_pixel_labels(DATA, os.path.join(FEATURE_DIR, "ordered_labels.npy"), True)
# get_pixels_and_labels(FEATURE_DIR, os.path.join(FEATURE_DIR, "ordered_labels.npy"))
# gen_pixel_labels_combined("/u50/chandd9/al3/ocr-pixel-combined", "/u50/chandd9/al3/ocr-pixel-combined/ordered_labels.npy", f"{DIR}/ocr-repo-files/english.csv", True)


print("labels in combined directory:")
labels = np.load("/u50/chandd9/al3/ocr-pixel-combined/ordered_labels.npy")
# print(labels)
print(f"Total labels: {len(labels)}")
print(f"Unique labels: {set(labels)}")
print(f"Unique labels: {len(set(labels))}")
print(f"labels count:")
from collections import Counter
label_counts = Counter(labels)
for label, count in label_counts.items():
    print(f"Label: {label}, Count: {count}")