import os
import sys
import cv2
import pandas as pd
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

# print numpy arrays without truncation
# np.set_printoptions(threshold=sys.maxsize)

DIR = "/windows/Users/thats/Documents/ocr-repo-files"
DATA = "dataset2/Img"  # set directory path
FEATURE_DIR = f"{DIR}/features"

os.makedirs(FEATURE_DIR, exist_ok=True)


# deskewing images means to remove the natural slant that some people
#   have in their writing.
# Is a useful preprocessing thing so the model doesn't learn people's
#   writing quirks
# def deskew(img):
#     m = cv2.moments(img)
#     if abs(m['mu02']) < 1e-2 :
#     # no deskewing needed.
#         return img.copy()
#     # Calculate skew based on central momemts.
#     skew = m['mu11']/m['mu02']
#     # Calculate affine transform to correct skewness.
#     M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
#     # Apply affine transform
#     img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
#     return img

winSize = (450,600) # size of inputted images
cellSize = (30,40) # (250/30, 600/30)
blockSize = (60,80) # typically set to 2*cellSize
blockStride = (30,40) # typically set to 50% of blockSize.
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = False

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
    cellSize,nbins,derivAperture,
    winSigma,histogramNormType,L2HysThreshold,
    gammaCorrection,nlevels, useSignedGradients)

# for entry in os.scandir(directory):  
    # if entry.is_file():  # check if it's a file
        # print(entry.path)
im_test =  cv2.imread(f'{DATA}/img023-034.png',0)
descriptor = hog.compute(im_test)
# descriptor.shape
# print(hog.getDescriptorSize())
# print(descriptor.shape)
# im_test = cv2.resize(im_test, (0,0), fx=0.5, fy=0.5) 
# print(im_test.shape)
# _,hog_img = hog(im_test,orientations=9,pixels_per_cell=(8,8), cells_per_block=(1, 1),visualize=True)
# print(hog_img.shape)

# cv2.imshow('', hog_img)#, cmap='gray')
# cv2.imshow("", im_test)
# cv2.waitKey(0) # == ord('q')

def gen_hog_features(input_dir:str, output_dir:str,
                        overwrite_prev_files: bool=False) -> None :
    for file in os.scandir(input_dir) :
        if file.is_file() :
            npy_path = os.path.join(output_dir, f"{file.name.strip('.png')}.npy")
            if not overwrite_prev_files :
                if os.path.exists(npy_path) :
                    print(f"Features for {file.name} already exist, skipped...")
                    continue
            img = cv2.imread(f"{input_dir}/{file.name}",0)
            descriptor = hog.compute(img)
            np.save(npy_path, descriptor)
            print(f"Saved HOG features: {npy_path}")
# gen_hog_features(DATA, FEATURE_DIR)

def gen_hog_labels(csv_file:str, feature_dir:str, output_dir:str,
                        overwrite_prev_file: bool=False) -> None :
    """Assumes alphabetical/numerical ordering when generating labels;
       files which are processed first will have their label further 
       ahead in the label array."""
    npy_path = os.path.join(output_dir, "ordered_labels.npy")
    if not overwrite_prev_file and os.path.exists(npy_path) :
        print(f"Labels already exist. Set overwrite_prev_file to True to re-generate labels.")
        return
    df = pd.read_csv(csv_file, sep=",")
    df["image"] = df["image"].map(lambda x: x.lstrip("Img/").rstrip("png"))
    df["image"] = df["image"].map(lambda x: x + "npy")
    # idx, cols = pd.factorize(df['col'])
    # print(df[df["image"].isin(["img031-031.npy"])].iloc[0]["label"])
    # return
    # print(df)
    labels = []
    for file in os.scandir(feature_dir) :
        if file.is_file() :
            # print(df["image"].isin([file.name]).any())
            if df["image"].isin([file.name]).any() :
                # print(df[df["image"].isin([file.name])].iloc[0]["label"])
                labels.append(df[df["image"].isin([file.name])].iloc[0]["label"]) # gets label corresponding to file
                print(f"label {df[df['image'].isin([file.name])].iloc[0]['label']} added to labels.")
    np.save(npy_path, np.array(labels))

gen_hog_labels("dataset2/english.csv", FEATURE_DIR, FEATURE_DIR, True)

