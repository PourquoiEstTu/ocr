import os
import sys
import cv2
import pandas as pd
import numpy as np

# print numpy arrays without truncation
# np.set_printoptions(threshold=sys.maxsize)

# any of these can be changed to reflect your own directories
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
cellSize = (15,20) # (450/30, 600/30)
blockSize = (30,40) # typically set to 2*cellSize
blockStride = (15,20) # typically set to 50% of blockSize.
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

im_test =  cv2.imread(f'{DATA}/img023-034.png',0)
descriptor = hog.compute(im_test)

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
    labels = []
    for file in os.scandir(feature_dir) :
        if file.is_file() :
            if df["image"].isin([file.name]).any() :
                labels.append(df[df["image"].isin([file.name])].iloc[0]["label"]) # gets label corresponding to file
                print(f"label {df[df['image'].isin([file.name])].iloc[0]['label']} added to labels.")
    np.save(npy_path, np.array(labels))

# gen_hog_labels("dataset2/english.csv", FEATURE_DIR, FEATURE_DIR, True)

# 2000+ files being loaded breaks my computer, so added nfiles param :(
def concatenate_features(feature_dir: str, nfiles: int=100) -> np.ndarray :
    """Takes a directory with npy files containing 1D arrays 
       and concatenates them into one array"""
    X = []
    file_count = 0
    for file in os.scandir(feature_dir) :
        if ( file.name != "ordered_labels.npy" ) and ( file.is_file() ) :
            X.append( np.load(f"{feature_dir}/{file.name}") )
        file_count += 1
        if file_count >= nfiles :
            break
    return np.array(X)
# concatenate_features(FEATURE_DIR)

# horrendous function name
def get_same_length_features_and_labels(label_file: str, feature_dir: str,
            start_idx: int=0, end_idx: int=100) -> (np.ndarray, np.ndarray) :
    """Specify some starting index and ending index to get the features
       from feature_dir in that range along with their corresponding
       labels"""
    features = concatenate_features(feature_dir, end_idx)[start_idx:]
    truncated_labels = np.load(label_file)[start_idx:end_idx]
    return ( features, truncated_labels )
# print(len(get_same_length_features_and_labels(
#     f"{FEATURE_DIR}/ordered_labels.npy", FEATURE_DIR, 100)[0]) )
# print(len(get_same_length_features_and_labels(
#     f"{FEATURE_DIR}/ordered_labels.npy", FEATURE_DIR, 100)[1]) )
