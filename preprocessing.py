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

MNIST_DIR = f"{DIR}/mnist"
MNIST_DATA = f"{DIR}/mnist/dataset/v011_words_small"
MNIST_JSON = f"{DIR}/mnist/v011_labels_small.json"

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
cellSize = (45,60) # winSize / 10
blockSize = (90,120) # typically set to 2*cellSize
blockStride = (45,60) # typically set to 50% of blockSize.
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = False

# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
#     cellSize,nbins,derivAperture,
#     winSigma,histogramNormType,L2HysThreshold,
#     gammaCorrection,nlevels, useSignedGradients)

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
# gen_hog_features(DATA, FEATURE_DIR, True)

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

# with winSize / 50, 2000+ files being loaded breaks my computer, 
#   so added nfiles param :(
# with winSize / 10, 3100 files can be loaded
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
# concatenate_features(FEATURE_DIR,)

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

def segment_word(img_path: str) :
    """Takes a path to an image with a single word in it and outputs 
       a group of images with each character in the word on a simple
       image."""
    og_img = cv2.imread(img_path)
    img = cv2.cvtColor(og_img,cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img,(3, 3), 0)
    img = cv2.medianBlur(img, 3)
    _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    # img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    # thresh_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img = cv2.dilate(img, (0.5,0.5), iterations=20)
    # img = cv2.erode(img, (3, 15), iterations=15)

    contours,hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    def sorting_criteria(l) : # sort contours by first coordinate stored
        return l[0][0][0]
    contours.sort(key=sorting_criteria)
    characters = []
    # print(contours)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        characters.append(og_img[y:y+h, x:x+w])
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # visualization
        # cv2.rectangle(og_img,(x,y),(x+w,y+h),(0,255,0),1)
    # cv2.imshow('', img)
    # cv2.imshow('', og_img)
    # cv2.imshow('', thresh_color)
    # for c in characters :
    #     cv2.imshow('', c)
    #     cv2.waitKey(0)
    # cv2.waitKey(0)
    return characters
# segment_word(f"{MNIST_DATA}/4.png")
# segment_word(f"{MNIST_DATA}/28.png")
# segment_word(f"{MNIST_DATA}/27.png")
# segment_word(f"{MNIST_DATA}/31.png")
# segment_word(f"{MNIST_DATA}/26.jpeg")
# segment_word(f"{MNIST_DATA}/50.jpeg") # does not work very good on this
# segment_word(f"{MNIST_DATA}/53.png")
# segment_word(f"{MNIST_DATA}/61.png")
# segment_word(f"{MNIST_DATA}/63.jpeg") # appears to not work too well on u's and y's?
# segment_word(f"{MNIST_DATA}/65.png")

# implements character segmentation in the 'A New Character Segmentation Approach 
#   for Off-Line Cursive Handwritten Words' paper at 
#   https://www.researchgate.net/publication/257719290_A_New_Character_Segmentation_Approach_for_Off-Line_Cursive_Handwritten_Words"""
def potential_segmentation_columns(img_path: str) :
    og_img = cv2.imread(img_path)
    img = cv2.cvtColor(og_img,cv2.COLOR_BGR2GRAY)
    h,w = img.shape
    print(f"height = {h}, width = {w}")
    seg_cols = np.zeros(w) # entry i is 1 if i is a potential segmentation column (pcs)
    # magic number to decide whether a column has enough black pixels to be 
    #   a pcs
    seg_threshold = 0.023*h # usually is around < 1 
    # print(h,w)
    # return
    # img = cv2.GaussianBlur(img, (3,3),0)
    # img = cv2.bilateralFilter(img, 15, 41, 21)
    # img = cv2.medianBlur(img, 1) # little too strong on some images
    _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # print(img[15,10])
    # thresh_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # img = cv2.erode(img, (3, 3), iterations=2)
    img = cv2.ximgproc.thinning(img, cv2.ximgproc.THINNING_ZHANGSUEN)
    for i in range(w) :
        sum = 0
        for j in range(h) :
            sum += img[j,i] // 255 # make sure white pixels have value 1
            # cv2.rectangle(drawing,(i,j),(i+10,j+10),(100,100,0),1) # visualization
        if sum < seg_threshold :
            seg_cols[i] = 1
            # cv2.line(og_img, (i,0), (i,h-1), (0,0,255),2)
    print(seg_cols)
    # remove pcs at beginning of image
    for i in range(w) :
        if seg_cols[i] == 1 :
            seg_cols[i] = 0
        else :
            break
    # remove pcs at end of image
    for i in range(w-1, -1, -1) :
        if seg_cols[i] == 1 :
            seg_cols[i] = 0
        else :
            break
    # average out block of columns into one columns
    for i in range(w) :
        if seg_cols[i] == 1 :
            num_of_consecutive_cols = 1
            for j in range(i,w) :
                if seg_cols[i] == 1 :
                    # finish tmr
    # print(seg_cols)
    # cv2.waitKey(0)
    while 1 :
        cv2.imshow('', og_img)
        k = cv2.waitKey(100000)
        if k==27:    # Esc key to stop
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue
# potential_segmentation_columns(f"{MNIST_DATA}/4.png")
# potential_segmentation_columns(f"{MNIST_DATA}/14.png")
# potential_segmentation_columns(f"{MNIST_DATA}/39.jpeg")
# potential_segmentation_columns(f"{MNIST_DATA}/28.png")
# potential_segmentation_columns(f"{MNIST_DATA}/27.png")
# potential_segmentation_columns(f"{MNIST_DATA}/31.png")
potential_segmentation_columns(f"{MNIST_DATA}/26.jpeg")
# potential_segmentation_columns(f"{MNIST_DATA}/50.jpeg")
# potential_segmentation_columns(f"{MNIST_DATA}/53.png")
#potential_segmentation_columns(f"{MNIST_DATA}/61.png")
#potential_segmentation_columns(f"{MNIST_DATA}/63.jpeg") 
#potential_segmentation_columns(f"{MNIST_DATA}/65.png")
