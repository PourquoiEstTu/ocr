import pandas as pd
import os
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

df = pd.read_csv("dataset2/english.csv", sep=",")
# print(df)

directory = 'dataset2/Img'  # set directory path

# for entry in os.scandir(directory):  
    # if entry.is_file():  # check if it's a file
        # print(entry.path)
im_test =  cv2.imread(f'{directory}/img062-055.png',0)
_,hog_img = hog(im_test,orientations=9,pixels_per_cell=(8,8), cells_per_block=(1, 1),visualize=True)

cv2.imshow('', hog_img)#, cmap='gray')
cv2.waitKey(0) # == ord('q')
