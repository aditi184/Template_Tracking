import ipdb # remove if needed
from tqdm import tqdm # remove if needed
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Template Tracking')
    parser.add_argument('-i', '--data_dir', type=str, default='Liquor', required=True, \
        help="Path to the dataset")
    parser.add_argument('-m', '--method', type=int, default=3, \
        help="method of template tracking: 1-appearence, 2-LK, 3-PyramidLK")
    parser.add_argument('-t', '--technique',type = str,default = 'ncc' , help = "technique for appearance based tracking")
    args = parser.parse_args()
    return args

def get_patch(image, bb, gray = True):
    x, y, w, h = bb[0], bb[1], bb[2], bb[3] # w is along x (right), h is along y (down)
    return image[y:y+h, x:x+w] if gray == True else image[y:y+h, x:x+w, :]

def save_predictions(predictions, name):
    # input a list of tuples
    predictions = "\n".join(predictions)+"\n"
    write_file = open("predictions_part%s.txt"%(str(name)), "w")
    write_file.writelines(predictions)
    print("predictions saved...!")

def show_img(img):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()