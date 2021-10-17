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

def read_dataset(args):
    # predictions of bounding boxes
    predictions = []
    
    # video sequence
    frame_names = os.listdir(os.path.join(args.data_dir, "img"))
    frame_names.sort()
    
    # ground truth - only for getting template
    ground_truth = open(os.path.join(args.data_dir, "groundtruth_rect.txt"), 'r').readlines()
    template_coord = ground_truth[0][:-1].split(",") # (x,y,w,h)
    # str to int
    for i in range(4):
        template_coord[i] = int(template_coord[i])

    # add first bounding box or the template coordinates although it is not included in mIOU calculation
    predictions.append(",".join(map(str,template_coord)))

    # template from template image (cropped)
    template_img = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), frame_names[0]), 0) # 0 denotes gray scale

    return predictions, frame_names, template_img, template_coord

def save_predictions(predictions, name):
    # input a list of tuples
    predictions = "\n".join(predictions)+"\n"
    write_file = open("predictions_part%s.txt"%(str(name)), "w")
    write_file.writelines(predictions)
    print("predictions saved...!")

def show_img(img):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()