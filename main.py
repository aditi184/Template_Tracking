import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Template Tracking')
    parser.add_argument('-i', '--data_dir', type=str, default='Liquor', required=True, \
                                                        help="Path to the dataset")
    args = parser.parse_args()
    return args

def main(args):
    # video sequence
    frame_names = os.listdir(os.path.join(args.data_dir, "img"))
    frame_names.sort()
    allimgs = []
    for img_name in frame_names:
        img = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), img_name)) 
        allimgs.append(img)
    imgs = allimgs[1:]
    templateImage = allimgs[0]
    
    # ground truth - only for getting template
    ground_truth = open(os.path.join(args.data_dir, "groundtruth_rect.txt"), 'r').readlines()
    template = ground_truth[0][:-1].split(",") # (x,y,w,h)

if __name__ == "__main__":
    args = parse_args()
    main(args)