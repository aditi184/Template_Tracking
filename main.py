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

    # ground truth - only for getting template
    ground_truth = open(os.path.join(args.data_dir, "groundtruth_rect.txt"), 'r').readlines()
    template = ground_truth[0][:-1].split(",") # (x,y,w,h)

if __name__ == "__main__":
    args = parse_args()
    main(args)