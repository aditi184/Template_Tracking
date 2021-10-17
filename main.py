import ipdb # remove if needed
from tqdm import tqdm # remove if needed
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import *

def appearence_based_tracking(args):
    # predictions of bounding boxes
    predictions = []
    
    # video sequence
    frame_names = os.listdir(os.path.join(args.data_dir, "img"))
    frame_names.sort()
    
    # ground truth - only for getting template
    ground_truth = open(os.path.join(args.data_dir, "groundtruth_rect.txt"), 'r').readlines()
    template_coord = ground_truth[0][:-1].split(",") # (x,y,w,h)
    for i in range(4):
        template_coord[i] = int(template_coord)

    # add first bounding box or the template coordinates although it is not included in mIOU calculation
    predictions.append(",".join(template_coord))

    # template from template image (cropped)
    template_img = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), frame_names[0]))
    template = get_patch(template_img, template_coord)
    
    for img_name in tqdm(frame_names[1:]):
        image = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), img_name) )

        '''
        @aditi:
            1. de-plag this code
            2. instead of using threshold, get the path with min SSD (somanshu has done this)
            3. we also need to do this with NCC metric
        '''

        res = cv2.matchTemplate(i,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(i, pt, (pt[0] + width, pt[1] + height), (0,255,255), 1)
        pred_x = pt[0]
        pred_y = pt[1]
        pred_w = pt[0] + width   
        pred_h = pt[1] + height 

        # predicted tuple
        predictions.append(str(pred_x)  + ',' + str(pred_y)  + ',' + str(pred_w) + ',' + str(pred_h))

    save_predictions(predictions, name = args.method)

def lucas_kanade(args):
    pass 

def pyramid_lk(args):
    pass

if __name__ == "__main__":
    args = parse_args()
    if args.method not in [1,2,3]:
        raise ValueError("method should be one of 1/2/3 - Found: %u"%args.method)
    FUNCTION_MAPPER = {
            1: appearence_based_tracking,
            2: lucas_kanade,
            3: pyramid_lk,
        }

    print("using %s for template tracking"%(FUNCTION_MAPPER[args.method]))
    FUNCTION_MAPPER[args.method](args)