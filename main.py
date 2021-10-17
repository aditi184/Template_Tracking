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
    template_coord_str = template_coord
    
    for i in range(4):
        template_coord[i] = int(template_coord[i])

    # add first bounding box or the template coordinates although it is not included in mIOU calculation
    predictions.append(",".join(map(str,template_coord_str)))

    # template from template image (cropped)
    template_img = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), frame_names[0]),0)
    template = get_patch(template_img, template_coord)
     
    for img_name in tqdm(frame_names[1:]):
        image = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), img_name),0)
        image = cv2.GaussianBlur(image,(3,3),0)

        if(args.technique == 'ncc'):
            result = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
            cv2.normalize(result,result,cv2.NORM_MINMAX)
            _,_,_,maxLoc = cv2.minMaxLoc(result)
            (pred_x,pred_y) = maxLoc
        else:
            result = cv2.matchTemplate(image,template,cv2.TM_SQDIFF)
            # cv2.normalize(result,result,cv2.NORM_MINMAX,-1)
            _,_,minLoc,_ = cv2.minMaxLoc(result)
            (pred_x,pred_y) = minLoc

        pred_w = template_coord[2]  
        pred_h = template_coord[3] 
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
