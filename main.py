import ipdb # remove if needed
from tqdm import tqdm # remove if needed
import subprocess # remove if needed
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import *

def appearence_based_tracking(args):
    predictions, frame_names, template_img, template_coord = read_dataset(args)
    template = get_patch(template_img, template_coord, gray=True)
    
    for img_name in tqdm(frame_names[1:]):
        image = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), img_name), 0)
        image = cv2.GaussianBlur(image,(11,11),0)

        if(args.m1_method == 'ncc'):
            result = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
            cv2.normalize(result,result,cv2.NORM_MINMAX)
            _,_,_,maxLoc = cv2.minMaxLoc(result)
            (pred_x,pred_y) = maxLoc
        else:
            result = cv2.matchTemplate(image,template,cv2.TM_SQDIFF)
            cv2.normalize(result,result,cv2.NORM_MINMAX)
            _,_,minLoc,_ = cv2.minMaxLoc(result)
            (pred_x,pred_y) = minLoc

        pred_w = template_coord[2]  
        pred_h = template_coord[3] 
        
        # predicted tuple
        predicted_bb = [pred_x, pred_y, pred_w, pred_h]
        predictions.append(",".join(map(str, predicted_bb)))

    save_predictions(predictions, args)

def lucas_kanade(args):
    # predictions is a list containing strings of bounding boxes | right now it just contains the template coordinates
    predictions, frame_names, template_img, template_coord = read_dataset(args)
    template = get_patch(template_img, template_coord, gray=True)

    for img_name in tqdm(frame_names[1:]):
        image = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), img_name), 0)

        # preprocessing on the current frame
        image = cv2.GaussianBlur(image, (3,3),0) # @aditi: check this hyperparams; we may remove this if useless 
        # further pre-processing required ? like normalization ??

        # image, template, template_img all are gray-scale
        warp_params = run_LK_algo(frame=image, template=template, template_coord=template_coord, args=args)
        W = get_Warp(warp_params=warp_params, transformation=args.transformation)

        # predict the template_coord using warp_params | this is geometric transformation
        point_1 = np.array(template_coord[0:2] + [1]).reshape(3,1)
        point_1 = np.dot(W, point_1).astype(int).reshape(-1)
        x2, y2 = template_coord[0] + template_coord[2], template_coord[1] + template_coord[3]
        point_2 = np.array([x2, y2] + [1]).reshape(3,1)
        point_2 = np.dot(W, point_2).astype(int).reshape(-1)

        predicted_bb = point_1.tolist() + (point_2 - point_1).tolist()
        predictions.append(",".join(map(str, predicted_bb)))

        # update the template, template_img, and template_coord
        # template_coord[0], template_coord[1], template_coord[2], template_coord[3] = point_1[0], point_1[1], point_2[0] - point_1[0], point_2[1] - point_1[1]
        # template_img = image
        # template = get_patch(template_img, template_coord)

    save_predictions(predictions, args)

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

    if args.eval:
        print("evaluating...")
        subprocess.call("python eval.py -g %s/groundtruth_rect.txt -p %s/predictions_part%s.txt"%(args.data_dir, args.data_dir, args.method), shell=True)