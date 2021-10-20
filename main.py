import ipdb # remove if needed
from tqdm import tqdm # remove if needed
import subprocess # remove if needed
import os
import cv2
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

def lucas_kanade(args, pyramid_lk=False):
    # predictions is a list containing strings of bounding boxes | right now it just contains the template coordinates
    predictions, frame_names, template_img, template_coord = read_dataset(args)
    template = get_patch(template_img, template_coord, gray=True)

    # idx = 1
    for img_name in tqdm(frame_names[1:]):
        image = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), img_name), 0)

        # preprocessing on the current frame
        # image = cv2.GaussianBlur(image,(11,11),0)

        # image, template, template_img all are gray-scale
        if pyramid_lk == False:
            warp_params = run_LK_algo(frame=image, template=template, template_coord=template_coord, iterations=args.iterations, args=args)
            W = get_Warp(warp_params=warp_params, transformation=args.transformation)
        else:
            num_pyr_lyrs = args.num_pyr_lyr - 1
            # iter_list = [args.iterations] * (num_pyr_lyrs+1)
            iter_list = args.iter_list.split(",")
            iter_list = [int(it) for it in iter_list]
            # for i in range(len(iter_list)):
                # iter_list[i] = iter_list[i] - args.iterations + i + 2
            image_list, template_list, template_coord_list = [image], [template], [template_coord]
            curr_image, curr_template, curr_template_coord = image, template, template_coord
            for i in range(num_pyr_lyrs):
                curr_image = cv2.pyrDown(curr_image)
                curr_template = cv2.pyrDown(curr_template)
                curr_template_coord = (np.array(curr_template_coord) * 1/2).astype(int).tolist()
                image_list.append(curr_image)
                template_list.append(curr_template)
                template_coord_list.append(curr_template_coord)

            warp_params, init_wp = None, True
            while len(image_list) != 0:
                image, template, template_coord = image_list.pop(), template_list.pop(), template_coord_list.pop()
                warp_params = run_LK_algo(frame=image, template=template, template_coord=template_coord, iterations=iter_list.pop(), args=args, warp_params=warp_params, init_wp=init_wp)
                init_wp = False
            W = get_Warp(warp_params=warp_params, transformation=args.transformation)

        # predict the template_coord using warp_params | this is geometric transformation
        point_1 = np.array(template_coord[0:2] + [1]).reshape(3,1)
        if args.transformation == 2:
            point_1 = np.dot(W, point_1).reshape(-1)
            point_1[0] /= point_1[2]
            point_1[1] /= point_1[2]
            point_1 = point_1[:-1].astype(int)
        else:
            point_1 = np.dot(W, point_1).astype(int).reshape(-1)

        x2, y2 = template_coord[0] + template_coord[2], template_coord[1] + template_coord[3]
        point_2 = np.array([x2, y2] + [1]).reshape(3,1)
        if args.transformation == 2:
            point_2 = np.dot(W, point_2).reshape(-1)
            point_2[0] /= point_2[2]
            point_2[1] /= point_2[2]
            point_2 = point_2[:-1].astype(int)
        else:
            point_2 = np.dot(W, point_2).astype(int).reshape(-1)

        predicted_bb = point_1.tolist() + (point_2 - point_1).tolist()
        predictions.append(",".join(map(str, predicted_bb)))

        # update the template, template_img, and template_coord
        # if idx%10 == 0:
        #     template_coord[0], template_coord[1], template_coord[2], template_coord[3] = point_1[0], point_1[1], point_2[0] - point_1[0], point_2[1] - point_1[1]
        #     template_img = image
        #     template = get_patch(template_img, template_coord)
        # idx += 1

    save_predictions(predictions, args)

def pyramid_lk(args):
    lucas_kanade(args, pyramid_lk=True)

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