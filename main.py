import ipdb # remove if needed
from tqdm import tqdm # remove if needed
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
    # predictions is a list containing strings of bounding boxes | right now it just contains the template coordinates
    predictions, frame_names, template_img, template_coord = read_dataset(args)
    template = get_patch(template_img, template_coord, gray=True)

    for img_name in tqdm(frame_names[1:]):
        image = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), img_name) )

        # preprocessing on the current frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # is it really needed from implementation perspective?
        image = cv2.GaussianBlur(image, (3,3)) # @aditi: check this hyperparams; we may remove this if useless 
        # further pre-processing required ? like normalization ??
        
        warp_params = run_LK_algo(frame=image, template=template, template_coord=template_coord)
        W = get_Warp(warp_params=warp_params, affine=True)

        # predict the template_coord using warp_params | this is geometric transformation
        point_1 = np.array(template_coord[0:2]).reshape(2,1)
        point_1 = np.dot(W, point_1).astype(np.int)
        x2, y2 = template_coord[0] + template_coord[2], template_coord[1] + template_coord[3]
        point_2 = np.array([x2, y2]).reshape(2,1)
        point_2 = np.dot(W, point_2).astype(np.int)

        predicted_bb = list(point_1) + list(point_2 - point_1)
        predictions.append(",".join(predicted_bb))

        # update the template, template_img, and template_coord
        template_coord[0], template_coord[1], template_coord[2], template_coord[3] = point_1[0], point_1[1], point_2[0] - point_1[0], point_2[1] - point_1[1]
        template_img = image
        template = get_patch(template_img, template_coord)

    save_predictions(predictions, name = args.method)

def run_LK_algo(frame, template, template_coord, iterations=100):
    (h, w) = frame.shape

    # initialize warp_params such that W = get_Warp(warp_params) is Identity matrix
    warp_params = np.zeros(6)

    for i in range(iterations):
        # 1. warp the frame with W(warp_params)
        W = get_Warp(warp_params=warp_params, affine=True)
        warped_frame = cv2.warpAffine(src=frame, M=W, dsize=(frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP) # flags = cv2.INTER_CUBIC # adding warp_inv_map as it should be IG
        warped_patch = get_patch(frame, template_coord, gray=True).astype(np.int)

        # 2. Get the error between T(x) and I(W(x;p))
        assert template.dtype == warp_params.dtype
        error = template - warped_patch

        # 3. compute warped gradients Del I, evaluated at W(x;p)
        # Reference: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
        sobel_x, sobel_y = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5) # find gradients
        sobel_x, sobel_y = cv2.warpAffine(sobel_x, W, (w, h)), cv2.warpAffine(sobel_y, W, (w, h)) # warp the gradient-image
        sobel_x, sobel_y = get_patch(sobel_x, template_coord), get_patch(sobel_y, template_coord) # get the warped-gradient-patch

        # 4. evaluate the jacobian of the warping

        # 5. Compute the steepest descent using Jacobian and Image(frame) gradient

        # 6. Compute Inverse Hessian

        # 7. Multiply Steepest descent with the error

        # 8. Compute del_warp_params 

        # 9. update warp_params

        # 10. repeat until del_warp_params < epsilon

    pass

def get_Warp(warp_params, affine=True):
    W = None
    if affine:
        W = [1+warp_params[0], warp_params[2], warp_params[4], warp_params[1], 1+warp_params[3], warp_params[5]] # de-plag
        W = np.array(W)
        W = W.reshape(2,3)
    
    return W

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
