import ipdb
from numpy.lib.type_check import imag # remove if needed
from tqdm import tqdm # remove if needed
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Template Tracking')
    
    parser.add_argument('-i', '--data_dir', type=str, default='Liquor', required=True, help="Path to the dataset")
    parser.add_argument('-m', '--method', type=int, default=1, help="method of template tracking: 1-appearence, 2-LK, 3-PyramidLK")
    parser.add_argument('-e', '--eval', action="store_true", help="evaluate or not")

    parser.add_argument('-m1', '--m1_method', type=str, default = 'ncc', help = "method for appearance based tracking")
    parser.add_argument('-tf', '--transformation', type=int, default=1, help = "transformation: 0-translation, 1-affine, 2-projective")
    parser.add_argument('--iterations', type=int, default=10, help="max iterations for LK")
    parser.add_argument('--num_pyr_lyr', type=int, default=3, help="number of pyramid layers") # one means simple LK
    parser.add_argument('--iter_list', type=str, help="iteration list for each level in pyramid-LK")

    args = parser.parse_args()
    return args

def run_LK_algo(frame, template, template_coord, iterations, args, warp_params=None, init_wp=True, epsilon=0.001):
    # frame and template are gray scale
    (h, w) = frame.shape

    # initialize warp_params such that W = get_Warp(warp_params) is Identity matrix
    warp_params = get_init_warp_params(args.transformation) if init_wp == True else warp_params

    for i in range(iterations):
        # 1. warp the frame with W(warp_params)
        W = get_Warp(warp_params=warp_params, transformation=args.transformation)
        if args.transformation == 2:
            warped_frame = cv2.warpPerspective(src=frame, M=W, dsize=(frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC)
        else:
            warped_frame = cv2.warpAffine(src=frame, M=W, dsize=(frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC) #flags=cv2.WARP_INVERSE_MAP
        warped_patch = get_patch(warped_frame, template_coord, gray=True).astype(np.uint8)

        # 2. Get the error between T(x) and I(W(x;p))
        assert template.shape == warped_patch.shape
        error = template - warped_patch

        # 3. compute warped gradients Del I, evaluated at W(x;p)
        # Reference: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
        # find gradients
        sobel_x, sobel_y = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
        # warp the gradient-image
        if args.transformation == 2:
            sobel_x, sobel_y = cv2.warpPerspective(sobel_x, W, (w, h), flags=cv2.INTER_CUBIC), cv2.warpPerspective(sobel_y, W, (w, h), flags=cv2.INTER_CUBIC)
        else:
            sobel_x, sobel_y = cv2.warpAffine(sobel_x, W, (w, h), flags=cv2.INTER_CUBIC), cv2.warpAffine(sobel_y, W, (w, h), flags=cv2.INTER_CUBIC)
        
        # get the warped-gradient-patch
        sobel_x, sobel_y = get_patch(sobel_x, template_coord), get_patch(sobel_y, template_coord)
        del_I = np.expand_dims(np.stack([sobel_x, sobel_y], axis=2), axis=2)

        # 4. evaluate the jacobian of the warping
        warp_jacobian = get_warp_jacobian(shape=template.shape, transformation=args.transformation, warp_params=warp_params) # (template.shape,2,6)

        # 5. Compute the steepest descent using Jacobian and Image(frame) gradient
        stp_dsc = np.matmul(del_I, warp_jacobian) # (template.shape,1,6)

        # 6. Compute Inverse Hessian
        hessian = np.matmul(stp_dsc.transpose(0,1,3,2), stp_dsc) # (template.shape,6,6)
        hessian = np.sum(hessian, axis=(0,1))
        # assert hessian.shape == (6,6)
        H_inv = np.linalg.pinv(hessian)

        # 7. Multiply Steepest descent with the error
        error = np.expand_dims(np.expand_dims(error, axis=2), axis=2)
        intermediate = np.matmul(stp_dsc.transpose(0,1,3,2), error) # (template.shape,6,1)
        intermediate = np.sum(intermediate, axis=(0,1))

        # 8. Compute del_warp_params 
        del_warp_params = np.matmul(H_inv, intermediate).reshape(-1)

        # 9. update warp_params
        warp_params += del_warp_params

        # 10. repeat until del_warp_params < epsilon
        if np.linalg.norm(del_warp_params) < epsilon:
            # print("norm of del_warp_params less than epsilon, and equal to", np.linalg.norm(del_warp_params))
            break

    return warp_params

def get_patch(image, bb, gray = True):
    x, y, w, h = bb[0], bb[1], bb[2], bb[3] # w is along x (right), h is along y (down)
    return image[y:y+h+1, x:x+w+1] if gray == True else image[y:y+h+1, x:x+w+1, :]

def get_init_warp_params(transformation=1):
    warp_params = None
    if transformation == 0:
        # translation transformation
        warp_params = np.zeros(2)
    elif transformation == 1:
        # affine transformation
        warp_params = np.zeros(6)
    elif transformation == 2:
        # projective transformation
        warp_params = np.zeros(9)
    return warp_params

def get_Warp(warp_params, transformation=1):
    W = None
    if transformation == 0:
        # translation transformation
        W = np.array([
            [1, 0, warp_params[0]],
            [0, 1, warp_params[1]]
        ])
    elif transformation == 1:
        # affine transformation
        W = np.array([
            [1+warp_params[0], warp_params[2], warp_params[4]],
            [warp_params[1], 1+warp_params[3], warp_params[5]]
        ]) # this ordering matters while computing the jacobian wrt (p0, p1, p2, p3, p4, p5)
    elif transformation == 2:
        # projective transformation
        W = np.array([
            [1+warp_params[0], warp_params[3], warp_params[6]],
            [warp_params[1], 1+warp_params[4], warp_params[7]],
            [warp_params[2], warp_params[5], 1+warp_params[8]]
        ]) # this ordering matters while computing the jacobian wrt (p0, p1, p2, p3, p4, p5, p6, p7, p8)
    return W

def get_warp_jacobian(shape, transformation, warp_params):
    jacobian = None
    (h, w) = shape

    ones_patch = np.ones(shape)
    zeros_patch = np.zeros(shape)
    img_patch = np.meshgrid(np.arange(0,w), np.arange(0,h))

    # since we will be using matrix multiplication where matrices are 4-D and contain elements as matrices itself
    # following np.matmul documentation 

    if transformation == 0:
        # translation transformation
        # grad_W = [
        #     [1, 0],
        #     [0, 1]
        # ]
        jacobian_x = np.stack([ones_patch, zeros_patch], axis=2)
        jacobian_y = np.stack([zeros_patch, ones_patch], axis=2)
        jacobian = np.stack([jacobian_x, jacobian_y], axis=2)
        assert len(jacobian.shape) == 4 and jacobian.shape[-2:] == (2,2)

    elif transformation == 1:
        # affine transformation
        # grad_W = [
        #     [x, 0, y, 0, 1, 0],
        #     [0, x, 0, y, 0, 1]
        # ]
        jacobian_x = np.stack([img_patch[0], zeros_patch, img_patch[1], zeros_patch, ones_patch, zeros_patch], axis=2)
        jacobian_y = np.stack([zeros_patch, img_patch[0], zeros_patch, img_patch[1], zeros_patch, ones_patch], axis=2)
        jacobian = np.stack([jacobian_x, jacobian_y], axis=2)
        assert len(jacobian.shape) == 4 and jacobian.shape[-2:] == (2,6)

    elif transformation == 2:
        # projective tranformation
        # grad_W = [
        #     [],
        #     []
        # ]
        numerator_patch = (1 + warp_params[0]) * img_patch[0] + warp_params[3] * img_patch[1] + warp_params[6]
        denominator_patch = warp_params[2] * img_patch[0] + warp_params[5] * img_patch[1] + 1 + warp_params[8]
        
        jacobian_x = [img_patch[0]/denominator_patch, zeros_patch, -1 * numerator_patch * img_patch[0]/ (denominator_patch**2)]
        jacobian_x+= [img_patch[1]/denominator_patch, zeros_patch, -1 * numerator_patch * img_patch[1]/ (denominator_patch**2)]
        jacobian_x+= [1/denominator_patch, zeros_patch, -1 * numerator_patch / (denominator_patch**2)]
        jacobian_x = np.stack(jacobian_x, axis=2)

        jacobian_y = [zeros_patch, img_patch[0]/denominator_patch, -1 * numerator_patch * img_patch[0]/ (denominator_patch**2)]
        jacobian_y+= [zeros_patch, img_patch[1]/denominator_patch, -1 * numerator_patch * img_patch[1]/ (denominator_patch**2)]
        jacobian_y+= [zeros_patch, 1/denominator_patch, -1 * numerator_patch / (denominator_patch**2)]
        jacobian_y = np.stack(jacobian_y, axis=2)

        jacobian = np.stack([jacobian_x, jacobian_y], axis=2)
        assert len(jacobian.shape) == 4 and jacobian.shape[-2:] == (2,9)
    return jacobian

def read_dataset(args):
    # predictions of bounding boxes
    predictions = []
    
    # video sequence
    frame_names = os.listdir(os.path.join(args.data_dir, "img"))
    frame_names.sort()
    
    # ground truth - only for getting template
    ground_truth = open(os.path.join(args.data_dir, "groundtruth_rect.txt"), 'r').readlines()
    template_coord = ground_truth[0][:-1].split(",") # [x,y,w,h]
    # str to int
    for i in range(4):
        template_coord[i] = int(template_coord[i])

    # add first bounding box or the template coordinates although it is not included in mIOU calculation
    predictions.append(",".join(map(str,template_coord)))

    # template from template image (cropped)
    template_img = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), frame_names[0]), 0) # 0 denotes gray scale

    return predictions, frame_names, template_img, template_coord

def save_predictions(predictions, args):
    # input a list of tuples
    predictions = "\n".join(predictions)+"\n"
    
    write_file = open(os.path.join(args.data_dir, "predictions_part%s.txt"%(str(args.method))), 'w')
    write_file.writelines(predictions)
    print("predictions saved...!")

def show_img(img):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()