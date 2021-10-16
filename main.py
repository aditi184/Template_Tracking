import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Template Tracking')
    parser.add_argument('-i', '--data_dir', type=str, default='Liquor', required=True, \
                                                        help="Path to the dataset")
    args = parser.parse_args()
    return args


def show_img(img):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()


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
    templateCoordinates = ground_truth[0][:-1].split(",") # (x,y,w,h)

    x1 = int(templateCoordinates[0])
    y1 = int(templateCoordinates[1])
    width = int(templateCoordinates[2])
    height = int(templateCoordinates[3])
    x2 = x1 + width
    y2 = y1 + height
    template = templateImage[y1:y2,x1:x2]
    # show_img(template)
    c = 0
    for idx, i in enumerate(allimgs):
        res = cv2.matchTemplate(i,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.5
        loc = np.where( res >= threshold)
        # print("LOC = " , loc)
        # print("star LOC = " , *loc)
        # print("IN FOR LOOP LOC = " , *loc[::-1])
        for pt in zip(*loc[::-1]):
            cv2.rectangle(i, pt, (pt[0] + width, pt[1] + height), (0,255,255), 1)
        
        pred_x = pt[0]
        # print("_____________________________________" , idx , "_____________________________" )
        # print(pred_x)
        pred_y = pt[1]
        pred_w =  width + 1
        pred_h =  height + 1
        outputString = str(pred_x)  + ',' + str(pred_y)  + ',' + str(pred_w) + ',' + str(pred_h)
        outputFile = open('output.txt', 'a')    
        outputFile.write(outputString)
        outputFile.write('\n')
        # c = c+1
        # if(c==20):
        #     break
    outputFile.close()




if __name__ == "__main__":
    args = parse_args()
    main(args)




