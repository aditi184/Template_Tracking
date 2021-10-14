import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Template Tracking Evaluation')
    parser.add_argument('-g', '--gold', type=str, required=True, \
                                                        help="Path to the gold rectangles")
    parser.add_argument('-p', '--prediction', type=str, required=True, \
                                                        help="Path to the predicted rectangles")
    args = parser.parse_args()
    return args

def main(args):
    gold_rects = open(args.gold, 'r').readlines()
    pred_rects = open(args.prediction, 'r').readlines()

    assert len(gold_rects) == len(pred_rects)

    mIOU = 0
    N = len(gold_rects)
    for i in range(1,N):
        gold = gold_rects[i].split(",")
        pred = pred_rects[i].split(",")

        for i in range(4):
            gold[i] = int(gold[i])
            pred[i] = int(pred[i])
        
        gold_rect_area = (1 + gold[2]) * (1 + gold[3])
        pred_rect_area = (1 + pred[2]) * (1 + pred[3])

        # intersection area
        x1 = max(gold[0], pred[0])
        y1 = max(gold[1], pred[1])
        x2 = min(gold[0]+gold[2], pred[0]+pred[2])
        y2 = min(gold[1]+gold[3], pred[1]+pred[3])
        intersxn_area = max(0, x2-x1+1) * max(0, y2-y1+1)

        mIOU += (intersxn_area/float(gold_rect_area+pred_rect_area-intersxn_area))

    mIOU /= (N-1)
    print("mIOU:", mIOU)

if __name__ == "__main__":
    args = parse_args()
    main(args)