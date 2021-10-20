import os
import cv2
import pandas
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='make video using frames and corresponding bounding boxes')
    parser.add_argument('--data_dir', type=str) # data directory should contain frames in "img" sub-directory
    parser.add_argument('--gold', type=str) # path to txt file containing gold bounding boxes 
    parser.add_argument('--pred', type=str) # path to txt file containing predicted bounding boxes 
    parser.add_argument('--video_name', type=str, default="video")
    args = parser.parse_args()
    return args

def main(args):
    # bounding box
    bb_gold = pandas.read_csv(args.gold, header=None)
    bb_pred = pandas.read_csv(args.pred, header=None)

    # video sequence
    frame_names = os.listdir(os.path.join(args.data_dir, "img"))
    frame_names.sort()
    video_frames = []
    for idx, img_name in enumerate(frame_names):
        img = cv2.imread(os.path.join(os.path.join(args.data_dir, "img"), img_name))
        h, w, ch = img.shape
        size = (w, h)

        x1 = bb_gold.iloc[idx,0] 
        y1 = bb_gold.iloc[idx,1]
        x2 = x1 + bb_gold.iloc[idx, 2]
        y2 = y1 + bb_gold.iloc[idx, 3]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,0,255), thickness=2) # red
        x1 = bb_pred.iloc[idx,0] 
        y1 = bb_pred.iloc[idx,1]
        x2 = x1 + bb_pred.iloc[idx, 2]
        y2 = y1 + bb_pred.iloc[idx, 3]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(255,0,0), thickness=2) # blue
        video_frames.append(img)

    video_output = cv2.VideoWriter(os.path.join(args.data_dir, args.video_name + ".mp4"), cv2.VideoWriter_fourcc(*'XVID'), 30, size)
    for video_frame in video_frames:
        video_output.write(video_frame)
    video_output.release()

if __name__ == "__main__":
    args = parse_args()
    main(args)