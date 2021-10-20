import cv2
from utils import *

def appearence_based_tracking(template, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(11,11),0)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(img, template,cv2.TM_CCOEFF_NORMED)
    cv2.normalize(result,result,cv2.NORM_MINMAX)
    _,_,_,maxLoc = cv2.minMaxLoc(result)
    (pred_x,pred_y) = maxLoc
    pred_w = template.shape[1]  
    pred_h = template.shape[0] 
    # predicted tuple

    point1 = (pred_x, pred_y)
    point2 = (pred_x+pred_w, pred_y)
    point3 = (pred_x, pred_y+pred_h)
    point4 = (pred_x+pred_w, pred_y+pred_h)
    
    return point1, point2, point3, point4

def main():
    print("Live Tracking System...")
    print("Press ESC to quit")
    print("Press s to select template")
    
    camera = cv2.VideoCapture(0)
    tracking_started = False
    while True:
        ret, frame = camera.read()

        if cv2.waitKey(1) == 115:
            tracking_started = True
            print("select roi")
            bb = cv2.selectROI("live tracking", frame, showCrosshair=False, fromCenter=False)
            # x,y,w,h = bb
            template = get_patch(frame, bb, gray=False)
            # cv2.imshow('Template', template)

        if tracking_started:
            # get the 4 coordinates of predicted template using the best model
            # this needs the original template and the current frame
            point1, point2, point3, point4 = appearence_based_tracking(template, frame)

            # update the frame
            # frame = cv2.rectangle(frame, point1, point4, (0,0,255), 1)

            frame = cv2.line(frame, point1, point2, (255,0,0), 1)
            frame = cv2.line(frame, point2, point4, (255,0,0), 1)
            frame = cv2.line(frame, point3, point4, (255,0,0), 1)
            frame = cv2.line(frame, point1, point3, (255,0,0), 1)

        cv2.imshow('live tracking', frame)

        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()