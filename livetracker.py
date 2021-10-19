import cv2

def appearence_based_tracking(template,img):
        
    result = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    # cv2.normalize(result,result,cv2.NORM_MINMAX)
    _,_,_,maxLoc = cv2.minMaxLoc(result)
    (pred_x,pred_y) = maxLoc
    pred_w = template.shape[1]  
    pred_h = template.shape[0] 
    # predicted tuple
    return pred_x, pred_y, pred_w, pred_h

def main():
    camera = cv2.VideoCapture(0)
    print("ADITI KHANDELWAL")
    template_img = cv2.imread("A2\\LiveTrack1.png")
    print(template_img)
    while True:
        print("HEELLOO")
        _,img = camera.read()
        x,y,w,h = appearence_based_tracking(template_img,img)
        img = cv2.rectangle(img,(x,y),(x+w+1,y+h+1),(255,0,0),1 )
        cv2.imshow("Tracking",img)
        if cv2.waitKey(2) and 0xff==ord('q'):
            break

if __name__ == "__main__":
    main()