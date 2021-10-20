import cv2
from utils import *

def appearence_based_tracking(template, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img,(11,11),0)
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

def lucas_kanade(template, image, template_coord, args):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(11,11),0)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    warp_params = run_LK_algo(frame=image, template=template, template_coord=template_coord, iterations=args.iterations, args=args)
    W = get_Warp(warp_params=warp_params, transformation=args.transformation)

    point1 = get_point(x = template_coord[0], y = template_coord[1], W=W, args=args)
    point2 = get_point(x = template_coord[0]+template_coord[2], y = template_coord[1], W=W, args=args)
    point3 = get_point(x = template_coord[0], y = template_coord[1]+template_coord[3], W=W, args=args)
    point4 = get_point(x = template_coord[0]+template_coord[2], y = template_coord[1]+template_coord[2], W=W, args=args)
    return point1, point2, point3, point4

def pyramid_lk(template, image, template_coord, args):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(11,11),0)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    num_pyr_lyrs = 5 - 1
    iter_list = [1, 5, 7, 10, 20]
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

    point1 = get_point(x = template_coord[0], y = template_coord[1], W=W, args=args)
    point2 = get_point(x = template_coord[0]+template_coord[2], y = template_coord[1], W=W, args=args)
    point3 = get_point(x = template_coord[0], y = template_coord[1]+template_coord[3], W=W, args=args)
    point4 = get_point(x = template_coord[0]+template_coord[2], y = template_coord[1]+template_coord[2], W=W, args=args)   
    return point1, point2, point3, point4

def get_point(x, y, W, args):
    point_1 = np.array([x,y,1]).reshape(3,1)
    if args.transformation == 2:
        point_1 = np.dot(W, point_1).reshape(-1)
        point_1[0] /= point_1[2]
        point_1[1] /= point_1[2]
        point_1 = point_1[:-1].astype(int)
    else:
        point_1 = np.dot(W, point_1).astype(int).reshape(-1)
    return point_1

def main(args):
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
            
            if args.method == 1:
                point1, point2, point3, point4 = appearence_based_tracking(template, frame)
            elif args.method == 2:
                point1, point2, point3, point4 = lucas_kanade(template, frame, bb, args)
            else:
                point1, point2, point3, point4 = pyramid_lk(template, frame, bb, args)

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
    parser = argparse.ArgumentParser(description='Live tracker')
    parser.add_argument('-tf', '--transformation', type=int, default=2, help = "transformation: 0-translation, 1-affine, 2-projective")
    parser.add_argument('-m', '--method', type=int, default=1, help="method of template tracking: 1-appearence, 2-LK, 3-PyramidLK")
    parser.add_argument('--iterations', type=int, default=10, help="max iterations for LK")
    args = parser.parse_args()
    
    main(args)