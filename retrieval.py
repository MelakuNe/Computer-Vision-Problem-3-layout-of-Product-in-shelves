import cv2 as cv
import numpy as np


def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    one = cv.cvtColor(query, cv.COLOR_BGR2GRAY)
    two = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    j.append(1)
    list_of_bboxes = main(one, two)
    return list_of_bboxes

def bounding_box(dst, main_image):
    main_height, main_width = main_image.shape
    four = dst.reshape(4,2)
    xmin, ymin = np.min(four, 0)[0], np.min(four, 0)[1]
    xmax, ymax = np.max(four, 0)[0], np.max(four, 0)[1]
    xmin = xmin / main_width
    ymin = ymin / main_height
    width = (xmax - xmin) / main_width
    height = (ymax - ymin) / main_height
    final_box = (xmin, ymin, width, height)
    return final_box
    
# ------------------------------------------------
#                   Method 1 
# ------------------------------------------------
def method1(Query, Gallery, which):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(Query, None)
    kp2, des2 = sift.detectAndCompute(Gallery, None)
    bf = cv.BFMatcher()
    match1 = bf.knnMatch(des1, des2, k=1)
    sort_matches = sorted(match1, key = lambda x:x[0].distance)
    best_matches = sort_matches[:105]
    get_items = []
    for i in best_matches:
        get_items.append(i[0])
 
    sr = np.array([kp1[m.queryIdx].pt for m in get_items]).reshape(-1, 1, 2)
    de = np.array([kp2[n.trainIdx].pt for n in get_items]).reshape(-1, 1, 2)
    return sr, de

def submethod1(sr, de, Qr, Gal):
    M, mask = cv.findHomography(sr, de, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = Qr.shape[:2]
    pts = np.float32([ [0,0],[0, h-1],[w-1, h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts, M)
    lined_img = cv.polylines(Gal, [np.int32(dst)], True, (0,0,255),5, cv.LINE_AA)
    return dst, lined_img

def decision(points):
    status = True
    k = points.reshape(4, 2)
    st = [False for i,j in k if i < 0 or j < 0]
    diff = k[0][0] - k[1][0]
    if np.abs(diff) > 40:
        status = False
    if len(st) is not 0:
        if not st[0]:
            status = False
            print("Negative location detected\n", k)
    xmin, ymin = np.min(k, 0)[0], np.min(k, 0)[1]
    xmax, ymax = np.max(k, 0)[0], np.max(k, 0)[1]
    w1 = np.abs(k[0][0] - k[3][0])
    w2 = np.abs(k[1][0] - k[2][0])
    h1 = np.abs(k[0][1] - k[1][1])
    h2 = np.abs(k[2][1] - k[3][1])
    if (ymax-ymin) > 30 and (xmax - xmin) > 20:
        pass
    else:
        status = False
        print("problem with the points\n", k)
        print("ymax and ymin: ", ymax, ymin, '\nxmax and xmin: ', xmax, xmin)
    return status

def manipulate(Q, G, which):
    sc, ds = method1(Q, G, which)
    return sc, ds

def main(meje, ketay, which = 'some'):
    count = 0 
    sebsib = []
    while 1:
        size1, size2 = manipulate(meje, ketay, which)
        if size1.shape[0] > 3 or size2.shape[0] > 3:
            metu, digame = submethod1(size1, size2, meje, ketay)
            poly = cv.fillPoly(digame, [np.int32(metu)], (0, 0, 0))
            out_box = bounding_box(metu, poly)
            cross_check = decision(metu)
            if count == 0:
                boox = out_box
                area_one = boox[2] * boox[3]
            count += 1
            if cross_check:
                area = out_box[2] * out_box[3]
                if area < 1.5*area_one or area_one < 1.5*area:
                    sebsib.append(out_box)
                    ketay = poly
                else:
                    break
            else:
                break
        else:
            break
    return sebsib
