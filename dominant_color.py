'''
K means, find_histogram
https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
center_crop()
https://medium.com/curious-manava/center-crop-and-scaling-in-opencv-using-python-279c1bb77c74
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def find_dom_color(hist, centroids):
    dominant_color = [0, 0, 0]
    max_percent = 0
    for (percent, color) in zip(hist, centroids):
        if percent > max_percent:
            max_percent = percent
            dominant_color = color.astype("uint8").tolist()
    return dominant_color


def center_crop(img, dim):
    """Returns center cropped image

    Args:Image Scaling
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]
    #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    
    cv2.rectangle(frame,(mid_x-cw2,mid_y-cw2),(mid_x+cw2,mid_y+cw2),(0,255,0),2)
    return crop_img


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while(1):

    # Take each frame
    _, frame = cap.read()
    center_frame = center_crop(frame, (100, 100))
    frame = rescale_frame(frame, percent=40)

    center_frame = cv2.cvtColor(center_frame, cv2.COLOR_BGR2RGB)

    center_frame = center_frame.reshape((center_frame.shape[0] * center_frame.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(center_frame)

    hist = find_histogram(clt)
    #bar = plot_colors2(hist, clt.cluster_centers_)
    color = find_dom_color(hist, clt.cluster_centers_)
    print(color)

    cv2.imshow('frame',frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()