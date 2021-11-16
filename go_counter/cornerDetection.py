#!/usr/bin/env python3
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
def shi_tomashi(image):
    """
    Use Shi-Tomashi algorithm to detect corners

    Args:
        image: np.array

    Returns:
        corners: list

    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 100)
    corners = np.int0(corners)
    corners = sorted(np.concatenate(corners).tolist())
    print('\nThe corner points are...\n')

    im = image.copy()
    for index, c in enumerate(corners):
        x, y = c
        cv2.circle(im, (x, y), 3, 255, -1)
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(im, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    plt.imshow(im)
    plt.title('Corner Detection: Shi-Tomashi')
    plt.show()
    return corners


def get_destination_points(corners):
    """
    -Get destination points from corners of warped images
    -Approximating height and width of the rectangle: we take maximum of the 2 widths and 2 heights

    Args:
        corners: list

    Returns:
        destination_corners: list
        height: int
        width: int

    """

    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

    print('\nThe destination points are: \n')
    for index, c in enumerate(destination_corners):
        character = chr(65 + index) + "'"
        print(character, ':', c)

    print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w


def unwarp(img, src, dst):
    """

    Args:
        img: np.array
        src: list
        dst: list

    Returns:
        un_warped: np.array

    """
    h, w = img.shape[:2]
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    # plot

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(img)
    ax1.set_title('Original Image')

    x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
    y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]

    ax2.imshow(img)
    ax2.plot(x, y, color='yellow', linewidth=3)
    ax2.set_ylim([h, 0])
    ax2.set_xlim([0, w])
    ax2.set_title('Target Area')

    plt.show()
    return un_warped


def example_one():
    """
    Skew correction using homography and corner detection using Shi-Tomashi corner detector

    Returns: None

    """
    image = cv2.imread('images/example_1.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title('Original Image')
    plt.show()

    corners = shi_tomashi(image)
    destination, h, w = get_destination_points(corners)
    un_warped = unwarp(image, np.float32(corners), destination)
    cropped = un_warped[0:h, 0:w]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor='w', edgecolor='k')
    # f.subplots_adjust(hspace=.2, wspace=.05)

    ax1.imshow(un_warped)
    ax2.imshow(cropped)

    plt.show()


def apply_filter(image):
    """
    Define a 5X5 kernel and apply the filter to gray scale image
    Args:
        image: np.array

    Returns:
        filtered: np.array

    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    #plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Image')
    plt.show()
    return filtered

def apply_threshold(filtered):
    """
    Apply OTSU threshold

    Args:
        filtered: np.array

    Returns:
        thresh: np.array

    """

    



    thresh = cv2.adaptiveThreshold(filtered,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    #plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    plt.title('After applying OTSU threshold')
    plt.show()
    return thresh

def detect_contour(img, image_shape):
    """

    Args:
        img: np.array()
        image_shape: tuple

    Returns:
        canvas: np.array()
        cnt: list

    """
    canvas = np.zeros(image_shape, np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    

    cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)
    plt.title('Largest Contour')
   # plt.imshow(canvas)
    plt.show()

    return canvas, cnt

def detect_corners_from_contour(canvas, cnt):
    """
    Detecting corner points form contours using cv2.approxPolyDP()
    Args:
        canvas: np.array()
        cnt: list

    Returns:
        approx_corners: list

    """
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    print('\nThe corner points are ...\n')
    for index, c in enumerate(approx_corners):
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Rearranging the order of the corner points
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]

   # plt.imshow(canvas)
    plt.title('Corner Points: Douglas-Peucker')
    plt.show()
    return approx_corners


def example_two(image_path):
    """
    Skew correction using homography and corner detection using contour points
    Returns: None

    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   # plt.imshow(image)
    plt.title('Original Image')
    plt.show()


    filtered_image = apply_filter(image)
    ###Experimental###
    dst = cv2.cornerHarris(filtered_image,2,3,0.04)

#result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    image[dst>0.01*dst.max()]=[0,0,255]

   # cv2.imshow('dst', image)

    thresh = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    cv2.imshow("Mean Adaptive Thresholding", thresh)
    ###########################################################################
    threshold_image = apply_threshold(filtered_image)

    cnv, largest_contour = detect_contour(threshold_image, image.shape)
    corners = detect_corners_from_contour(cnv, largest_contour)

    destination_points, h, w = get_destination_points(corners)

    un_warped = unwarp(image, np.float32(corners), destination_points)

    cropped = un_warped[0:h, 0:w]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    ax1.imshow(un_warped)
    ax2.imshow(cropped)

    plt.show()

def corner_detection(image):
    filtered_image = apply_filter(image)
    thresh = apply_threshold(filtered_image)
    ###Experimental###
    dst = cv2.cornerHarris(thresh,5,3,0.15, cv2.BORDER_CONSTANT)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    image[dst>0.3*dst.max()]=[0,0,255]
    #cv2.imshow("dst", image)


def cd_2(img, n):
    #img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray = np.float32(gray)
    #cv2.imshow("gray", gray)
    # input of selecting (img, n best corners, ?, distance )
    corners = cv2.goodFeaturesToTrack(gray, n, 0.01, 10)
    corners = np.int0(corners)
       
    return corners

def drawCorners(img, corners):
     for corner in corners:
        x,y = corner.ravel()
        cv2.circle(img,(x,y),3,255,-1)

def getMinMax(points):
    MinX = sys.maxsize
    MinY = sys.maxsize
    MaxX = 0
    MaxY = 0
    for point in points:
        x,y = point.ravel()
        MinX = MinX if MinX < x else x
        MinY = MinY if MinY < y else y
        MaxX = MaxX if MaxX > x else x
        MaxY = MaxY if MaxY > y else y

    return [MinX, MinY, MaxX, MaxY]

def sortX(points):
    s = np.sort(points, key= lambda x: x[0])
    return s

def sortY(points):
    s = np.sort(points, key = lambda y: y[1])
    return s

def slope(p1, p2):
    return p2[0]-p1[0]/p2[1]-p1[0]

img = cv2.imread(sys.argv[1])

print(sys.argv[1])
if img is not None:
    width = img.shape[1]
    height = img.shape[0]

    image = cv2.resize(img, (int(width * .25), int(height*.25)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #corner_detection('./Images/2021-10-09-134228.jpg')
    corners1 = cd_2(image, 10)
    print(corners1)
    drawCorners(image, corners1)
    cv2.imwrite("./results/points/"+str(random.randrange(2000))+".jpg", image)
    _minX, _minY, _maxX, _maxY = getMinMax(corners1)
    print(_minX, _minY, _maxX, _maxY)
    cropped = image[_minY:_maxY, _minX:_maxX]
    #cropped.width = 
    corners2 = cd_2(cropped, 20)
    drawCorners(cropped, corners2)
    cv2.imwrite("./results/crop_points/"+str(random.randrange(2000))+".jpg", cropped)