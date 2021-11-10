import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import math

from shapely.geometry import LineString


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


img = cv2.imread("2021-10-11-171946.jpg")


#points 215, 192
#       492, 176
#       113, 466
#       639, 460
def skew_correction(img):
    plt.title('original')
    plt.imshow(img)
    plt.show()
    corners = [[215, 192], [492, 176], [113, 466], [639, 460]]

    dst, h, w = get_destination_points(corners)
    un_warped = unwarp(img, np.float32(corners), dst)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor='w', edgecolor='k')
        # f.subplots_adjust(hspace=.2, wspace=.05)
    cropped = un_warped[0:h, 0:w]

    ax1.imshow(un_warped)
    ax2.imshow(cropped)

    plt.show()

    return cropped

def crop(img, p1, width, height):
    return img[p1[0]:p1[0]+width][p1[1]:p1[1]+height]

def resize_to_square(img):
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    longestSide = max(width, height)

    resized = cv2.resize(img, (longestSide, longestSide))
    return resized
    
cropped = resize_to_square(skew_correction(img))

"""
#converted = convert_hls(img)
image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
lower = np.uint8([0, 0, 0])
upper = np.uint8([255, 130, 255])
white_mask = cv2.inRange(image, lower, upper)
# yellow color mask
#lower = np.uint8([10, 0,   100])
#upper = np.uint8([40, 255, 255])
#yellow_mask = cv2.inRange(image, lower, upper)
# combine the mask
#mask = cv2.bitwise_or(white_mask, yellow_mask)
result = img.copy()
cv2.imshow("mask",white_mask)
"""

grey = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)

kernel_size = 3
blur_gray = cv2.GaussianBlur(grey,(kernel_size, kernel_size),0)

#blur_gray = white_mask
#canny edge detection

low_threshold = 30
high_threshold = 80
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
cv2.imshow("edges", edges)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 50  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10  # minimum number of pixels making up a line
max_line_gap = 10  # maximum gap in pixels between connectable line segments
line_image = np.copy(cropped) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)


print(lines)
#print(lines)
"""
for line in lines:
    for x1,y1,x2,y2 in line:
    	cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
"""

#print(lines)

""" extra math code seeing what I can find from other libs
def find_slope(x1, y1, x2, y2):
    numerator = y2 - y1
    divisor = x2 - x1
    return [numerator, divisor]

def find_length( line ):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1

    return sqrt(dx*dx + dy*dy)

def intersects( line1, line2 )
    l1_x1, l1_y1, l1_x2, l1_y2 = line1
    l2_x1, l2_y1, l2_x2, l2_y2 = line2



def point_val(x1, y1, width):
    return y*width + x

def perpendicular_slopes:

    return [1, 1]

# tells if two slopes are perpendicular( within a threshold )
def is_perpendicular( slope1, slope2, thresh ):
    

    return True


# tells if two slopes are parallel with a threshold
def is_parallel( slope1, slope2, thresh ):


    return True

for x1, y1, x2, y2 in lines[0]:
"""

intersectingLines = []
#print(lines)
# runs through every line in array



def line_doesnt_exist( l, arr ):
    if len(arr) == 0:
        return True

    x1, y1, x2, y2 = l
    for line in arr:
        lx1, ly1, lx2, ly2 = line
        if x1 == lx1 and y1 == ly1:
            return False 
    return True    

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    """for line2 in lines:
        cx1, cy1, cx2, cy2 = line2[0]
        l1 = LineString([(x1, y1), (x2, y2)])
        l2 = LineString([(cx1, cy1), (cx2, cy2)])
        if l1.intersects(l2) and x1 != cx1:
            if line_doesnt_exist(line[0], intersectingLines):
                intersectingLines.append([x1, y1, x2, y2])
    """

def find_slope(x1, y1, x2, y2):
    numerator = x2 - x1
    divisor = y2 - y1
    return [numerator, divisor]

downwards = []
sideways = []

"""
for line in intersectingLines:
    x1, y1, x2, y2 = line
    slope = find_slope(x1, y1, x2, y2)
    if slope[1] == 0:
        rad = 0
    else:
        rad = math.atan2(-slope[1],slope[0])

    print(abs(rad*57.2958))
    if abs(rad*57.2958) < 45:
        sideways.append([x1, y1, x2, y2])
        downwards.append([x1, y1, x2, y2])
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
"""

"""
for line in intersectingLines:
    x1,y1,x2,y2 = line
    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
"""

lines_edges = cv2.addWeighted(cropped, 0.8, line_image, 1, 0)

cv2.imshow("Lines", lines_edges)

cv2.waitKey(0)