import cv2
import math
import numpy as np
from scipy import stats

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
    #print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
    y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]

    return un_warped

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
    return destination_corners, w, h


## definitely can be improved probably with modulus operator
def combination_lock(n, _min, _max):
	if(n > _max):
		return combination_lock((n - _max) + _min, _min, _max)

	if(n < _min):
		return combination_lock((_max - n), _min, _max)
	return n

def grid_reg(p1, p2, row, col):
    grid = [[], []]
    y_addr = (p2[1]-p1[1])/row
    x_addr = (p2[0]-p1[0])/col
    print(y_addr)
    # horizontal lines
    for i in range(row+1):
        y1 = p1[1]+i*y_addr
        x1 = p1[0]
        y2 = p1[1]+i*y_addr
        x2 = p2[0]
        print(x1, y1, x2, y2)
        grid[0].append([(x1, y1), (x2, y2)])
    
#usage
# array points 
#   Reference point (scalar x and y)
#    where to begin sorting (in radians)
def circle_sort(points, point = [], start=0):
	# if no point defined set as centroid of set 
	if(len(point) == 0):
		point = [np.sum(points[:][0]) / len(points), np.sum(points[0][:])/len(points)]
	
	return sorted(points, key= lambda x: math.atan2(x[0], x[1]))


for i in range(1, 15):
    path = str(i)+ ".jpg"
    img = cv2.imread(path)
    imgContour = img.copy()

    imgResized = cv2.resize(imgContour, (int(imgContour.shape[1] * .1), int(imgContour.shape[0] * .1)))
    imgGray = cv2.cvtColor(imgResized,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)

    imgCanny = cv2.Canny(imgBlur,100,200)

    contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    largestContours = []

    longestContours = []

    for cnt in contours:
        if len(cnt) > 1000:
            longestContours.append(cnt)

    c = 0

    for c in range(len(longestContours)):
        hull = cv2.convexHull(longestContours[c])
        cv2.drawContours(imgResized, [hull], 0, (0, 255, 0), 2)
        
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx_corners = cv2.approxPolyDP(hull, epsilon, True)
        approx_corners = sorted(np.concatenate(approx_corners).tolist())
        approx_corners = circle_sort(approx_corners)
        destination_points, w, h = get_destination_points(approx_corners)

        #destination_points = sorted(destination_points, key=lambda x: x[0])
        #circle_sort(destination_points, [w/2, h/2])
        un_warped = unwarp(imgGray, np.float32(approx_corners), destination_points)
        #crop image
        un_warped = un_warped[0:h, 0:w]

        cv2.imshow("unwarped", un_warped)
        cv2.imshow("Stack", imgResized)
        cv2.waitKey(0)


# Notes:
    # Check for order of corners ( if it's always the same )
    # 