import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def correct_skew(image, delta=5, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)
    return best_angle, rotated




image = cv2.imread("2.jpg", cv2.IMREAD_COLOR)
#angle, rotated = correct_skew(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (1, 1))
  
# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 10, param1 = 30,
               param2 = 30, minRadius = 0, maxRadius = 24)
  

# Draw circles that are detected.
if detected_circles is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
    print("{\n\"circles\": [")
    for pt in detected_circles[0]:

        a, b, r = pt[0], pt[1], pt[2]

        print("{\n\"A\": "+ np.array2string(a) + ",\n\"B\": " + np.array2string(b) + ",\n\"R\": " + np.array2string(r) + "},\n")

        # Draw the circumference of the circle.
        cv2.circle(image, (a, b), r, (0, 255, 0), 2)
  
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(image, (a, b), 1, (0, 0, 255), 3)

    print("]\n}")
#print(angle)
cv2.imshow('rotated', image)
#cv2.imwrite('rotated.png', rotated)
#cv2.waitKey()
        
#cv2.imshow("Detected Circle", rotated) 
cv2.waitKey(0)