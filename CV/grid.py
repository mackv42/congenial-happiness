# Creates a grid based on 4 different points
# Assumptions:
	# The grid is a square
	# If 4 points are not a square assume it's because of skew and pitch

#takes 4 points and number of rows and columns we want
# returns grid lines
import cv2
import numpy as np

def draw_lines(grid, canvas):
	# draw horizantal
	for i in range(len(grid[0])):
		p = grid[0][i]
		x1 = int(p[0][0])
		y1 = int(p[0][1])
		x2 = int(p[1][0])
		y2 = int(p[1][1])
		cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

	for i in range(len(grid[1])):
		p = grid[1][i]
		x1 = int(p[0][0])
		y1 = int(p[0][1])
		x2 = int(p[1][0])
		y2 = int(p[1][1])
		cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
# returns list of lines for flat non rotated grid
# p1 is upper left p2 is bottom right corner
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

	# vertical lines
	for i in range(col+1):
		x1 = p1[0]+i*x_addr
		y1 = p1[1]
		x2 = p1[0]+i*x_addr
		y2 = p2[1]
		grid[1].append([(x1, y1), (x2, y2)])

	return grid


def grid_rotated(p1, p2, p3, p4, row, col):
	grid = [[], []]
	dx = p4[0] - p1[0]
	dy = p3[1] - p2[1]
	
	y_addr = dy/row
	x_addr = dx/col

	# horizontal lines
	for i in range(row+1):
		y1 = p1[1]+i*y_addr
		x1 = p1[0]+i*x_addr
		y2 = p2[1]+i*y_addr
		x2 = p2[0]+i*x_addr
		
		grid[0].append([(x1, y1), (x2, y2)])

	
	dx = p3[0] - p4[0]
	dy = p2[1] - p1[1]
	y_addr = dy/col
	x_addr = dx/row

	for i in range(col+1):
		y1 = p1[1]+i*y_addr
		y2 = p4[1]+i*y_addr
		x1 = p1[0]+i*x_addr
		x2 = p4[0]+i*x_addr
		
		grid[1].append([(x1, y1), (x2, y2)])
	
	return grid


# Copied from S Overflow
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# Generic function to detect line intersections in grid
def grid_intersections(grid):
	# horizantal lines intersecting
	#  Vertical
	points = []
	for lineh in grid[0]:
		for linev in grid[1]:
			point = line_intersection(lineh, linev)
			points.append(point)

	return points

def draw_points(points, canvas):
	for point in points:
		point = (int(point[0]),int(point[1]))
		# draw 3x3 red rectangle for each point 
		cv2.rectangle(canvas, (point[0]-1, point[1]-1), (point[0]+1, point[1]+1), (0,0,255), -1)

# make blank white 640x640 rgb canvas
canvas = np.ones((640,640,3), np.uint8)*255
grid = grid_rotated([140, 140], [500, 150], [420, 510], [60, 500], 10, 10)
i_points = grid_intersections(grid)

draw_lines(grid, canvas)
draw_points(i_points, canvas)

cv2.imshow("hi", canvas)
cv2.waitKey(0)