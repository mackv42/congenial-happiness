# Creates a grid based on 4 different points
# Assumptions:
	# The grid is a square
	# If 4 points are not a square assume it's because of skew and pitch

#takes 4 points and number of rows and columns we want
# returns grid lines
import cv2
import numpy as np

def draw_lines(grid, canvas):
	for i in range(0, len(grid[0])):
		p = grid[0][i]
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
	# horizantal lines
	for i in range(0, row):
		y1 = p1[1]+i*y_addr
		x1 = p1[0]
		y2 = p1[1]+i*y_addr
		x2 = p2[0]
		print(x1, y1, x2, y2)
		grid[0].append([(x1, y1), (x2, y2)])

	# vertical lines
	for i in range(0, row):
		x1 = p1[0]+i*x_addr
		y1 = p1[1]
		x2 = p1[0]+i*x_addr
		y2 = p2[1]
		grid[1].append([(x1, y1), (x2, y2)])

	return grid


# define blank white 640x640 rgb canvas
canvas = np.ones((640,640,3), np.uint8)*255

grid = grid_reg([0, 0], [100, 100], 10, 10)
print(len(grid[0]))
draw_lines(grid, canvas)

cv2.imshow("hi", canvas)
cv2.waitKey(0)
# returns list of lines for flat rotated grid
'''
def grid_rotated(p1, p2, row, col):
	dx = p2[0] - p1[0]
	dy = p2[1] - p1[1]
	
	# numerator and denominator of slope 
	slope = dx / dy

	#remember y = mx
	# x = y/m

	# horizantal lines
	for i in range(0, row):
		y1 = p1[1]
		# y = mx+b
		y2 = slope*p2[0]+p1[1]
		x1 = i/m + p1[0]
		'''