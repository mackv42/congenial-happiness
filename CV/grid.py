# Creates a grid based on 4 different points
# Assumptions:
	# The grid is a square
	# If 4 points are not a square assume it's because of skew and pitch

#takes 4 points and number of rows and columns we want
# returns grid lines


# returns list of lines for flat non rotated grid
def grid_reg(p1, p2, row, col):
	grid = [[], []]
	y_addr = (p2[1]-p1[1])/row
	x_addr = (p2[0]-p1[0])/col

	# horizantal lines
	for i in range(0, row):
		y1 = p1[1]+i*y_addr
		x1 = p1[0] > p2[0] ? p2[0] : p1[0]
		y2 = p1[1]+i*y_addr
		x2 = p1[0] > p2[0] ? p1[0] : p2[0]
		grid[0].push([x1, y1, x2, y2])

	# vertical lines
	for i in range(0, row):
		x1 = p1[0]+i*x_addr
		y1 = p1[1] > p2[1] ? p2[1] : p1[1]
		x2 = p1[0]+i*x_addr
		y2 = p1[1] > p2[1] ? p1[1] : p2[1]
		grid[1].push([x1, y1, x2, y2])

	return grid

def grid_skewed(p1, p2, p3, p4, row, col)