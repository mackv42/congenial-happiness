# Creates a grid based on 4 different points
# Assumptions:
	# The grid is a square
	# If 4 points are not a square assume it's because of skew and pitch

#takes 4 points and number of rows and columns we want
# returns grid lines
def grid(p1, p2, p3, p4, row, col):
	return []