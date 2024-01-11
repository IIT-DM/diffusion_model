import cv2
import numpy as np

# Load the image
image = cv2.imread('1.png', cv2.IMREAD_COLOR)

# Define the region of the image where you want to black out
# For example, let's black out the top right quarter of the image
height, width, _ = image.shape
start_y = 0
end_y = height // 2
start_x = width // 2
end_x = width

# Black out the selected region of the image
image[start_y:end_y, start_x:end_x] = 0

# Save the modified image
cv2.imwrite('blacked_out_image.jpg', image)
