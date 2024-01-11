import cv2
import numpy as np

# Load the image
image = cv2.imread('1.png', cv2.IMREAD_COLOR)

# Define the region of the image where you want to add noise
# For example, let's add noise to the top right quarter of the image
height, width, _ = image.shape
start_y = 0
end_y = height // 2
start_x = width // 2
end_x = width

# Calculate the number of pixels to be affected
num_pixels = int(0.7 * (end_y - start_y) * (end_x - start_x))

# Generate random indices within the defined region
indices = np.random.choice(np.arange((end_y - start_y) * (end_x - start_x)), replace=False, size=num_pixels)

# Convert indices to 2D coordinates
y_indices, x_indices = np.unravel_index(indices, ((end_y - start_y), (end_x - start_x)))

# Generate Gaussian noise
mean = 0
stddev = 35
noise = np.random.normal(mean, stddev, num_pixels).astype(np.uint8)

# Add noise to the selected pixels in the image
for i in range(num_pixels):
    y, x = y_indices[i] + start_y, x_indices[i] + start_x
    for j in range(3):  # For each color channel
        image[y, x, j] = np.clip(image[y, x, j] + noise[i], 0, 255)

# Save the noisy image
cv2.imwrite('noisy_image.jpg', image)
