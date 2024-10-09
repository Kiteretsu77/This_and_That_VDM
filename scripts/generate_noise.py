import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set the dimensions of the image
height = 256
width = 256

# Generate random pixel values
noise = np.random.rand(height, width, 3) * 255  # Scale to 255 for grayscale image


for idx in range (4):
    cv2.imwrite("noise"+str(idx)+".png", noise)