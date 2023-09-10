import matplotlib.pyplot as plt
import cv2
import numpy as np

from src.hog import create_image_histograms
from src.hog import create_descriptor
from src.utils import plot_grad_histogram_grid

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the image (assuming the image is in the folder "data/images/")
im = cv2.resize(cv2.imread("data/nyuma.jpeg")[:, :, ::-1], (256, 256)).astype(np.float32)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


cell_size = 8
histogram_grid, bins = create_image_histograms(gray, cell_size=cell_size)
# plot_grad_histogram_grid(gray, cell_size, histogram_grid, bins)

block_size = 2
blocks = create_descriptor(histogram_grid, block_size=block_size, step_size=1)










