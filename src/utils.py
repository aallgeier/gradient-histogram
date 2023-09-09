import torch
import numpy as np
from torch import nn
import cv2
import matplotlib.pyplot as plt
import math


def plot_grad_histogram_grid(image_gray, cell_size, histogram_grid, bins):
    fig, ax = plt.subplots(figsize = (7, 7))
    ax.imshow(image_gray, cmap="gray")

    for k in range(len(histogram_grid)):
        for l in range(len(histogram_grid[0])):
            histogram = histogram_grid[k][l]

            # set histogram values within 0 and 1.
            if 0 < np.max(histogram):
                histogram = (histogram - np.min(histogram))/(np.max(histogram) - np.min(histogram))
            
            for i, bin in enumerate(bins):
                if i != len(bins)-1 and histogram[i] > 0.1:
                    rep_angle = (bins[i] + bins[i+1])/2

                    unit_vec = (np.cos(rep_angle), np.sin(rep_angle))
                    unit_vec = unit_vec/np.linalg.norm(unit_vec)

                    rep_vec = (cell_size * 0.38) * histogram[i] * unit_vec

    
                    start_x = (cell_size// 2 ) + l * cell_size
                    start_y = (cell_size// 2 ) + k * cell_size
                    
                    # -rep_vec[1] to account for the y-axis being flipped
                    ax.arrow(start_x, start_y, rep_vec[0], -rep_vec[1], width = 0.001, color="red", head_width=0.4)

    ax.set_xlim([0, image_gray.shape[1]])   
    ax.set_ylim([image_gray.shape[0], 0])      
    ax.set_xticks(np.arange(0, image_gray.shape[1], cell_size))
    ax.set_yticks(np.arange(0, image_gray.shape[0], cell_size))

    # Show grid
    ax.grid()
       
    plt.show()

        
