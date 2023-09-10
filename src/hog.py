import numpy as np
from torch import nn
import torch

def compute_image_gradients(img_gray):
    """Use convolution with Sobel filters to compute the first derivative of an image.

    Args:
        img_gray: A numpy array of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image
            w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image
            w.r.t. y-direction
    """
    SOBEL_X_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype(np.float32)

    SOBEL_Y_KERNEL = np.array(
    [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]).astype(np.float32)
    
    # Paddings to keep image dimensions
    px = (len(SOBEL_X_KERNEL)-1) //2
    py = (len(SOBEL_Y_KERNEL)-1) //2

    # Manipulate input dimensions to meet specifications of nn.functional.conv2d().
    # kernel: (out_channels, in_channels, k_h, k_w), assuming groups is 1.
    kernel_X = torch.tensor(SOBEL_X_KERNEL).unsqueeze(0).unsqueeze(0)
    kernel_Y = torch.tensor(SOBEL_Y_KERNEL).unsqueeze(0).unsqueeze(0)
    # image: (batch, in_channels, H, W)
    img_gray = torch.tensor(img_gray).unsqueeze(0).unsqueeze(0)

    # Convolution. The convolution output has shape (minibatch, out_channels, h, w)
    # soremove the first two dimensions before returning.
    Ix = torch.squeeze(nn.functional.conv2d(img_gray, kernel_X, padding = px)).numpy()
    Iy = torch.squeeze(nn.functional.conv2d(img_gray, kernel_Y, padding = py)).numpy()

    return Ix, Iy

def compute_magnitudes_and_orientations(image_gray: np.ndarray):
    """
    Computes gradient magnitude (Ix^2 + Iy^2) and gradient orientation
    (radian) at each pixel location

    Args:
        image_gray: black and white image

    Returns:
        orientation of gradient at each pixel location
        magnitude of the gradient at each pixel location
    """
    Ix, Iy = compute_image_gradients(image_gray)
    
    grad_magnitude = Ix **2 + Iy ** 2
    orientation = np.arctan2(Iy, Ix)

    return orientation, grad_magnitude

def create_histogram(cell_magnitudes, cell_orientations, num_bins = 8):
    """
    For a given cell of m (pixels) x n (pixels), compute the gradient 
    orientation histogram weighted by magnitude.
    """

    bins = np.linspace(-np.pi-np.pi/8, np.pi-np.pi/8, num_bins+1)
    histogram = np.histogram(np.around(cell_orientations, decimals=5), bins, weights= cell_magnitudes)[0]

    return histogram, bins

def create_image_histograms(image_gray: np.ndarray, cell_size=8):
    """
    Args: 
        image_gray: grayscale image

    Returns:
        (M//cell_size, N//cell_size) grid of numpy histograms
    
    """
    N, M = image_gray.shape

    orientation, grad_magnitude = compute_magnitudes_and_orientations(image_gray)
    
    assert M % cell_size == 0
    assert N % cell_size == 0

    num_bins = 8
    histogram_grid = np.zeros((M//cell_size, M//cell_size,  num_bins))

    bins = None

    for i in range(M//cell_size):
        for j in range(N//cell_size):
            cell_orientations = orientation[i * cell_size: (i+1)*cell_size, j * cell_size: (j+1)*cell_size]
            cell_magnitudes   = grad_magnitude[i * cell_size: (i+1)*cell_size, j * cell_size: (j+1)*cell_size]

            histogram, bins = create_histogram(cell_magnitudes, cell_orientations, num_bins=num_bins)

            histogram_grid[i, j] = histogram


    return histogram_grid, bins

def create_descriptor(histogram_grid, block_size=2, step_size=1):
    """
    2D list of histograms

    Blocks can overlap
    """

    M = len(histogram_grid)
    N = len(histogram_grid[0])

    assert (block_size <= M) and (block_size <= N)
    
    m = (len(histogram_grid)- (block_size-1)) // step_size
    n = (len(histogram_grid[0])-(block_size-1)) // step_size

    block_grid = np.zeros((m, n, len(histogram_grid[0][0])* block_size**2))

    for i in range(m):
        for j in range(n):
            # Get block
            block = histogram_grid[i*step_size:i*step_size + block_size, j*step_size:j*step_size + block_size]
            # concatenate the histograms
            block_concat = block.flatten()

            # normalize block to account for different light and contrast
            block_concat = block_concat/np.linalg.norm(block_concat)

            block_grid[i, j] = block_concat

    return block_grid
 



