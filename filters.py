import numpy as np

# Define common convolutional filters
COMMON_FILTERS = {
    "Edge Detection (Horizontal)": np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ]),
    "Edge Detection (Vertical)": np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]),
    "Sobel (Horizontal)": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]),
    "Sobel (Vertical)": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    "Scharr (Horizontal)": np.array([
        [-3, -10, -3],
        [ 0,   0,  0],
        [ 3,  10,  3]
    ]),
    "Scharr (Vertical)": np.array([
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]
    ]),
    "Prewitt (Horizontal)": np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ]),
    "Prewitt (Vertical)": np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]),
    "Laplacian": np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ]),
    "Sharpening": np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ]),
    "Box Blur": np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ]),
    "Gaussian Blur": np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ]),
    "Emboss": np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ]),
    "Identity": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]),
    "Ridge Detection": np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
}