import numpy as np
from imageio import imread


class DataPrep:

    def flip_horizontal(image):
        return np.flip(image, 1)
        
    def flip_vertical(image):
        return np.flip(image, 0)
