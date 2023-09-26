from typing import List

import numpy as np

from util import ops
from util.gaussian import GaussianFilter 


class GaussianDerivativeFilter(GaussianFilter):

    HORIZONTAL_SOBEL_FILTER = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

    VERTICAL_SOBEL_FILTER = [
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ]

    def __init__(self, sigma: int = None) -> None:
        super().__init__(sigma)
        # create the 2D Gaussian filter
        self.filter_matrix = self.create_gaussian_filter(self.sigma)

    def _compute_derivatives(self, image: List[List[float]]) -> None:
        '''Setter function for the partial derivatives of the iage'''
        # 2) "separate into x and y"
        # for x: convolve filter with the horizontal Sobel
        gaussian_derivative_x_filter = ops.convolution(self.filter_matrix, self.HORIZONTAL_SOBEL_FILTER)
        # then convolve the image with the convolved filter
        partial_derivative_x = ops.convolution(image, gaussian_derivative_x_filter)
        # do the same for y
        gaussian_derivative_y_filter = ops.convolution(self.filter_matrix, self.VERTICAL_SOBEL_FILTER)
        partial_derivative_y = ops.convolution(image, gaussian_derivative_y_filter)
        return partial_derivative_x, partial_derivative_y

    def detect_edges(self, image: List[List[float]]) -> np.array:
        '''Do edge detection: use the convolved images in the magnitude formula --> visualize it'''
        ### HELPERS
        def _compute_magnitude(partial_derivative_x, partial_derivative_y):
            # convert to numpy array (only for the purpose of making element-wise computations easier)
            partial_derivative_x = np.array(partial_derivative_x)
            partial_derivative_y = np.array(partial_derivative_y)
            magnitude = np.sqrt(
                (partial_derivative_x)^2 + (partial_derivative_y)^2
            )
            return magnitude

        ### DRIVER
        partial_derivative_x, partial_derivative_y = self._compute_derivatives(image)
        edges = _compute_magnitude(partial_derivative_x, partial_derivative_y)
        # TODO: pick a threshold for the gradient magnitude so that the number of remaining edge pixels is similar to the examples in the slides.
        
        return edges


if __name__ == "__main__":
    ...
