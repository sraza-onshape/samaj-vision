from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .ops import convolution as convolution_op
from .gaussian_base import BaseGaussianFilter 


class GaussianDerivativeFilter(BaseGaussianFilter):

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

    def __init__(self, sigma: int = 1) -> None:
        super().__init__(sigma)
        # create the 2D Gaussian filter
        self.filter_matrix = self.create_gaussian_filter(self.sigma)

    def _compute_derivatives(self, image: List[List[float]]) -> None:
        '''Setter function for the partial derivatives of the iage'''
        # 2) "separate into x and y"
        # for x: convolve filter with the horizontal Sobel
        gaussian_derivative_x_filter = convolution_op(self.filter_matrix, self.HORIZONTAL_SOBEL_FILTER)
        # then convolve the image with the convolved filter
        partial_derivative_x = convolution_op(image, gaussian_derivative_x_filter)
        # do the same for y
        gaussian_derivative_y_filter = convolution_op(self.filter_matrix, self.VERTICAL_SOBEL_FILTER)
        partial_derivative_y = convolution_op(image, gaussian_derivative_y_filter)
        return partial_derivative_x, partial_derivative_y

    def detect_edges(
            self,
            image: List[List[int]],
            threshold: float = float("-inf")
        ) -> np.array:
        '''Do edge detection: use the convolved images in the magnitude formula --> visualize it'''
        ### HELPERS
        def _compute_magnitude(partial_derivative_x, partial_derivative_y):
            # convert to numpy array (only for the purpose of making element-wise computations easier)
            partial_derivative_x = np.array(partial_derivative_x)
            partial_derivative_y = np.array(partial_derivative_y)
            magnitude = np.sqrt(
                (partial_derivative_x ** 2) + (partial_derivative_y ** 2)
            )
            return magnitude

        ### DRIVER
        partial_derivative_x, partial_derivative_y = self._compute_derivatives(image)
        edges = _compute_magnitude(partial_derivative_x, partial_derivative_y)
        # apply the threshold to zero out extraneous magnitudes
        edges = np.where(edges > threshold, edges, 0)  # default: no change
        
        return edges
    
    @classmethod
    def detect_edges_and_visualize(cls,
            image: List[List[int]],
            image_name: str,
            sigma: int = 1,
            threshold: float = float('-inf')
        ) -> np.array:
        '''Convenience wrapper + uses Matplotlib to plot the edges found.'''
        edge_detector = cls(sigma=sigma)
        detected_edges = edge_detector.detect_edges(image, threshold)
        plt.imshow(detected_edges, cmap='gray', vmin=0, vmax=255)
        plt.title(f"{image_name} Edges, sigma={sigma}, threshold={threshold}")
        plt.show()
        return detected_edges


if __name__ == "__main__":
    # TODO: add test cases
    print("code interprets without errors")
