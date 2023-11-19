import abc, heapq
from abc import ABCMeta
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .gaussian_base import BaseGaussianFilter
from . import ops
from .ops import (
    convolution as convolution_op,
    pad as padding_op,
    convolve_matrices as convolve,
    pad as padding_op,
    HORIZONTAL_SOBEL_FILTER,
    # IDENTITY_FILTER,
    VERTICAL_SOBEL_FILTER,
)


class BaseCornerDetector(metaclass=ABCMeta):
    @classmethod
    @abc.abstractmethod
    def execute_and_visualize(cls):
        pass


class HarrisCornerDetector(BaseCornerDetector):
    CORNER_RESPONSE_CONSTANT = 0.05
    TOP_MANY_FEATURES_TO_DETECT = 1000  # as outlined in the hw 3 description

    def detect_features(
        self,
        image: np.ndarray,
        use_non_max_suppression: bool = False,
    ) -> np.ndarray:
        """TODO[Zain]"""

        ### HELPERS
        def _compute_derivatives_in_gaussian_window(
            image: np.ndarray,
            gaussian_window: List[List[float]],
        ) -> np.ndarray:
            # compute second order gradient
            (
                second_order_derivator_x,
                second_order_derivator_y,
                second_order_derivator_xy,
            ) = (
                convolution_op(
                    HORIZONTAL_SOBEL_FILTER,
                    HORIZONTAL_SOBEL_FILTER,
                    padding_type="zero",
                ),
                convolution_op(
                    VERTICAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER, padding_type="zero"
                ),
                convolution_op(
                    HORIZONTAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER, padding_type="zero"
                ),
            )
            image_list = image.tolist()
            (
                hessian_xx,
                hessian_yy,
                hessian_xy
            ) = (
                np.array(
                    convolution_op(
                        image_list, second_order_derivator_x, padding_type="zero"
                    )
                ),
                np.array(
                    convolution_op(
                        image_list, second_order_derivator_y, padding_type="zero"
                    )
                ),
                np.array(
                    convolution_op(
                        image_list, second_order_derivator_xy, padding_type="zero"
                    )
                ),
            )

            # compute the second moment matrix in a Gaussian window around each pixel
            (
                convolved_hessian_xx,
                convolved_hessian_yy,
                convolved_hessian_xy
            ) = (
                np.array(
                    convolution_op(hessian_xx, gaussian_window, padding_type="zero")
                ),
                np.array(
                    convolution_op(hessian_yy, gaussian_window, padding_type="zero")
                ),
                np.array(
                    convolution_op(hessian_xy, gaussian_window, padding_type="zero")
                ),
            )

            return (convolved_hessian_xx, convolved_hessian_yy, convolved_hessian_xy)

        def _compute_corner_response(
            kernel: List[List[float]],
            # these 3 have the same dims as the input image
            convolved_hessian_xx: np.ndarray,
            convolved_hessian_yy: np.ndarray,
            convolved_hessian_xy: np.ndarray,
            stride: int = 1,
        ) -> np.array:
            """TODO[Zain] add docstring"""
            # ensure the corner response matrix has the same dims as the input image
            convolved_hessian_xx, _, _ = padding_op(
                convolved_hessian_xx, kernel, stride, "zero"
            )
            convolved_hessian_yy, _, _ = padding_op(
                convolved_hessian_yy, kernel, stride, "zero"
            )
            convolved_hessian_xy, _, _ = padding_op(
                convolved_hessian_xy, kernel, stride, "zero"
            )

            # computation begins below - TODO[Zain]: try to make this more DRY
            corner_response = list()
            kernel_h, kernel_w = len(kernel), len(kernel[0])
            # iterate over the rows and columns
            starting_row_ndx = 0
            while starting_row_ndx <= len(convolved_hessian_xy) - kernel_h:
                # convolve the next row of this response
                response_row = list()
                starting_col_ndx = 0
                while starting_col_ndx <= len(convolved_hessian_xy[0]) - kernel_w:
                    # compute the response for this window
                    col_index = starting_col_ndx
                    block_of_derivative_xx = convolved_hessian_xx[
                        starting_row_ndx : (kernel_h + starting_row_ndx),
                        col_index : (kernel_w + col_index),
                    ]
                    block_of_derivative_xy = convolved_hessian_xy[
                        starting_row_ndx : (kernel_h + starting_row_ndx),
                        col_index : (kernel_w + col_index),
                    ]
                    block_of_derivative_yy = convolved_hessian_yy[
                        starting_row_ndx : (kernel_h + starting_row_ndx),
                        col_index : (kernel_w + col_index),
                    ]
                    second_moment_matrix_elements = [
                        np.sum(block_of_derivative_xx),
                        np.sum(block_of_derivative_xy),
                        np.sum(block_of_derivative_yy),
                    ]
                    # add it to the output
                    determinant = (
                        second_moment_matrix_elements[0]
                        * second_moment_matrix_elements[2]
                    ) - (second_moment_matrix_elements[1] ** 2)
                    trace = (
                        second_moment_matrix_elements[0]
                        + second_moment_matrix_elements[2]
                    )
                    corner_response_element = determinant - (
                        self.CORNER_RESPONSE_CONSTANT * (trace**2)
                    )
                    response_row.append(corner_response_element)
                    # move on to the next starting column, using the stride
                    starting_col_ndx += stride
                # now, add the new row to the list
                corner_response.append(response_row)
                # move to the next starting row for the corner response calculation
                starting_row_ndx += stride
            return np.array(corner_response)

        ### DRIVER
        gaussian_window = BaseGaussianFilter().create_gaussian_filter()
        (
            convolved_hessian_xx,
            convolved_hessian_yy,
            convolved_hessian_xy,
        ) = _compute_derivatives_in_gaussian_window(
            image, gaussian_window=gaussian_window
        )
        corner_response = _compute_corner_response(
            gaussian_window,
            convolved_hessian_xx,
            convolved_hessian_yy,
            convolved_hessian_xy,
        )

        if use_non_max_suppression is True:
            corner_response = ops.non_max_suppression_2D(corner_response)

        return corner_response

    @classmethod
    def execute_and_visualize(
        cls: "HarrisCornerDetector",
        image: np.ndarray,
        image_name: str,
        top_many_features: int = TOP_MANY_FEATURES_TO_DETECT,
        use_non_max_suppression: bool = False,
    ):
        # detect_features
        detector = cls()
        corner_response = detector.detect_features(image, use_non_max_suppression)
        # pick top 1000
        height, width = corner_response.shape
        coordinate_value_pairs = np.zeros((width * height, 3))
        for val_index in range(coordinate_value_pairs.shape[0]):
            row_index = val_index // width
            col_index = val_index - (width * row_index)
            coordinate_value_pairs[val_index] = [
                row_index,
                col_index,
                corner_response[row_index, col_index],
            ]
        top_k_points = np.array(heapq.nlargest(
            top_many_features,
            coordinate_value_pairs,
            key=lambda coords_followed_by_val: coords_followed_by_val[2],
        ))
        # plotting
        plt.imshow(image, cmap="gray")
        plt.scatter(y=top_k_points[:, 0], x=top_k_points[:, 1], color="red")
        plt.title(f'Corner Points Detected for Image: "{image_name}"')
        plt.show()

        return super().execute_and_visualize()


if __name__ == "__main__":
    HarrisCornerDetector.execute_and_visualize()
