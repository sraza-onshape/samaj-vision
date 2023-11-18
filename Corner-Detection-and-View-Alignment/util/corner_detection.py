import abc
from abc import ABCMeta
from typing import List

import numpy as np

from .gaussian_base import BaseGaussianFilter
from .ops import (
    convolution as convolution_op,
    pad as padding_op,
    convolve_matrices as convolve,
    # pad as padding_op,
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

    def detect_features(
        self,
        image: np.ndarray,
        use_non_max_suppression: bool = False,
    ) -> np.ndarray:
        """TODO[Zain]"""
        ### HELPER(S)
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
            (hessian_xx, hessian_yy, hessian_xy) = (
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
            (convolved_hessian_xx, convolved_hessian_yy, convolved_hessian_xy) = (
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
            # ensure the corner response matrix has the same dims as the input image
            convolved_hessian_xx = padding_op(
                convolved_hessian_xx, kernel, stride, "zero"
            )
            convolved_hessian_yy = padding_op(
                convolved_hessian_yy, kernel, stride, "zero"
            )
            convolved_hessian_xy = padding_op(
                convolved_hessian_xy, kernel, stride, "zero"
            )

            # computation begins below - TODO[Zain]: try to make this more DRY
            corner_response = list()
            kernel_h, kernel_w = len(kernel), len(kernel[0])
            # iterate over the rows and columns
            starting_row_ndx = 0
            while starting_row_ndx <= len(convolved_hessian_xy) - kernel_h:
                # convolve the next row of this response
                conv_channel_row = list()
                starting_col_ndx = 0
                while starting_col_ndx <= len(convolved_hessian_xy[0]) - kernel_w:
                    # compute the convolutions
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
                    conv_channel_row.append(corner_response_element)
                    # move on to the next starting column, using the stride
                    starting_col_ndx += stride

                # now, add the convolved row to the list
                corner_response.append(conv_channel_row)
                # move to the next starting row for the corner response calculation
                starting_row_ndx += stride
            return np.array(corner_response)

        ### DRIVER
        # compute second moment matrix
        gaussian_window = BaseGaussianFilter().create_gaussian_filter()
        (convolved_hessian_xx, convolved_hessian_yy, convolved_hessian_xy) = _compute_derivatives_in_gaussian_window(
            image, gaussian_window=gaussian_window
        )
        corner_response = _compute_corner_response(
            gaussian_window,
            convolved_hessian_xx,
            convolved_hessian_yy,
            convolved_hessian_xy
        )

        # TODO[Zain]: add non-max suppression
        return corner_response

    @classmethod
    def execute_and_visualize(cls, a):
        # TODO[Zain] - beef this up later
        # detect_features
        # pick top 1000
        # plotting
        return super().execute_and_visualize()


if __name__ == "__main__":
    HarrisCornerDetector.execute_and_visualize()
