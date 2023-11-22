import abc, heapq
from abc import ABCMeta
import functools
from typing import List, Literal, Tuple

from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np

from .gaussian_base import BaseGaussianFilter
from . import ops
from .ops import (
    convolution as convolution_op,
    pad as padding_op,
    pad as padding_op,
    Filter2D,
    SimilarityMeasure,
)


class BaseCornerDetector(metaclass=ABCMeta):
    @classmethod
    @abc.abstractmethod
    def execute_and_visualize(cls):
        pass


class HarrisCornerDetector(BaseCornerDetector):
    CORNER_RESPONSE_CONSTANT = 0.05
    TOP_MANY_FEATURES_TO_DETECT = 1000  # as outlined in the hw 3 description
    TOP_MANY_SIMILARITIES_TO_SELECT = 20  # as outlined in the hw 3 description

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
                    Filter2D.HORIZONTAL_SOBEL_FILTER.value,
                    Filter2D.HORIZONTAL_SOBEL_FILTER.value,
                    padding_type="zero",
                ),
                convolution_op(
                    Filter2D.VERTICAL_SOBEL_FILTER.value,
                    Filter2D.VERTICAL_SOBEL_FILTER.value,
                    padding_type="zero",
                ),
                convolution_op(
                    Filter2D.HORIZONTAL_SOBEL_FILTER.value,
                    Filter2D.VERTICAL_SOBEL_FILTER.value,
                    padding_type="zero",
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

    def pick_top_features(
        self,
        corner_response: np.ndarray,
        top_many_features: int = TOP_MANY_FEATURES_TO_DETECT,
    ):
        """TODO[Zain]: add docstring"""
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
        return np.array(
            heapq.nlargest(
                top_many_features,
                coordinate_value_pairs,
                key=lambda coords_followed_by_val: coords_followed_by_val[2],
            )
        )

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
        # pick top features
        top_k_points = detector.pick_top_features(corner_response, top_many_features)
        # plotting
        plt.imshow(image, cmap="gray")
        plt.scatter(y=top_k_points[:, 0], x=top_k_points[:, 1], color="red")
        plt.title(f'Corner Points Detected for Image: "{image_name}"')
        plt.show()

        return super().execute_and_visualize()

    @classmethod
    def visualize_patch_similarity(
        cls: "HarrisCornerDetector",
        left_img: np.ndarray,
        right_img: np.ndarray,
        plot_title: str,
        top_many_features: int = TOP_MANY_FEATURES_TO_DETECT,
        top_many_similarities: int = TOP_MANY_SIMILARITIES_TO_SELECT,
        use_non_max_suppression: bool = False,
        similarity_metric: Literal[
            SimilarityMeasure.SSD, SimilarityMeasure.NCC
        ] = SimilarityMeasure.SSD,
    ):
        ### HELPER(S)
        def _compute_similiarity_of_array_items(
            index1: int,
            index2: int,
            array1: np.ndarray,
            array2: np.ndarray,
            index_of_compare_val: int,
            metric: Literal[SimilarityMeasure.SSD, SimilarityMeasure.NCC],
        ) -> Tuple[int, int, float]:
            """Return type represents: (index of 1, index of 2, similarity measure)."""
            icv = index_of_compare_val
            elem1, elem2 = (
                array1[index1][icv].reshape(1, 1),
                array2[index2][icv].reshape(1, 1),
            )
            return (
                index1,
                index2,
                ops.compute_similarity(metric, elem1, elem2),
            )

        ### DRIVER
        # detect_features
        detector = cls()
        corner_response1 = detector.detect_features(left_img, use_non_max_suppression)
        corner_response2 = detector.detect_features(right_img, use_non_max_suppression)
        # pick top features
        top_k_points1 = detector.pick_top_features(corner_response1, top_many_features)
        top_k_points2 = detector.pick_top_features(corner_response2, top_many_features)
        # compute the similarities, and then grab the highest ones
        custom_similarity_func = functools.partial(
            _compute_similiarity_of_array_items,
            array1=top_k_points1,
            array2=top_k_points2,
            metric=similarity_metric,
            index_of_compare_val=2,
        )
        # TODO[Zain - vectorize later]
        similarities = list()
        for index1 in range(top_k_points1.shape[0]):
            for index2 in range(top_k_points2.shape[0]):
                similarities.append(custom_similarity_func(index1, index2))
        top_similarities = np.array(
            heapq.nlargest(
                top_many_similarities,
                similarities,
                key=lambda indicies_and_similarity: indicies_and_similarity[2],
            )
        )

        assert top_similarities.shape == (
            top_many_similarities,
            3,
        ), f"Expected to pick up ({top_many_similarities}, 3) similarities, actually have: {top_similarities.shape}"

        # Create a new figure
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first image
        ax[0].imshow(left_img, cmap="gray")
        ax[0].set_title("Left Image")

        # Plot the second image
        ax[1].imshow(right_img, cmap="gray")
        ax[1].set_title("Right Image")

        # Loop through the size list and draw lines between connected points
        for left_img_index, right_img_index, _ in top_similarities:
            left_img_index, right_img_index = int(left_img_index), int(right_img_index)
            left_img_y, left_img_x, _ = top_k_points1[left_img_index]
            right_img_y, right_img_x, _ = top_k_points2[right_img_index]

            # Plot red dots on the images
            ax[0].plot(left_img_x, left_img_y, "ro")
            ax[1].plot(right_img_x, right_img_y, "ro")

            # Draw a line connecting the points
            connector = ConnectionPatch(
                xyA=(left_img_x, left_img_y),
                coordsA=ax[0].transData,
                xyB=(right_img_x, right_img_y),
                coordsB=ax[1].transData,
                color="green",
            )

            fig.add_artist(connector)

        # Set axis limits to include the entire images
        ax[0].set_xlim([0, left_img.shape[1]])
        ax[0].set_ylim([0, left_img.shape[0]])
        ax[1].set_xlim([0, left_img.shape[1]])
        ax[1].set_ylim([0, right_img.shape[0]])
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        plt.title(plot_title)
        plt.show()

        return super().execute_and_visualize()


if __name__ == "__main__":
    HarrisCornerDetector.execute_and_visualize()
