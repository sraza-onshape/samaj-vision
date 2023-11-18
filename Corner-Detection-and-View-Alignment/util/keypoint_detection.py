from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .gaussian_base import BaseGaussianFilter
from .ops import (
    convolution as convolution_op,
    pad as padding_op,
    HORIZONTAL_SOBEL_FILTER,
    IDENTITY_FILTER,
    VERTICAL_SOBEL_FILTER,
)


class AbstractKeypointDetector:
    """This class is intentionally left blank."""

    # TODO: utilize the `abc.ABC` module here, to learn cool new Python skills :)
    pass


class HessianDetector(AbstractKeypointDetector):
    """Utilizes the determinant of the Hessian matrix to find keypoints."""

    DEFAULT_PERCENTILE_FOR_DETERMINANT = 75  # totally arbitrary

    def __init__(self, threshold: float = None):
        self.threshold = threshold  # should be an exact value, expected to be in the range of the image determinant

    def _set_threshold(self, values: np.array, percentile: float) -> None:
        """Sets the determinant based on a percentile of an n-dimensional array (representing the range of some function)."""
        self.threshold = np.percentile(values, percentile)

    def _get_threshold(self) -> float:
        return self.threshold

    def find_keypoints(
        self, image: np.array, percentile: float = DEFAULT_PERCENTILE_FOR_DETERMINANT
    ) -> np.array:
        ### HELPERS
        def _suppress(keypoints: np.array) -> np.array:
            """After the determinant has been thresholded, use non-max suppression to recover more distinguishable keypoints."""
            # prevent potential loss of keypoints via padding
            padded_matrix, num_added_rows, num_added_cols = padding_op(
                keypoints.tolist(),
                img_filter=IDENTITY_FILTER,
                stride=1,
                padding_type="zero",
            )
            # traverse the matrix, to begin non-max suppression
            for center_val_row in range(
                num_added_rows // 2, padded_matrix.shape[0] - (num_added_rows // 2)
            ):
                for center_val_col in range(
                    num_added_cols // 2, padded_matrix.shape[1] - (num_added_cols // 2)
                ):
                    # determine if the given value should be suppressed, or its neighbors
                    center_val = padded_matrix[center_val_row][center_val_col]
                    neighbors = padded_matrix[
                        center_val_row - 1 : center_val_row + 2,
                        center_val_col - 1 : center_val_col + 2,
                    ]
                    neighbors[1][
                        1
                    ] = 0  # hack to prevent the center value from "self-suppressing" (I have no idea, I made that term up)
                    # zero out the appropiate value(s)
                    if center_val > neighbors.max():  # suppression of neighbors
                        padded_matrix[
                            center_val_row - 1 : center_val_row + 2,
                            center_val_col - 1 : center_val_col + 2,
                        ] = 0
                        padded_matrix[center_val_row][center_val_col] = center_val
                    else:  # suppression of the center
                        padded_matrix[center_val_row][center_val_col] = 0

            # return the modified matrix - TODO[optimize later]
            return padded_matrix[
                num_added_rows // 2 : keypoints.shape[0] - (num_added_rows // 2),
                num_added_cols // 2 : keypoints.shape[1] - (num_added_cols // 2),
            ]

        ### DRIVER
        # compute the second order partial derivatives
        (
            second_order_derivator_x,
            second_order_derivator_y,
            second_order_derivator_xy,
        ) = (
            convolution_op(
                HORIZONTAL_SOBEL_FILTER, HORIZONTAL_SOBEL_FILTER, padding_type="zero"
            ),
            convolution_op(
                VERTICAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER, padding_type="zero"
            ),
            convolution_op(
                HORIZONTAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER, padding_type="zero"
            ),
        )

        # apply a Gaussian smoothening
        image_list = image.tolist()
        smoother = BaseGaussianFilter()
        image_list = smoother.smooth(image_list)

        # formulate the Hessian matrix
        hessian_xx = np.array(
            convolution_op(image_list, second_order_derivator_x, padding_type="zero")
        )
        hessian_yy = np.array(
            convolution_op(image_list, second_order_derivator_y, padding_type="zero")
        )
        hessian_xy = np.array(
            convolution_op(image_list, second_order_derivator_xy, padding_type="zero")
        )

        # find the determinant
        determinant_hessian = hessian_xx * hessian_yy - (hessian_xy**2)

        # (if needed) set the threshold (should be an actual value, in the range of determinant)
        lower_threshold = self.threshold
        if lower_threshold is None and (percentile is not None):
            self._set_threshold(determinant_hessian, percentile)
            lower_threshold = self._get_threshold()

        # zero out non-keypoints - via thresholding
        keypoints = np.where(
            determinant_hessian > lower_threshold, determinant_hessian, 0
        )

        # zero out any non-keypoints - via non max suppression
        keypoints_suppressed = _suppress(keypoints)

        keypoint_locations = [[], []]
        for y in range(keypoints_suppressed.shape[0]):
            for x in range(keypoints_suppressed.shape[1]):
                if keypoints_suppressed[y][x] > 0:
                    keypoint_locations[0].append(y)
                    keypoint_locations[1].append(x)
        return np.array(keypoint_locations)

    @classmethod
    def find_keypoints_and_visualize(
        cls: "HessianDetector",
        image: np.array,
        image_name: str,
        percentile: float = DEFAULT_PERCENTILE_FOR_DETERMINANT,
    ) -> None:
        # run the algorithm
        keypoint_detector = cls(threshold=None)
        keypoint_locations = keypoint_detector.find_keypoints(image, percentile)
        # show the results -
        plt.imshow(image, cmap="gray")
        plt.scatter(y=keypoint_locations[0], x=keypoint_locations[1], color="red")
        plt.title(f'Keypoints Detected for Image: "{image_name}"')
        plt.show()


if __name__ == "__main__":
    print("script interprets without errors :)")
