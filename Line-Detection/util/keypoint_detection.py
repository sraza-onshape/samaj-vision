from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .ops import convolution as convolution_op
from .ops import HORIZONTAL_SOBEL_FILTER
from .ops import VERTICAL_SOBEL_FILTER


class AbstractKeypointDetector:
    """This class is intentionally left blank."""

    # TODO: utilize the `abc.ABC` module here, to learn cool new Python skills :)
    pass


class HessianDetector(AbstractKeypointDetector):
    """Utilizes the determinant of the Hessian matrix to find keypoints."""

    DEFAULT_PERCENTILE_FOR_DETERMINANT = 75  # totally arbitrary

    def __init__(self, threshold: float = None):
        self.threshold = threshold

    def _set_threshold(self, values: np.asarray) -> None:
        self.threshold = np.percentile(values, self.DEFAULT_PERCENTILE_FOR_DETERMINANT)

    def _get_threshold(self) -> float:
        return self.threshold

    def find_keypoints(self, image: np.array) -> np.array:
        # compute the second order partial derivatives
        (
            second_order_derivator_x,
            second_order_derivator_y,
            second_order_derivator_xy,
        ) = (
            convolution_op(HORIZONTAL_SOBEL_FILTER, HORIZONTAL_SOBEL_FILTER, padding_type="zero"),
            convolution_op(VERTICAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER, padding_type="zero"),
            convolution_op(HORIZONTAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER, padding_type="zero")
        )

        # formulate the Hessian matrix 
        image_list = image.tolist()
        IR = convolution_op(image_list, second_order_derivator_x, padding_type="zero")
        hessian_xx = np.array(IR)
        hessian_xy = np.array(convolution_op(image_list, second_order_derivator_xy, padding_type="zero"))
        hessian_yy = np.array(convolution_op(image_list, second_order_derivator_y, padding_type="zero"))

        # find the determinant
        determinant_hessian = hessian_xx * hessian_yy - hessian_xy**2

        # (if needed) set the threshold
        lower_threshold = self.threshold
        if lower_threshold is None:
            self._set_threshold(determinant_hessian)
            lower_threshold = self._get_threshold()

        # zero out any non-keypoints - via thresholding
        keypoints = np.where(determinant_hessian > lower_threshold, determinant_hessian, 0)

        # zero out any non-keypoints - via non max suppression
        print(keypoints.shape)
        keypoint_locations = [[], []]
        for y in range(keypoints.shape[0]):
            for x in range(keypoints.shape[1]):
                if keypoints[y][x] > 0:
                    keypoint_locations[0].append(y)
                    keypoint_locations[1].append(x)
        return keypoint_locations

    @classmethod
    def find_keypoints_and_visualize(
        cls: "HessianDetector",
        image: np.array,
        image_name: str,
        threshold: float = None,
    ) -> None:
        # run the algorithm
        keypoint_detector = cls(threshold=threshold)
        keypoint_locations = keypoint_detector.find_keypoints(image)
        print(len(keypoint_locations), len(keypoint_locations[0]), len(keypoint_locations[1]))
        # show the results - 
        plt.imshow(image, cmap='gray')
        plt.scatter(y=keypoint_locations[0], x=keypoint_locations[1], color='red')
        plt.title(f"Keypoints Detected for Image: \"{image_name}\"")
        plt.show()


if __name__ == "__main__":
    # TODO: add test cases for find_keypoints_and_visualize()
    ...
