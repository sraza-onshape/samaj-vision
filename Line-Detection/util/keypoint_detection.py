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
            convolution_op(HORIZONTAL_SOBEL_FILTER, HORIZONTAL_SOBEL_FILTER),
            convolution_op(VERTICAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER),
            convolution_op(HORIZONTAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER)
        )

        # formulate the Hessian matrix 
        image_list = image.tolist()
        hessian_xx = convolution_op(image_list, second_order_derivator_x)
        hessian_xy = convolution_op(image_list, second_order_derivator_xy)
        hessian_yy = convolution_op(image_list, second_order_derivator_y)

        matrix = np.array([
            [hessian_xx, hessian_xy],
            [hessian_xy, hessian_yy]
        ])

        # find the determinant
        determinant_hessian = hessian_xx * hessian_yy - hessian_xy**2

        # (if needed) set the threshold
        lower_threshold = self.threshold
        if lower_threshold is None:
            self._set_threshold(matrix)
            lower_threshold = self._get_threshold()

        # zero out any non-keypoints - and then we're done!
        keypoints = np.where(determinant_hessian > lower_threshold, determinant_hessian, 0)
        return keypoints

    @classmethod
    def find_keypoints_and_visualize(cls: "HessianDetector"):
        # TODO
        pass


if __name__ == "__main__":
    # TODO: add test cases
    ...
