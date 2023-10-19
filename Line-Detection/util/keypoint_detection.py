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
        # TODO[debug dims of keypoints matrix]
        (
            second_order_derivator_x,
            second_order_derivator_y,
            second_order_derivator_xy,
        ) = (
            convolution_op(HORIZONTAL_SOBEL_FILTER, HORIZONTAL_SOBEL_FILTER, padding_type="zero"),
            convolution_op(VERTICAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER, padding_type="zero"),
            convolution_op(HORIZONTAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER, padding_type="zero")
        )
        print(f"Actual shape of 'second_order_derivator_x': {len(second_order_derivator_x), len(second_order_derivator_x[0])}, expected 3 x 3")
        print(f"Actual shape of 'second_order_derivator_y': {len(second_order_derivator_y), len(second_order_derivator_y[0])}, expected 3 x 3")
        print(f"Actual shape of 'second_order_derivator_xy': {len(second_order_derivator_xy), len(second_order_derivator_xy[0])}, expected 3 x 3")

        # formulate the Hessian matrix 
        image_list = image.tolist()
        IR = convolution_op(image_list, second_order_derivator_x, padding_type="zero")
        hessian_xx = np.array(IR)
        hessian_xy = np.array(convolution_op(image_list, second_order_derivator_xy, padding_type="zero"))
        hessian_yy = np.array(convolution_op(image_list, second_order_derivator_y, padding_type="zero"))

        print(f"Actual shape of 'hessian_xx': {len(IR), len(IR[0])}, expected 548 x 407")
        print(f"Actual shape of 'hessian_xy': {hessian_xy.shape}, expected 548 x 407")
        print(f"Actual shape of 'hessian_yy': {hessian_yy.shape}, expected 548 x 407")

        # find the determinant
        determinant_hessian = hessian_xx * hessian_yy - hessian_xy**2

        print(f"Actual shape of 'determinant_hessian': {determinant_hessian.shape}, expected 548 x 407")

        # (if needed) set the threshold
        lower_threshold = self.threshold
        if lower_threshold is None:
            self._set_threshold(determinant_hessian)
            lower_threshold = self._get_threshold()

        # zero out any non-keypoints - and then we're done!
        keypoints = np.where(determinant_hessian > lower_threshold, determinant_hessian, 0)
        print(f"Actual shape of 'keypoints': {keypoints.shape}, expected 548 x 407")
        return keypoints

    @classmethod
    def find_keypoints_and_visualize(
        cls: "HessianDetector",
        image: np.array,
        image_name: str,
        threshold: float = None,
    ) -> None:
        # run the algorithm
        keypoint_detector = cls(threshold=threshold)
        keypoints = keypoint_detector.find_keypoints(image)
        # TODO[debug overly of keypoints in plot]
        print(keypoints.shape, keypoints[0], keypoints[1])
        # show the results
        plt.imshow(image, cmap='gray')
        # overlay the keypoints
        # plt.scatter(keypoints[1], keypoints[0], color='red', s=5)
        plt.title(f"Keypoints Detected for Image: \"{image_name}\"")
        # plt.show()


if __name__ == "__main__":
    # TODO: add test cases for find_keypoints_and_visualize()
    ...
