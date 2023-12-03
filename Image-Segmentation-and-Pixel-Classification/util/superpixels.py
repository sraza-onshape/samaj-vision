import cv2
import functools
import numpy as np

from .gaussian_derivative import GaussianDerivativeFilter
from . import ops


class SLIC:
    def execute_and_visualize(
        self,
        img: np.array,
        step_size: int = 50,
    ) -> None:
        """TODO[Zain]: Add docstrings"""
        ### HELPERS

        ### DRIVER
        S = step_size  # aliasing for convenience

        # Divide the image in blocks
        pixel_block_boundaries_x = np.arange(0, img.shape[1], S)
        if pixel_block_boundaries_x[-1] < img.shape[1] - 1:
            pixel_block_boundaries_x = np.concatenate(
                [pixel_block_boundaries_x, [img.shape[1] - 1]]
            )

        pixel_block_boundaries_y = np.arange(0, img.shape[0], S)
        if pixel_block_boundaries_y[-1] < img.shape[0] - 1:
            pixel_block_boundaries_y = np.concatenate(
                [pixel_block_boundaries_y, [img.shape[0] - 1]]
            )

        # initialize a centroid at the center of each block.
        centroid_coordinates = np.zeros(
            (pixel_block_boundaries_x.shape[0] * pixel_block_boundaries_y.shape[0], 2)
        )

        centroid_coordinates_index = 0
        for index_x in range(pixel_block_boundaries_x.shape[0] - 1):
            for index_y in range(pixel_block_boundaries_y.shape[0] - 1):
                block_coords_x = np.array(
                    [
                        pixel_block_boundaries_x[index_x],
                        pixel_block_boundaries_x[index_x + 1],
                    ]
                )
                block_coords_y = np.array(
                    [
                        pixel_block_boundaries_y[index_y],
                        pixel_block_boundaries_y[index_y + 1],
                    ]
                )
                centroid_coordinates[centroid_coordinates_index] = [
                    block_coords_y.mean(),
                    block_coords_x.mean(),
                ]
                centroid_coordinates_index += 1

        # compute gradient magnitude
        grad_img = img.copy()
        derivator = GaussianDerivativeFilter()
        for channel_index in range(img.shape[2]):
            channel = img[:, :, channel_index]
            partial_derivative_x, partial_derivative_y = derivator._compute_derivatives(
                channel
            )
            magnitude_matrix = derivator._compute_magnitude(
                partial_derivative_x, partial_derivative_y
            )
            grad_img[:, :, channel_index] = magnitude_matrix
        combined_grad_magnitude = np.sqrt(np.sum(grad_img**2, axis=2))

        # Local Shift: move centroids to the smallest magnitude position in 3x3 windows
        def _find_smallest_grad_position(current_coordinates, combined_grad_magnitude):
            window = np.zeros((3, 3))
            window[:, :] = np.inf
            # try to fill as much of the window as possible, with true values
            current_y, current_x = current_coordinates
            if (
                combined_grad_magnitude.shape[0] - current_y >= 1
                and combined_grad_magnitude.shape[1] - current_x >= 1
            ):
                sub_image = combined_grad_magnitude[
                    current_y - 1 : current_y + 2, current_x - 1 : current_x + 2
                ]
                window[:, :] = sub_image
            else:  # assume neither coord is 0, and we're at the boundary
                sub_image = combined_grad_magnitude[current_y - 1 :, current_x - 1 :]
                window[: sub_image.shape[0], : sub_image.shape[1]] = sub_image

            smallest_magnitude_coords_in_window_space_1d = np.argsort(
                window, axis=None
            )[0]
            smallest_magnitude_coords_in_window_space_2d = ops.convert_1d_indices_to_2d(
                window, np.array([smallest_magnitude_coords_in_window_space_1d])
            )
            smallest_magnitude_coords_in_channel_space_2d = (
                -1 * (1 - smallest_magnitude_coords_in_window_space_2d[0]),
                -1 * (1 - smallest_magnitude_coords_in_window_space_2d[1]),
            )
            return smallest_magnitude_coords_in_channel_space_2d

        _find_smallest_grad_position_short = functools.partial(
            _find_smallest_grad_position,
            combined_grad_magnitude=combined_grad_magnitude,
        )

        shifted_centroid_centers = np.apply_along_axis(
            _find_smallest_grad_position_short,
            axis=0,
            arr=centroid_coordinates,
        )
        # TODO[Zain] Centroid Update...
