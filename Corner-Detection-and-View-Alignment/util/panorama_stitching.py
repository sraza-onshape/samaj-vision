import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


class PanoramaStitcher:
    @staticmethod
    def composite_images(
        left_img,
        right_img,
        affine_transform_matrix_from_left_to_right,
        affine_transform_offset,
        plot_title: str = "",
    ) -> np.ndarray:
        """
        Horizontally align two overlapping images using an affine transform.
        """
        panorama_output_shape = (
            max(left_img.shape[0], right_img.shape[0]),
            left_img.shape[1] + right_img.shape[1],
        )

        # Prepare for the affine transformation
        affine_transform_matrix_from_right_to_left = np.linalg.inv(
            affine_transform_matrix_from_left_to_right
        ).T
        affine_transform_offset_inv = affine_transform_offset
        affine_transform_offset_inv[:2] *= -1

        # Apply the affine transformation to warp the left image onto the right
        left_img_transformed = ndimage.affine_transform(
            left_img,
            affine_transform_matrix_from_right_to_left,
            offset=np.squeeze(affine_transform_offset_inv),
        )

        panorama_img = np.zeros(panorama_output_shape)
        panorama_img[
            0 : left_img.shape[0], 0 : left_img.shape[1]
        ] = left_img_transformed

        panorama_img[: right_img.shape[0], left_img.shape[1] :] = right_img

        # Display the result
        plt.imshow(panorama_img, cmap="gray")
        plt.title(plot_title)
        plt.axis("off")  # for cleanliness, hide the axis lines and ticks
        plt.show()
