import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


class PanoramaStitcher:
    @staticmethod
    def composite_images(
        left_img,
        right_img,
        affine_transform_matrix,
        affine_transform_offset,
        plot_title: str = "",
    ):
        """
        Horizontally align two overlapping images using an affine transform.
        """
        # Determine output shape for the stitched panorama
        output_shape = (
            max(left_img.shape[0], right_img.shape[0]),
            left_img.shape[1] + right_img.shape[1],
        )

        # Apply the affine transformation to warp the left image onto the right
        left_img_transformed = ndimage.affine_transform(
            left_img,
            affine_transform_matrix,
            offset=np.squeeze(affine_transform_offset),
        )
        stitched_image = np.zeros(output_shape)
        stitched_image[
            0 : left_img.shape[0], 0 : left_img.shape[1]
        ] = left_img_transformed

        # Blend the right image onto the panorama
        stitched_image[: right_img.shape[0], left_img.shape[1] :] = right_img

        # Display the result
        plt.imshow(stitched_image)
        plt.title(plot_title)
        plt.axis("off")  # for cleanliness, hide the axis lines and ticks
        plt.show()
