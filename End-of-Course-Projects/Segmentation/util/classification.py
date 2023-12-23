from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .clustering import KMeans


class PixelClassifier:
    """Class is intended to be used only for segmenting out 1 area, of a single image."""

    WHITE_PIXEL = np.array(
        [
            255.0,
            255,
            255.0,
        ]
    )

    def _reshape_into_5d(self, img: np.ndarray) -> np.ndarray:
        """
        Transforms a RGB image into an array of 5D (x, y, r, g, b) points.

        Parameters
            img(array): a single RGB image. Has shape of (rows, cols, 3).

        Returns: array of shape (rows * cols, 5).
        """
        pixels = np.zeros((img.shape[1] * img.shape[0], 5))
        pixel_index = 0
        for x in np.arange(img.shape[1]):
            for y in np.arange(img.shape[0]):
                pixels[pixel_index] = np.concatenate([[x, y], img[y, x, :]])
                pixel_index += 1

        return pixels

    def __init__(self, original_img: np.ndarray, mask_img: np.ndarray) -> None:
        """
        Partitions the pixels in a given image for the purpose of supervised binary segmentation.

        Parameters:
            original_img(NumPy array): RGB image of shape (rows, cols, 3) that we want to segment.
            mask_img(NumPy array): RGB image with the same shape as the original_img.
                                    You are required to preprocess this such that the area we want to segment out,
                                    has been colored in white (and no other white regions exist in the image).

        Returns: None
        """
        original_in_5d = self._reshape_into_5d(original_img)
        mask_in_5d = self._reshape_into_5d(mask_img)

        # separate pixels into the pos and neg classes
        self.positive_examples = list()
        self.negative_examples = list()
        self.clusters = list()  # for now, leave empty

        for pixel_index in np.arange(original_in_5d.shape[0]):
            original_5d = original_in_5d[pixel_index, :]
            mask_5d = mask_in_5d[pixel_index, :]

            if np.equal(mask_5d[2:], self.WHITE_PIXEL).all():
                self.positive_examples.append(original_5d)
            else:  # the pixel belongs to the negative set
                self.negative_examples.append(original_5d)

        self.positive_examples = np.array(self.positive_examples)
        self.negative_examples = np.array(self.negative_examples)

    def train(
        self, num_clusters_per_class: int = 10, max_iter: int = float("inf")
    ) -> None:
        """
        Unsupervised clustering of the pixels in the image, done separately on both classes.

        Parameters:
            num_clusters_per_class(int): the value of K we will use in the clusterings.
            max_iter(int): upper bound on how many iterations the clustering is allowed to take.

        Returns: None
        """
        # cluster separately for both classes
        kmeans_positive = KMeans(k=num_clusters_per_class)
        kmeans_positive.fit(self.positive_examples, max_iter=max_iter)
        kmeans_negative = KMeans(k=num_clusters_per_class)
        kmeans_negative.fit(self.negative_examples, max_iter=max_iter)

        # concat the clusters - first 10 are sky, 2nd are non-sky
        self.clusters = np.concatenate(
            [
                np.array(list(kmeans_positive.centroid_coords.values())),
                np.array(list(kmeans_negative.centroid_coords.values())),
            ],
            axis=0,
        )

    def predict(
        self,
        test_img: np.ndarray,
        positive_color: np.ndarray,
    ) -> np.ndarray:
        """
        Perform segmentation on a new image not seen in training.

        Pixels in the region labeled as the "positive" class will get a new color. Others will remain the same.

        Parameters:
            test_img(NumPy array): a new RGB image in "channels_last" format, e.g., the shape can be (rows, cols, 3).
            positive_color(NumPy array): expects array of shape (3,), of the RGB value to paint the positive region.

        Returns: NumPy array that is the segmented image.
        """
        ### HELPER(S)
        def _predict_visual_word(
            test_pixels_in_5d: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """For each pixel, determine if it's positive or negative."""
            measure_distance = lambda x, y: np.linalg.norm(x - y, axis=1)
            distances = np.apply_along_axis(
                measure_distance, arr=test_pixels_in_5d, axis=1, y=self.clusters
            )
            centroid_assignments = np.argmin(distances, axis=1)

            # separate the two classes
            num_centroids = self.clusters.shape[0]
            positive_labels = test_pixels_in_5d[
                centroid_assignments < (num_centroids // 2)
            ]
            negative_labels = test_pixels_in_5d[
                centroid_assignments >= (num_centroids // 2)
            ]

            return positive_labels, negative_labels

        ### DRIVER
        # predict pos/negative classes for the new img
        test_pixels_in_5d = self._reshape_into_5d(test_img)
        positive_labels, negative_labels = _predict_visual_word(test_pixels_in_5d)

        # visualize the results
        segmented_img = np.zeros_like(test_img)
        for pixel_5d in positive_labels:
            pixel_5d = pixel_5d.reshape(1, 5).astype(int)
            x, y = pixel_5d[0, :2]
            segmented_img[y, x, :] = positive_color[1]
        for pixel_5d in negative_labels:
            pixel_5d = pixel_5d.reshape(1, 5).astype(int)
            x, y = pixel_5d[0, :2]
            segmented_img[y, x, :] = pixel_5d[0, 2:]

        return segmented_img

    def classify_and_visualize(
        self,
        test_img: List[np.ndarray],
        test_img_title: List[str],
        positive_class_name: str,
        positive_color_pair: Tuple[str, np.ndarray] = (
            "yellow",
            np.array([255, 255, 0]),
        ),
    ) -> None:
        """Convenience wrapper for both computing and visualizing the segmented image."""
        positive_color_name, positive_color_rgb = positive_color_pair

        segmented_img = self.predict(
            test_img,
            positive_color_rgb,
        )

        # do plotting
        plt.imshow(segmented_img)
        plt.title(
            f'Segmentation for "{test_img_title}" ({positive_class_name} shown in {positive_color_name})'
        )
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    ...
