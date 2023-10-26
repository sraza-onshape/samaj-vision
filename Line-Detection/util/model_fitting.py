import heapq
import math
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .keypoint_detection import HessianDetector
from util.ops import (
    IDENTITY_FILTER,
    pad as padding_op,
)


class AbstractLineDetector:
    """This class is intentionally left blank."""
    # IDEA: utilize the `abc.ABC` module here, to learn cool new Python skills :)
    pass


class RANSACDetector(AbstractLineDetector):
    """Implements Random Sample Consensus (RANSAC)."""

    REQUIRED_NUM_MODELS_FOR_ASSIGNMENT = 4
    # this is the probability we want to achieve, that we achieve a model with no outliers. 
    CONFIDENCE_LEVEL_FOR_NUM_ITERATIONS = 0.99

    def fit(
        self,
        keypoints: np.array,
        required_number_of_inliers: int = 2,
        distance_threshold: float = 3.0,
        num_top_models_to_return: int = REQUIRED_NUM_MODELS_FOR_ASSIGNMENT,
    ) -> Tuple[List[Tuple[np.array, float]], int]:
        """
        Executes the RANSAC algorithm to fit multiple models across a dataset.

        Note: this function does NOT handle reporting the results of RANSAC.

        Parameters:
            keypoints: np.array: the output of problem 1 (see `HessianDetector.find_keypoints`). Shape is (2, num_keypoint_locations).
            required_number_of_inlier: int: default is to find 2D lines
            distance_threshold: float: default is based on assuming Gaussian noise in a Z-dist --> ergo, 3 * stddev of 1 = 3
            num_top_models_to_return: int. Defaults to 4 (for the purposes of Hw 2, problem 2).

        Returns: (array-like, int): a tuple of two values
            1) a list of n-tuples, representing the top k models. The elements in each tuple represent the following:
                a ) a matrix of the inlier points for that model
                b) the 2nd (and following elements, if there are any) represent the parameters of the model found.
                    E.g., in the case of a line, this would be the slope of the line (and then there'd be a 3rd element also, for the y-intercept).
            2) the number of iterations for which we ran RANSAC
        """
        ### HELPERS
        def _distance_from_a_point_to_a_line(
                slope: float,
                y_intercept: float,
                x_coord, y_coord
            ):
            """
            Based on the math described on Wikipedia:
            https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_an_equation
            """
            # get the params for the formula, from y = mx + b to ax + by + c = 0
            a = slope
            b = -1
            c = y_intercept
            # compute the distance
            numerator = abs(a * x_coord + b * y_coord + c)
            denominator = math.sqrt(sum([a ** 2, b ** 2]))  

            return numerator / denominator 

        def _sample_line(
            keypoint_coordinates: np.array,
            t: float,
            s: int,
        ) -> Tuple[np.array, float]:
            inlier_threshold = t
            # Randomly select a minimal sample of keypoints
            sample_indices = np.random.choice(
                range(keypoint_coordinates.shape[0]),
                size=s, replace=False
            )
            sample = keypoint_coordinates[sample_indices]

            # Estimate a line model (y = mx + b) using the selected points
            point1, point2 = sample
            m = (point2[0] - point1[0]) / (point2[1] - point1[1])
            b = (-1 * m * point1[1]) + point1[0]

            orthogonal_distances = list()
            for point in keypoint_coordinates:
                dist = _distance_from_a_point_to_a_line(m, b, point[1], point[0])
                orthogonal_distances.append(dist)
            inlier_indices = np.array([
                i for i, dist in enumerate(orthogonal_distances) 
                if dist < inlier_threshold
            ]).astype(int)
            inliers = keypoint_coordinates[inlier_indices]

            # ensure the same inliers not used twice, and return the infor about this line
            mask = np.ones(keypoint_coordinates.shape[0], bool)
            mask[inlier_indices] = 0
            # TODO[optimize later]
            # modified_keypoint_coords = keypoint_coordinates[mask]
            modified_keypoint_coords = np.array([
                keypoint_coordinates[i]
                for i, val in enumerate(mask)
                if val == 1
            ])

            return (modified_keypoint_coords, (inliers, (m, b)))
 
        def _run_RANSAC_adaptively(
                add_to_results: Callable,
                s: int,
                total_num_keypoints: int,
                t: float,
                p: float
            ) -> int:
            N = num_iterations = float("inf")
            best_inlier_ratio = float("-inf")
            sample_count = 0
            keypoint_coordinates = keypoints.T  # ordered pairs of (y, x) coordinates

            while num_iterations > sample_count and keypoint_coordinates.shape[0] > s:
                keypoint_coordinates, next_model = _sample_line(
                    keypoint_coordinates,
                    t,
                    s
                )
                add_to_results(next_model)
                num_inliers = next_model[0].shape[0]
                new_inlier_ratio = num_inliers / total_num_keypoints
                if new_inlier_ratio > best_inlier_ratio:
                    # recompute N from e
                    best_inlier_ratio = new_inlier_ratio
                    outlier_ratio = e = 1 - best_inlier_ratio
                    num_iterations = (
                        math.log((1 - p), 10) /
                        math.log(
                            (1 - ((1 - e) ** s)), 
                            10
                        )
                    )
                sample_count += 1
            N = num_iterations
            return N

        def _choose_top_k_results(
                all_results: List[Tuple[np.array, float]],
                k: int
            ) -> List:
            top_k_results_heap = heapq.nlargest(
                k,
                all_results,
                key=lambda group: group[0].shape[0]
            )
            return top_k_results_heap

        ### DRIVER
        # map input args to parameters of RANSAC
        s = required_number_of_inliers
        total_num_keypoints = keypoints.shape[1]
        t = distance_threshold
        p = self.CONFIDENCE_LEVEL_FOR_NUM_ITERATIONS

        # init return value 
        results = list()

        # populate the full list of RANSAC results
        N = _run_RANSAC_adaptively(
            results.append,
            s,
            total_num_keypoints,
            t,
            p
        )

        # return the top results
        top_k_results_heap = _choose_top_k_results(results, num_top_models_to_return)
        return top_k_results_heap, N
    
    @classmethod
    def fit_and_report(
        cls: 'RANSACDetector',
        image: np.array,
        keypoint_detector_algorithm: Callable,
        image_name: str = 'Image',
        required_number_of_inliers: int = 2,
        distance_threshold: float = 3.0,
        num_top_models_to_return: int = REQUIRED_NUM_MODELS_FOR_ASSIGNMENT,
    ) -> None:
        """
        Convenience wrapper around the `fit()` method.
        """
        keypoints = keypoint_detector_algorithm(image)
        line_detector = cls()
        top_4_models, num_iterations = line_detector.fit(
            keypoints,
            required_number_of_inliers=required_number_of_inliers,
            distance_threshold=distance_threshold,
            num_top_models_to_return=num_top_models_to_return
        )
        # 2a. Report choices for inlier thresholds, total number of iteraions and confidence values
        print("=============== Horray! You just ran RANSAC :) ===================")
        print(f"Inlier threshold distance: {distance_threshold}, so we can reach a confidence level of approx. 0.95.")
        print(f"Total number of iterations (rounded to the nearest integer): {int(num_iterations)}.")
        print(f"Confidence Level used in Calculating No. of Iterations: {cls.CONFIDENCE_LEVEL_FOR_NUM_ITERATIONS}.")
        # 2b. Overlay line segments in the image by connecting the two extreme inliers of each line.
        plt.imshow(image, cmap="gray")
        for model_data in top_4_models:
            inliers, line_params = model_data
            slope, y_intercept = line_params
            x_min, x_max = inliers[:, 1].min(), inliers[:, 1].max()
            x_range = np.linspace(x_min, x_max)
            y_range = x_range * slope + y_intercept
            plt.plot(x_range, y_range, color='green', marker='*')
            # Also plot the inliers as 3×3 squares.
            row_wise_coords = inliers.T
            plt.scatter(
                y=row_wise_coords[0],
                x=row_wise_coords[1],
                color="b", marker="s"
            )
        plt.title(f'RANSAC: Lines Detected for Image: "{image_name}"')
        plt.show()


class HoughTransformDetector(AbstractLineDetector):
    """Implements a Hough Transform for Line Detection."""

    REQUIRED_NUM_MODELS_FOR_ASSIGNMENT = 4

    def fit(
        self,
        image: np.array,
        keypoints: np.array,
        rho_bin_size: float = 1,
        theta_bin_size: float = np.pi / 180,
        num_top_models_to_return: int = REQUIRED_NUM_MODELS_FOR_ASSIGNMENT,
    ) -> Tuple[List[Tuple[int, int]], np.array]:
        """
        Executes a Hough transform to fit multiple models across a dataset.

        Parameters:
            image: np.array: pixel raster matrix.
            keypoints: np.array: the output of problem 1 (see `HessianDetector.find_keypoints`). Shape is (2, num_keypoint_locations).
            rho_bin_size, theta_bin_size (float, float): the resolutions of 1 discretized "bucket" in the voting histogram.
            num_top_models_to_return: int. Defaults to 4 (for the purposes of Hw 2, problem 3).

        Returns: (array-like, int): a tuple of two values
            1) a list of four 2-tuples - each represents a cell in the accumulator with the top-4 most votes.
            2) np.array: the accumulator, i.e., the histogram of votes in Hough space (using polar coordinates).
        """
        ### HELPERS
        def _non_max_suppression(matrix):
            '''prevent potential loss of keypoints via padding'''
            keypoints = matrix
            padded_matrix, num_added_rows, num_added_cols = padding_op(
                keypoints.tolist(),
                img_filter=IDENTITY_FILTER,
                stride=1,
                padding_type="zero",
            )
            # traverse the matrix, to begin non-max suppression
            for center_val_row in range(num_added_rows // 2, padded_matrix.shape[0] - (num_added_rows // 2)):
                for center_val_col in range(num_added_cols // 2, padded_matrix.shape[1] - (num_added_cols // 2)):
                    # determine if the given value should be suppressed, or its neighbors
                    center_val = padded_matrix[center_val_row][center_val_col]
                    neighbors = padded_matrix[
                        center_val_row - 1 : center_val_row + 2,
                        center_val_col - 1 : center_val_col + 2,
                    ]
                    neighbors[1][1] = 0  # hack to prevent the center value from "self-suppressing" (I have no idea, I made that term up)
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
                num_added_rows // 2 : padded_matrix.shape[0] - (num_added_rows // 2),
                num_added_cols // 2 : padded_matrix.shape[1] - (num_added_cols // 2),
            ]

        ### DRIVER
        keypoint_coords = keypoints.T
        # Define the parameter space for the Hough transform
        max_rho = np.hypot(image.shape[0], image.shape[1])

        # Calculate the new size of the accumulator array
        rho_bins = int(2 * max_rho / rho_bin_size)
        theta_bins = int(np.pi / theta_bin_size)

        # Create the accumulator array with the new size
        accumulator = np.zeros((rho_bins, theta_bins))

        # Voting in the accumulator array
        for point in keypoint_coords:
            for theta in np.arange(0, np.pi, theta_bin_size):
                col = x_coord = point[1]
                row = y_coord = point[0]
                rho = int(x_coord * np.cos(theta) + y_coord * np.sin(theta))
                rho_bin = int(rho / rho_bin_size)
                theta_bin = int(theta / theta_bin_size)
                accumulator[rho_bin, theta_bin] += 1

        # Thresholding to identify detected lines --> use non max suppression
        local_max_accumulator = _non_max_suppression(accumulator)

        # Extract and convert a sampling of detected lines to Cartesian coordinates
        local_max_accumulator_flat = local_max_accumulator.reshape(1, -1)
        least_to_greatest_votes = np.argsort(local_max_accumulator_flat)[0, -1 * num_top_models_to_return:]
        sample_indices = [
            (
                # work back into the dims of the 2D accumulator matrix, given 1D index into the flattened array
                flat_index // local_max_accumulator.shape[0], 
                ((flat_index // local_max_accumulator.shape[1]) % local_max_accumulator.shape[1]) - 1
            )
            for flat_index in least_to_greatest_votes
        ]
        return sample_indices, accumulator
    
    
    @classmethod
    def fit_and_report(
        cls: 'HoughTransformDetector',
        image: np.array,
        image_name: str,
        keypoint_detector_algorithm: Callable,
        rho_bin_size: float = 1,
        theta_bin_size: float = np.pi / 180,
        num_top_models_to_return: int = REQUIRED_NUM_MODELS_FOR_ASSIGNMENT,
    ) -> None:
        """Plot the image and detected lines."""
        keypoints = keypoint_detector_algorithm(image)
        detector = cls()

        sample_indices, accumulator = detector.fit(
            image=image,
            keypoints=keypoints,
            rho_bin_size=rho_bin_size,
            theta_bin_size=theta_bin_size,
            num_top_models_to_return=num_top_models_to_return,
        )

        # Create a figure with matching dimensions to the input image
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Plot the image and detected lines
        for rho_bin, theta_bin in sample_indices:
            theta = theta_bin * theta_bin_size
            rho = rho_bin * rho_bin_size
            # x_intercept = (rho - (0 * np.sin(theta))) / (np.cos(theta))
            # y_intercept = (rho - (0 * np.cos(theta))) / (np.sin(theta))

            # ax1.plot([x_intercept, 0], [0, y_intercept], color='green')
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Ensure endpoints are within image bounds
            x1 = max(0, min(x1, image.shape[1] - 1))
            y1 = max(0, min(y1, image.shape[0] - 1))
            x2 = max(0, min(x2, image.shape[1] - 1))
            y2 = max(0, min(y2, image.shape[0] - 1))

            ax1.plot([x1, x2], [y1, y2], color='green')
        ax1.set_title(f"Detected Lines on \"{image_name}\" Image (Cartesian Coordinates)")
        ax1.imshow(image, cmap='gray')  # plot the image in the background

        # Plot the accumulator array as a 2D histogram
        accumulator_in_pixel_scale = (
            (accumulator - accumulator.min()) / 
            (accumulator.max() - accumulator.min())
        ) * 255.

        ax2.imshow(accumulator_in_pixel_scale, cmap='gray')
        ax2.set_title('Votes in Hough Space (Polar Coordinates)')
        ax2.set_xlabel('Theta (radians)')
        ax2.set_ylabel('Rho (pixels)')


if __name__ == "__main__":
    # IDEA: add real test cases
    print("code interprets without errors :)")
