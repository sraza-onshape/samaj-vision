import heapq
import math
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .keypoint_detection import HessianDetector


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


if __name__ == "__main__":
    # IDEA: add real test cases
    print("code interprets without errors :)")
