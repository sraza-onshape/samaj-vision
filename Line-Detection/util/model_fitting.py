import heapq
import math
from typing import List, Tuple

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
            sample = np.random.choice(keypoint_coordinates, size=s, replace=False)

            # Estimate a line model (y = mx + b) using the selected points
            point1, point2 = sample
            m = (point2[0] - point1[0]) / (point2[1] - point1[1])
            b = (-1 * m * point1[1]) + point1[0]

            orthogonal_distances = list()
            for point in keypoint_coordinates:
                dist = _distance_from_a_point_to_a_line(m, b, point[1], point[0])
                orthogonal_distances.append(dist)
            inliers = np.array([
                keypoint_coordinates[i] for i, dist in enumerate(orthogonal_distances) 
                if dist < inlier_threshold
            ])

            return (inliers, (m, b))
 
        def _run_RANSAC_adaptively(
                add_to_results: function,
                s: int,
                total_num_keypoints: int,
                t: float,
                p: float
            ) -> int:
            N = num_iterations = float("inf")
            best_inlier_ratio = float("-inf")
            sample_count = 0
            keypoint_coordinates = keypoints.T  # ordered pairs of (y, x) coordinates

            while num_iterations > sample_count:
                next_model = _sample_line(
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
            return N

        def _choose_top_k_results(
                all_results: List[Tuple[np.array, float]],
                k: int
            ) -> List:
            top_k_results_heap = []
            for index in range(k):
                heapq.heappush(top_k_results_heap, all_results[index])
            for index in range(k, len(all_results)):
                model_tuple = all_results[index]
                current_num_inliers, kth_largest_inliers = (
                    model_tuple[0].shape[0],
                    top_k_results_heap[0][0].shape[0]
                )
                if current_num_inliers > kth_largest_inliers:
                    _ = heapq.heappushpop(top_k_results_heap, model_tuple)
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
        keypoint_detector: HessianDetector,
        image_name: str = 'Image',
        required_number_of_inliers: int = 2,
        distance_threshold: float = 3.0,
        num_top_models_to_return: int = REQUIRED_NUM_MODELS_FOR_ASSIGNMENT,
    ) -> None:
        """
        TODO:
            1) detect keypoints on the image
            2) get the top 4 models via RANSAC
            3) do parts 2a and 2b in the assignment --> see bottom of ChatGPT thread
        """
        pass


if __name__ == "__main__":
    # IDEA: add real test cases
    print("code interprets without errors :)")