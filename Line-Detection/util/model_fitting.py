import heapq
from typing import List, Tuple

import numpy as np

class AbstractLineDetector:
    """This class is intentionally left blank."""
    # TODO: utilize the `abc.ABC` module here, to learn cool new Python skills :)
    pass


class RANSACDetector(AbstractLineDetector):
    """Implements Random Sample Consensus (RANSAC)."""

    REQUIRED_NUM_MODELS_FOR_ASSIGNMENT = 4
    CONFIDENCE_LEVEL_FOR_NUM_ITERATIONS = 0.99

    def fit_and_report(
        self,
        keypoints: np.array,
        required_number_of_inliers: int = 2,
        distance_threshold: float = 3.0,
        num_top_models_to_return: int = REQUIRED_NUM_MODELS_FOR_ASSIGNMENT,
        confidence: float = CONFIDENCE_LEVEL_FOR_NUM_ITERATIONS,
    ) -> List[Tuple[np.array, float]]:
        """
        Executes the RANSAC algorithm to fit multiple models across a dataset.

        Note: this function does NOT handle reporting the results of RANSAC.

        Parameters:
            keypoints: np.array: the output of problem 1 (see `HessianDetector.find_keypoints`). Shape is (2, num_keypoint_locations).
            required_number_of_inlier: int: default is to find 2D lines
            distance_threshold: float: default is based on assuming Gaussian noise in a Z-dist --> ergo, 3 * stddev of 1 = 3
            num_top_models_to_return: int. Defaults to 4 (for the purposes of Hw 2, problem 2).
            confidence: float: this is the probability we want to achieve, that we achieve a model with no outliers. 

        Returns: array-like: a list of n-tuples, representing the top k models. The elements in each tuple represent the following:
            1) a matrix of the inlier points for that model
            2) the 2nd (and following elements, if there are any) represent the parameters of the model found.
                E.g., in the case of a line, this would be the slope of the line (and then there'd be a 3rd element also, for the y-intercept).
        """
        ### HELPERS
        def _choose_top_k_results(all_results, k):
            top_k_results_heap = all_results[:num_top_models_to_return]
            for index in range(num_top_models_to_return, len(all_results)):
                model_tuple = all_results[index]
                current_num_liers, kth_largest_inliers = (
                    model_tuple[0].shape[0],
                    top_k_results_heap[0].shape[0]
                )
                if current_num_liers > kth_largest_inliers:
                    _ = heapq.heappushpop(top_k_results_heap, model_tuple)
            return top_k_results_heap

        ### DRIVER
        # map input args to parameters of RANSAC
        s = required_number_of_inliers
        t = distance_threshold
        p = confidence

        # init return value 
        results = list()

        # TODO[populate the full list of results]
        ...

        # return the top results
        top_k_results_heap = _choose_top_k_results(results, num_top_models_to_return)
        return top_k_results_heap





