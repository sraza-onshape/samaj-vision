# Homework 2: Line Detection

## Student Info
- **Name**: Syed Zain Raza
- **Campus-wide ID**: 20011917

## Problem 1: Preprocessing

### Short Explanation

The `HessianDetector` class implements an algorithm for keypoint detection via the Hessian matrix of an image, as discussed in class. 
The `fit()` method handles the main steps of the algorithm - it preprocesses the image using a Gaussian filter (I reused the code from my homework 1 solution), utilizes the Sobel filter to compute the second derivative. Finally, we do a few vectorized operations to get the determinant, which is thresholded (the user provides the threshold), and finally, non-max suppression is used to further reduce the number of kyepoints. The final output of the `fit()` method is a 2D NumPy array in the form `np.array([y1, y2, ..., yn], [x1, x2, ..., xn])`, where each column represents the coordinate of a keypoint in the image.

### Code

```python
# util/keypoint_detection.py
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .gaussian_base import BaseGaussianFilter
from .ops import (
    convolution as convolution_op,
    pad as padding_op,
    HORIZONTAL_SOBEL_FILTER,
    IDENTITY_FILTER,
    VERTICAL_SOBEL_FILTER,
)

...

class HessianDetector(AbstractKeypointDetector):
    """Utilizes the determinant of the Hessian matrix to find keypoints."""

    DEFAULT_PERCENTILE_FOR_DETERMINANT = 75

    def __init__(self, threshold: float = None):
        self.threshold = threshold  # should be an exact value, expected to be in the range of the image determinant

    def _set_threshold(self, values: np.array, percentile: float) -> None:
        """Sets the determinant based on a percentile of an n-dimensional array (representing the range of some function)."""
        self.threshold = np.percentile(values, percentile)

    def _get_threshold(self) -> float:
        return self.threshold

    def find_keypoints(
        self, image: np.array, percentile: float = DEFAULT_PERCENTILE_FOR_DETERMINANT
    ) -> np.array:
        ### HELPERS
        def _suppress(keypoints: np.array) -> np.array:
            """After the determinant has been thresholded, use non-max suppression to recover more distinguishable keypoints."""
            # prevent potential loss of keypoints via padding
            padded_matrix, num_added_rows, num_added_cols = padding_op(
                keypoints.tolist(),
                img_filter=IDENTITY_FILTER,
                stride=1,
                padding_type="zero",
            )
            # traverse the matrix, to begin non-max suppression
            for center_val_row in range(
                num_added_rows // 2, padded_matrix.shape[0] - (num_added_rows // 2)
            ):
                for center_val_col in range(
                    num_added_cols // 2, padded_matrix.shape[1] - (num_added_cols // 2)
                ):
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

            # return the modified matrix
            return padded_matrix[
                num_added_rows // 2 : keypoints.shape[0] - (num_added_rows // 2),
                num_added_cols // 2 : keypoints.shape[1] - (num_added_cols // 2),
            ]

        ### DRIVER
        # compute the second order partial derivatives
        (
            second_order_derivator_x,
            second_order_derivator_y,
            second_order_derivator_xy,
        ) = (
            convolution_op(
                HORIZONTAL_SOBEL_FILTER, HORIZONTAL_SOBEL_FILTER, padding_type="zero"
            ),
            convolution_op(
                VERTICAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER, padding_type="zero"
            ),
            convolution_op(
                HORIZONTAL_SOBEL_FILTER, VERTICAL_SOBEL_FILTER, padding_type="zero"
            ),
        )

        # apply a Gaussian smoothening
        image_list = image.tolist()
        smoother = BaseGaussianFilter()
        image_list = smoother.smooth(image_list)

        # formulate the Hessian matrix
        hessian_xx = np.array(
            convolution_op(image_list, second_order_derivator_x, padding_type="zero")
        )
        hessian_yy = np.array(
            convolution_op(image_list, second_order_derivator_y, padding_type="zero")
        )
        hessian_xy = np.array(
            convolution_op(image_list, second_order_derivator_xy, padding_type="zero")
        )

        # find the determinant
        determinant_hessian = hessian_xx * hessian_yy - (hessian_xy**2)

        # (if needed) set the threshold (should be an actual value, in the range of determinant)
        lower_threshold = self.threshold
        if lower_threshold is None and (percentile is not None):
            self._set_threshold(determinant_hessian, percentile)
            lower_threshold = self._get_threshold()

        # zero out non-keypoints - via thresholding
        keypoints = np.where(
            determinant_hessian > lower_threshold, determinant_hessian, 0
        )

        # zero out any non-keypoints - via non max suppression
        keypoints_suppressed = _suppress(keypoints)

        keypoint_locations = [[], []]
        for y in range(keypoints_suppressed.shape[0]):
            for x in range(keypoints_suppressed.shape[1]):
                if keypoints_suppressed[y][x] > 0:
                    keypoint_locations[0].append(y)
                    keypoint_locations[1].append(x)
        return np.array(keypoint_locations)

    @classmethod
    def find_keypoints_and_visualize(
        cls: "HessianDetector",
        image: np.array,
        image_name: str,
        percentile: float = DEFAULT_PERCENTILE_FOR_DETERMINANT,
    ) -> None:
        # run the algorithm
        keypoint_detector = cls(threshold=None)
        keypoint_locations = keypoint_detector.find_keypoints(image, percentile)
        # show the results -
        plt.imshow(image, cmap="gray")
        plt.scatter(y=keypoint_locations[0], x=keypoint_locations[1], color="red")
        plt.title(f'Keypoints Detected for Image: "{image_name}"')
        plt.show()

```

### Results

To detect keypoints on the `road.png` image, 
I executed the following code, in `problems.ipynb`:

```python
HessianDetector.find_keypoints_and_visualize(
    np.array(original_image), 
    "Road",
    percentile=68.05  # note: I had to run the code several times to arrive at this value
)
```

The resulting image looks like the following, `problem1_preprocessing/hessian_matrix_keypoints.png`:

![visual of the output for plot](./problem1_preprocessing/hessian_matrix_keypoints.png).

## Problem 2: RANSAC

### Short Explanation
The `RANSACDetector` class has a `fit()` method that accepts the keypoints outputted by the `HessianDetector.fit()` method, and uses that to find a user-specified number of lines (in this case, it was `4` for the purposes of the assignment) in the image. We run RANSAC adaptively, so that there is no need to set the number of iterations ahead of time. The `distance_threshold` is set to `3`` by default, because I assume Gaussian noise in the image that falls in a Z-distribution; ergo, `3 * stddev of 1 = 3`. At the end, the lines with the most support are picked out using a min heap, via Python's `heapq` module. 

### Code
```python
# util/model_fitting.py

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
```

### Results

To detect 4 lines using RANSAC on the `road.png` image, 
I executed the following code, in `problems.ipynb`:

```python
import functools

# partially fill in the function to get keypoints - so it can inform the execution of RANSAC itself 
keypoint_detector = HessianDetector()
keypoint_detector_for_road = functools.partial(
    keypoint_detector.find_keypoints, percentile=68.05
)

RANSACDetector.fit_and_report(
    image=img_array,
    keypoint_detector_algorithm=keypoint_detector_for_road,
    image_name='Road',
    distance_threshold=3.0,
)
```

The resulting image looks like the following, `problem2_ransac/results.png`:

![visual of the output for plot](./problem2_ransac/results.png).

## Problem 3: Hough Transform

### Short Explanation
The `HoughTransformDetector` class implements the Hough Transform algorith, as discussed in class. Like the `RANSACDetector`, it has a `fit()` method that consumes the keypoints and finds multiple lines in the image. Non-max suppression is used here, so that for plotting we only look at lines that were maxima in local neighbors (but note, the non-suppressed accumulator array is used to plot the votes histogram, since that looked better visually). When plotting the lines themselves, I attempt to convert the polar coordinates back into Cartesian ones, and specifically find the x- and y-intercepts. One limitation of this approach is that sometimes, the line being plotted fell outside the dimensions of the image itself - so in the final images, it appears to not be there (when in reality, it just fell out of the "window" of the plot axes).

### Code
```python
# util/model_fitting.py
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

            # return the modified matrix
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
        sample_indices = list()
        for flat_index in least_to_greatest_votes:
            row_index = flat_index // local_max_accumulator.shape[1]
            col_index = flat_index - (row_index * local_max_accumulator.shape[1])
            sample_indices.append((row_index, col_index))
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
        print("=============== You just ran a Hough Transform - I'll do my best to plot the lines! ===================")
        # Create a figure with matching dimensions to the input image
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Plot the image and detected lines
        for index, (rho_bin, theta_bin) in enumerate(sample_indices):
            theta = theta_bin * theta_bin_size
            rho = rho_bin * rho_bin_size
            print(f"Line {index + 1} - rho and theta bin: ({rho_bin}, {theta_bin}) --> the line params, theta and rho, are: ({theta}, {rho})")
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

```

### Results

#### 3a. Baseline Bin Dimensions
To detect 4 lines in the `road.png` image using the Hough Transform, 
I executed the following code, in `problems.ipynb`:

```python

# as before, utilize a `partial` function to find the keypoints needed by the algorithm
keypoint_detector = HessianDetector()

keypoint_detector_for_road = functools.partial(
    keypoint_detector.find_keypoints, percentile=68.05
)

HoughTransformDetector.fit_and_report(
    image=np.array(original_image),
    image_name="Road",
    keypoint_detector_algorithm=keypoint_detector_for_road,
    # bin sizes
    rho_bin_size=2,
    theta_bin_size=np.pi / 180,
)
```

The resulting image looks like the following, `problem3_hough_transform/results_3a.png`:

![visual of the output for plot](./problem3_hough_transform/results_3a.png).

Unfortunately (for reasons I have not yet quite debugged), only 3 lines are plotted on the left-hand size. I believe 4 lines were successfully computed, as the following logs report the following (see the code cell in `problems.ipynb` to reproduce):

```
=============== You just ran a Hough Transform - I'll do my best to plot the lines! ===================
Line 1 - rho and theta bin: (0, 99) --> the line params, theta and rho, are: (1.7278759594743862, 0)
Line 2 - rho and theta bin: (108, 68) --> the line params, theta and rho, are: (1.1868238913561442, 216)
Line 3 - rho and theta bin: (0, 121) --> the line params, theta and rho, are: (2.111848394913139, 0)
Line 4 - rho and theta bin: (0, 119) --> the line params, theta and rho, are: (2.076941809873252, 0)
```

#### 3b. Half-as-Big Bin Dimensions
For this I executed the following code, in `problems.ipynb`:

```python

HoughTransformDetector.fit_and_report(
    image=img_array,
    image_name="Road",
    keypoint_detector_algorithm=keypoint_detector_for_road,
    # bin sizes
    rho_bin_size=1,
    theta_bin_size=np.pi / 360,
)

```

The resulting image looks like the following, `problem3_hough_transform/results_3b.png`:

![visual of the output for plot](./problem3_hough_transform/results_3b.png).

Unfortunately (for reasons I have not yet quite debugged), only 2 lines are plotted on the left-hand size. But like before, I believe 4 lines were successfully computed, as the following logs report the following (see the code cell in `problems.ipynb` to reproduce):

```
=============== You just ran a Hough Transform - I'll do my best to plot the lines! ===================
Line 1 - rho and theta bin: (229, 119) --> the line params, theta and rho, are: (1.038470904936626, 229)
Line 2 - rho and theta bin: (0, 230) --> the line params, theta and rho, are: (2.007128639793479, 0)
Line 3 - rho and theta bin: (0, 239) --> the line params, theta and rho, are: (2.0856684561332237, 0)
Line 4 - rho and theta bin: (0, 248) --> the line params, theta and rho, are: (2.1642082724729685, 0)
```

#### 3c. Twice-as-Big Bin Dimensions
For this I executed the following code, in `problems.ipynb`:

```python

HoughTransformDetector.fit_and_report(
    image=img_array,
    image_name="Road",
    keypoint_detector_algorithm=keypoint_detector_for_road,
    # bin sizes
    rho_bin_size=4,
    theta_bin_size=np.pi / 90,
)

```

The resulting image looks like the following, `problem3_hough_transform/results_3c.png`:

![visual of the output for plot](./problem3_hough_transform/results_3c.png).

Unfortunately (for reasons I have not yet quite debugged), only 3 lines are plotted on the left-hand size. But like before, I believe 4 lines were successfully computed, as the following logs report the following (see the code cell in `problems.ipynb` to reproduce):

```
=============== You just ran a Hough Transform - I'll do my best to plot the lines! ===================
Line 1 - rho and theta bin: (58, 29) --> the line params, theta and rho, are: (1.0122909661567112, 232)
Line 2 - rho and theta bin: (54, 34) --> the line params, theta and rho, are: (1.1868238913561442, 216)
Line 3 - rho and theta bin: (0, 47) --> the line params, theta and rho, are: (1.6406094968746698, 0)
Line 4 - rho and theta bin: (0, 59) --> the line params, theta and rho, are: (2.059488517353309, 0)
```






