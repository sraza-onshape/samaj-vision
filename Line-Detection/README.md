# Homework 2: Line Detection

## Pseudocode

1. Use the Hessian detector to detect keypoints (see 4th set of notes)
    1. use zero padding
    1. Apply a Gaussian filter first
    1. use the Sobel filters as derivative operators
    1. Threshold the determinant of the Hessian
    1. apply non-maximum suppression in 3×3 neighborhoods

### Problem 3: Hough Transform - TODO


1. use the starter code from chatGPT - will handle the algo + plotting
    - main input params
        - `rho_resolution = 1`
        - `theta_resolution = np.pi / 180`  # Reduced resolution for theta (angle) 
        - keypoints
        - line_survival_threshold  # will just have to tune it so we can get at least 4
    
## Debugging Issues:
- instead of a threshold, use non-max suppression (check lecture) to find 4 local maximums in the `accumulator`?
- in the first plot - restrict the domain and range of the lines plotted --> try `# Set the plot dimensions to match the image dimensions
plt.gcf().set_size_inches(image.shape[1] / 80, image.shape[0] / 80)`
- add a plot of the 2D histogram
- ask Siyuan re: all the output pics I have