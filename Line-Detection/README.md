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
✅- instead of a threshold, use non-max suppression (check lecture) to find 4 local maximums in the `accumulator`?
- [TODO] in the first plot - restrict the domain and range of the lines plotted --> try `# Set the plot dimensions to match the image dimensions
plt.gcf().set_size_inches(image.shape[1] / 80, image.shape[0] / 80)`
indicies: `[50 29 71 14]`
# TODO: try another idea: based on the rho_bin, theta_bin, 1 at least 1 point on the line, that's also on the pic
--> given the `rho_bin` and the `theta_bin`:
    --> iterate over `detected_lines[0][sample_indices]` and `detected_lines[1][sample_indices]`:
        --> theta = theta_bin * theta_resolution
        --> rho = rho_bin * rho_resolution
        --> rho = int(x_coord * np.cos(theta) + y_coord * np.sin(theta))
        --> x_coord1 = (rho - (y_coord * np.sin(theta))) / (np.cos(theta))
        --> y_coord1 = (rho - (x_coord * np.cos(theta))) / (np.sin(theta))

        --> x_coord2 = x_coord1 + 10
        --> y_coord2 = (rho - (x_coord2 * np.cos(theta))) / (np.sin(theta))


- [TODO] :?>?
:add a plot of the 2D histogram
    - idea: use imshow, but do so on a plot with the same dims as the input image (so, each cell in the accumulator, will turn into a box of cells in the final plot image)
- ask Siyuan re: all the output pics I have