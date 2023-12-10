# Homework 4: Image Segmentation and Pixel Classification

**Name: Syed Zain Raza**

## Problem 1: K-Means Segmentation

**Brief Description**

In this problem, I added a new module to my `util` package called `clustering`. It has a class called `KMeans`, which runs the clustering algorithm of the same name. 

Internally, this class keeps track of a Python dictionary called `centroid_coords` that maps each of the current `k` clusters (represented by a non-negative integer index) to a NumPy row vector representing the XY coordinates of that centroid (aka, a cluster center). It runs until convergence or until surpassed an upper bound called `max_iter`, which is specified by the user.

**Resulting Image(s)**

Note: although the right side of this plot got cutoff, the Z-axis represents the "blue" color channel of the image pixels.
![](./outputs/problem1/kmeans_plot.png)

## Problem 2: SLIC

**Brief Description**

In this problem, I added a new module to my `util` package called `superpixels`. It has a class called `SLIC`, which runs an approximation of the algorithm of the same name. 

This class only has 1 method called `execute_and_visualize()`. It first tries to divide the image into as many 50x50 blocks of pixels as it can. Then we initialize clusters at the position of lowest gradient magnitude (defined by taking the square root of the summed squares of gradient magnitudes across each of the RGB channels) in each of the blocks. Thirdly, we reuse much of the logic in the `KMeans` class to attempt clustering the pixels around what would be the best centroids in each block, in 5D space (i.e,, both the XY and RGB values are used). Finally, we random assign RGB colors to each centroid, and plot the pixels to be either black (if they are on the boundary between superpixels) or to be whatever color that at least ONE of the centroid (which they've been assigned to) is. 

Please note that although the implementation does run without errors, there may still be 1 or 2 logic errors in the code.  

**Resulting Image(s)**

![](./outputs/problem2/slic_for_white-tower.png)

![](./outputs/problem2/slic_for_wt_slic.png)

## Problem 3: Pixel Classification

**Brief Description**

In this problem, I added a new module to my `util` package called `classification`. It has a class called `PixelClassifier`. This class assumes that during the training phase, it will be able to see a set of "original" RGB images, and "masked" RGB images - in the latter, whichever pixels count as the "positive" class in the former (e.g., in our case, it is the pixels representing the blue sky) have been painted white. During the training phase, it then optimizes 2 different `KMeans` (see Problem 1) with `k = 10`. At the end, we finally stack the clusters from both objects vertically. During the prediction step, each new image will be labelled as positive or negative, based on whether it is closer (in the space of 5D XY-RGB points) to one of the first half of clusters (each belonging to the sky class) or the second half (in which case, it would be non-sky). Finally, the images are visualized with "sky" class pixels being painted bright yellow.

**Resulting Image(s)**
Unfortunately, it appears as though our classifier has learned to color any blue pixel as belonging to the sky class. This is technically overfitting, as with more time we could work to address it by perhaps finding a more descriptive vector space to use while clustering.

![](./outputs/problem3/classified_sky_test1.png)
![](./outputs/problem3/classified_sky_test2.png)
![](./outputs/problem3/classified_sky_test3.png)
![](./outputs/problem3/classified_sky_test4.png)