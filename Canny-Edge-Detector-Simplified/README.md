# Homework 1: The Canny Edge Detector

## Student Info
- **Name**: Syed Zain Raza
- **Campus-wide ID**: 20011917

## Quickstart 

Please ensure you have at least Python `3.7`:`
```bash
$ python3 -m venv env  # Environment Setup - virtualenv
$ source env/bin/activate
$ python -m pip install -r requirements.txt --upgrade pip
$ python canny_edge_detector.py  # main function, defaults args
$ python canny_edge_detector.py  --operation smooth --data <path_to_image>  # Gaussian filtering only
$ python canny_edge_detector.py  --operation detect_edges --data <path_to_image>  # Edge Detection
```
Note: with the defaults, the `python canny_edge_detector.py` will show edge detection on the "Plane" scene (with non-maximum suppression). Use `python canny_edge_detector.py -h` to learn more about the arguments you can pass to this script.

## Where to Find Stuff

1. **Code**: 4 main points of interest
    1. `util/ops.py`: reading the images and implements a custom 2D convolution function.
    1. `util/gaussian_base.py`: Gaussian filtering of images is located.
    1. `util/gaussian_derivative.py`: computing the image gradient, and using that for detecting the edges (along with non-maximum suppression).
    1. `problem1.ipynb`: Please see the code to see how their APIs are meant to work together, and reproduce the output images.

1. **Images**:
    1. `original_images_cs558_hw1/`: provided images for this assignment
    1. `part_1_smoothed_images/`: output images from Gaussian filtering
    1. `part_2_image_edges`: outputs for part 2 .
    1. `part_3_non_max_suppression`: outputs for part 3.


## Limitations

- main function only allows you to pass in a single file path at a time 
- the code is SLOW 

