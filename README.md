# CS 558 Computer Vision

Note to Self: [How to Add/Delete Kernels from Jupyter Notebook](https://janakiev.com/blog/jupyter-virtual-envs/).

```bash
$ python -m ipykernel install --name=env
```

Really useful equation for finding the dimensions of a feature map outputted by a convolution - inspired the formula I used to compute the `padding_distance` in `ops.py`: [Stanford CS 231](https://cs231n.github.io/convolutional-networks/)


# CS 532 HW 1 Problem Debugging 

## P1 - Sources of error
✅ - where we compute the destination points - probably can still tune the margin lengths?
✅- how we compute the homography matrix - take out normalization for now
- how we compute the image - debug the generate image
    - anyway to get around having to cast the pixel values as ints?

