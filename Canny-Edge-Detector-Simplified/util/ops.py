from typing import List


def convolve_matrices(
    matrix1: List[List[float]], matrix2: List[List[float]]
) -> float:
    '''asumes both matrices have the same, non-zero dimensions'''
    width, height = len(matrix1[0]), len(matrix1)
    
    product = 0

    for row_i in range(height):
       for col_i in range(width):
        product += (matrix1[row_i][col_i] * matrix2[row_i][col_i])

    return product

def apply_kernel(
        channel: List[List[float]],
        kernel: List[List[float]], 
        row_index: int, col_index: int
    ) -> float:
    """Applies the 2D kernel to 1 block of pixels on the image.
    
    Args:
        channel: 2D array - one of the channels of the input image
        kernel: 2D array representing the parameters to use
        row_index, col_index: int: the coordinates of the upper left corner
                            of the block of pixels being convolved

    Returns: float: the dot product of the kernel and the image pixels
    """
    # A: define useful vars
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    # B: get the block of pixels needed for the convolution
    block_of_pixels = channel[row_index:(kernel_h + row_index), col_index:(kernel_w + col_index)]
    # C: compute the convolution
    return convolve_matrices(kernel, block_of_pixels)


def slide_kernel_over_image(
    channel: List[List[float]],
    kernel: List[List[float]], 
    row_index: int, stride: int
) -> List[float]:
    """Applies the 2D kernel across the columns of 1 image channel.
    
    Args:
        channel: 2D array - one of the channels of the input image
        kernel: 2D array representing the parameters to use
        row_index, col_index: int: the coordinates of the upper left corner
                            of the block of pixels being convolved

    Returns: np.array: 1D array of the resulting values from performing 
                        the convolution at each "block" of pixels on the channel
    """
    # A: define useful vars + output
    _, kernel_w = len(kernel), len(kernel[0])
    conv_channel_row = list()
    # B: get the starting column
    starting_col_ndx = 0
    while starting_col_ndx < len(channel) - kernel_w:
        # compute the convolution
        conv_block_of_pixels = apply_kernel(channel, kernel, row_index, starting_col_ndx)
        # add it to the output
        conv_channel_row.append(conv_block_of_pixels)
        # move on to the next starting column, using the stride
        starting_col_ndx += stride
    return conv_channel_row


def convolve_2D(
    channel: List[List[float]],
    kernel: List[List[float]],
    stride: int
) -> List[List[float]]:
    """Performs a 2D convolution over 1 channel.

    Args:
        channel: 2D array - one of the channels of the input image
        filter: 2D array representing the parameters to use
        stride: int - using the same stride length for both directions

    Returns: np.array: the convolved channel
    """
    conv_channel = list()
    kernel_h, _ = len(kernel), len(kernel[0])
    # iterate over the rows and columns
    starting_row_ndx = 0
    while starting_row_ndx < len(channel) - kernel_h:
        # convolve the next row of this channel
        conv_channel_row = slide_kernel_over_image(channel, kernel, starting_row_ndx, stride)
        # now, add the convolved row to the list 
        conv_channel.append(conv_channel_row)
        # move to the next starting row for the convolutions
        starting_row_ndx += stride
    return conv_channel


def convolution(
    image: List[List[float]], 
    filter: List[List[float]], 
    stride=1,
    padding="SAME"
) -> List[List[float]]:
    """Performs a convolution on an input image.

    Assumptions:
        1. image is square and the size is an odd number
        2. filter is square and the size is an odd number.
        3. the filter is smaller than the image size

    Args:
        Image: 2D array - a grayscale raster image, aka a "pixel matrix"
        filter: 2D array representing the parameters to use
        stride: int - using the same stride length for both directions

    Returns: np.array: a new RGB image
    """
    # TODO[use same padding to ensure output matrix has same dims]
    convolved_channel = convolve_2D(image, filter, stride)
    return convolved_channel
  