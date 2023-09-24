import numpy as np
from typing import List


class GaussianFilter:
    '''A 2D Gaussian filter to use for smoothening images (and edge detection?).'''
    
    def __init__(self, sigma: int = None) -> None:
        if sigma:
            self._set_parameters(sigma)
    
    def _set_parameters(self, sigma: int):
        self.sigma = sigma
        self.filter_width = 6 * sigma + 1

    def _get_element_in_filter(self, x: int, y: int) -> float:
        '''Samples from a 2D Gaussian to determine what value goes in a given element.'''
        exponent = -1 * (((x ** 2) + (y ** 2)) / (2 * (self.sigma ** 2)))
        power = np.exp(exponent)  # Euler's number (e) is the base
        return power / (2 * np.pi * (self.sigma ** 2))

    def create_gaussian_filter(self, sigma: int) -> List[List[float]]:
        '''create 2D array for for the filter'''
        self._set_parameters(sigma)
        center_pos = 3 * self.sigma  # in both axes
        filter = list()

        for x_index in range(self.filter_width):
            filter_row = list()
            for y_index in range(self.filter_width):
                # coordinates are "centered" at the center element in the matrix
                x_coord, y_coord = abs(x_index - center_pos), abs(y_index - center_pos)
                filter_row.append(self._get_element_in_filter(x_coord, y_coord))
            filter.append(filter_row)
        
        return filter
    

if __name__ == "__main__":
    # test out the functions, ensure the coefs sum to 1
    sigma = 1
    filter = GaussianFilter()
    matrix = filter.create_gaussian_filter(sigma)
    print(f"Sum of values: {sum([sum(row) for row in matrix])}")
    print(f"The filter itself: {np.array(matrix)}")  # using NumPy soley for printability
