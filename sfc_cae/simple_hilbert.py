"""
This module contains simple implementation of Hilbert Curve generation on (2^n * 2^n) structured grids.
Author: Jin Yu
Github handle: acse-jy220
"""

import numpy as np
import matplotlib.pyplot as plt

def rotation(grid_in, pattern = 1):
    '''
    Some useful rotating/flipping operation for hilbert_space_filling_curve()
    '''
    if pattern == 1:
       return grid_in.T
    elif pattern == 2:
       return np.flipud(np.fliplr(grid_in))


def hilbert_space_filling_curve(num = 4, ver_bose = False, ver_bose_contour=False):
    '''
    This function generates Hilbert space-filling curve on a (2^n * 2^n) by recursive.

    ver_bose:

    Input:
    ---
    num: [int] the size of the square grid.
    ver_bose: [bool] If true, return a plot, view the space-filling curve directly.
    ver_bose_contour: [bool]  If true, return a contour plot of the indexing of the space-filling curve directly.

    Output:
    ---
    np.argsort(grid_1.flatten()): [1d-array] of shape (2^(2n), ) the flattened 1d-array of the Hilbert Space-filling curve on a structured grid.

    '''
    
    n_level = np.log2(num)
    if n_level // 1 != n_level:
        raise ValueError("square grid size not a exponential of 2!")
    elif n_level == 1:
        raise ValueError("grid size should be larger or equal than 4!")
    
    n_level = n_level.astype('int')

    # start from 2 * 2 grid
    grid_1 = np.array([0, 3, 1, 2]).reshape(2, 2)
    subsize = 4  

    for i in range(2, n_level + 1):

        grid_2 = rotation(grid_1, pattern = 1) + subsize

        grid_3 = rotation(grid_1, pattern = 1) + subsize * 2

        grid_4 = rotation(grid_1, pattern = 2) + subsize * 3      

        grid_1 = np.vstack((np.hstack((grid_1, grid_2)), np.hstack((grid_4, grid_3))))

        grid_1 = rotation(grid_1, pattern = 1)

        subsize *= 4

    if ver_bose:
        
        x_coords = np.argsort(grid_1.flatten()) // num 
        y_coords = np.argsort(grid_1.flatten()) % num 

        fig, ax = plt.subplots(figsize=(20,20))
        ax.set_title("Hilbert space-filling curve on a %d * %d square grid" % (num, num), fontsize = 25)

        ax.plot(x_coords, y_coords, color = "black")

        plt.show()

    if ver_bose_contour:

        fig, ax = plt.subplots(figsize=(20,20))
        ax.set_title("Hilbert space-filling curve on a %d * %d square grid" % (num, num), fontsize = 25)
        xx, yy = np.meshgrid(np.arange(0, num), np.arange(0, num))
        plot_levels = [i * num // 2 - 0.5 for i in range(1, num * 2 - 1)]
        plot_levels.insert(0, 0)
        plot_levels.append(num ** 2)

        cset = plt.contourf(xx, yy, grid_1.T, levels=plot_levels, cmap=None)
        fig.colorbar(cset, shrink=0.5, aspect=5)

        plt.show()
       
    return np.argsort(grid_1.flatten())