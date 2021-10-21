"""
This module contains simple implementation of Hilbert Curve generation on 2D/3D (2^n * 2^n (* 2^n)) structured grids.
Author: Jin Yu
Github handle: acse-jy220
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

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


########################################## rotation function for 3D ###################################################
def rotate_3d_xy_90(array, size, nrot = 1, reverse=False):
    array_3d = np.rot90(copy.deepcopy(array), k = nrot, axes = (0, 1))
    if reverse: array_3d = array_3d[::-1]
    return array_3d

def rotate_3d_xz_90(array, size, nrot = 1, reverse=False):
    array_3d = np.rot90(copy.deepcopy(array), k = nrot, axes = (0, 2))
    if reverse: array_3d = array_3d[::-1]
    return array_3d

def rotate_3d_yz_90(array, size, nrot = 1, reverse=False):
    array_3d = np.rot90(copy.deepcopy(array), k = nrot, axes = (1, 2))
    if reverse: array_3d = array_3d[::-1]
    return array_3d

def rotate_3d(array, size, zdir=(0, 1, 2), reverse=False):
    array_3d = np.transpose(copy.deepcopy(array), axes=zdir)
    if reverse: array_3d = array_3d[::-1]
    return array_3d


########################################## Hilbert Curve for 3D ###################################################
def hilbert_space_filling_curve_3d(num = 2):
    '''
    This function generates Hilbert space-filling curve in 3D on a (2^n * 2^n * 2^n) by recursive.

    ver_bose:

    Input:
    ---
    num: [int] the size of the square grid.

    Output:
    ---
    np.argsort(grid_1.flatten()): [1d-array] of shape (2^(2n), ) the flattened 1d-array of the Hilbert Space-filling curve on a 3D structured grid.
    '''

    n_level = np.log2(num)
    if n_level // 1 != n_level:
        raise ValueError("square grid size not a exponential of 2!")
    
    n_level = n_level.astype('int')

    # phase_1 3d Hilbert Curve, size of 2**3 = 8
    size = 2
    grid_sample = np.argsort(np.array([0, 2, 3, 1, 5, 7, 6, 4])).reshape(size, size, size)
    for i in range(2, n_level + 1):
         
        # grid_1
        grid_1 = rotate_3d_xy_90(grid_sample, size = size, nrot = 1, reverse=True)
        # print(grid_1)
        # grid_2 and 3
        grid_2 = rotate_3d(grid_sample, size = size, zdir = (2, 1, 0)) + size ** 3
        merge_1 = np.concatenate((grid_1, grid_2), axis = -2)
        # print(merge_1)
        grid_3 = rotate_3d(grid_sample, size = size, zdir = (2, 1, 0)) + 2 * size ** 3
        # grid_4 and 5
        grid_4 = rotate_3d_yz_90(grid_sample, size = size, nrot = 2, reverse=False) + 3 * size ** 3
        merge_2 = np.concatenate((grid_4, grid_3), axis = -2)
        # print(merge_2)
        half_1 = np.concatenate((merge_1, merge_2), axis = -1)
        # print(half_1)
        grid_5 = rotate_3d_yz_90(grid_sample, size = size, nrot = 2, reverse=False) + 4 * size ** 3
        # print(grid_5)
        # grid_6 and 7
        grid_6 = rotate_3d_xz_90(grid_sample, size = 2, nrot = 3, reverse=True) + 5 * size ** 3 
        merge_4 = np.concatenate((grid_5, grid_6), axis = -2)
        # print(merge_4)
        grid_7 = rotate_3d_xz_90(grid_sample, size = 2, nrot = 3, reverse=True) + 6 * size ** 3 
        # grid_8
        grid_8 = rotate_3d_xy_90(grid_sample, size = size, nrot = 3, reverse=True) + 7 * size ** 3
        merge_5 = np.concatenate((grid_8, grid_7), axis = -2)
        # print(merge_5)
        half_2 = np.concatenate((merge_5, merge_4), axis = -1)
        # print(half_2)
        
        # merge 8 grids to form a big grid of next iter.
        grid_sample =np.concatenate((half_1, half_2), axis = 0)
        # print(grid_sample)
        
        # increase the size
        size *= 2 

    return np.argsort(grid_sample.flatten())

