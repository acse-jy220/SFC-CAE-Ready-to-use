"""
This module contains extra functions for training.py/ sfc_cae.py as supplement.
Author: Jin Yu
Github handle: acse-jy220
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import space_filling_decomp_new as sfc
import sys
import vtk
import vtktools
import numpy as np
import time
import glob
import progressbar
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
import meshio
import re

# permutation lib
from itertools import permutations
import itertools

# create an animation
from matplotlib import animation
from IPython.display import HTML
# custom colormap
import cmocean

import copy

import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, TensorDataset, Dataset


#################################################### Functions for data pre-processing / data loading ######################################################################

def read_in_files(data_path, file_format='vtu', vtu_fields=None, write_out = False):
    '''
    This function reads in the vtu/txt files in a {data_path} as tensors, of shape [snapshots, number of Nodes, Channels]

    Input:
    ---
    data_path: [string] the data_path which holds vtu/txt files, no other type of files are accepted!!!
    file_format: [string] 'vtu' or 'txt', the format of the file.
    vtu_fields: [list] the list of vtu_fields if read in vtu files, the last dimension of the tensor, e.g. ['Velocity', 'Pressure'].
    write_out: [bool] whether write out those readed-in fields as indenpendent tensors, used for `MyTensorDataset` Class.

    Output:
    ---
    Case 1 - file_format='vtu': (3-tuple) [torch.FloatTensor] full_stage over times step, time along 0 axis; [torch.FloatTensor] coords of the mesh; [dictionary] cell_dict of the mesh.

    Case 2 - file_format='txt': [torch.FloatTensor] full_stage over times step, time along 0 axis

    '''
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[-2].split('_')
    file_prefix.pop(-1)
    if len(file_prefix) != 1: file_prefix = '_'.join(file_prefix) + "_"
    else: file_prefix = file_prefix[0] + "_"
    file_format = '.' + file_format
    print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
    cnt_progress = 0
    if (file_format == ".vtu"):
        print("Read in vtu Data......\n")
        bar=progressbar.ProgressBar(maxval=num_data)
        bar.start()
        data = []
        coords = None
        cells = None
        start = 0
        while(True):
            if not os.path.exists(F'{file_prefix}%d{file_format}' % start):
                print(F'{file_prefix}%d{file_format} not exist, starting number switch to {file_prefix}%d{file_format}' % (start, start+1))
                start += 1
            else: break
        for i in range(start, num_data + start):
            data.append([])
            vtu_file = meshio.read(F'{file_prefix}%d{file_format}' % i)
            if not (coords == vtu_file.points).all():
               coords = vtu_file.points
               cells = vtu_file.cells_dict
               print('mesh adapted at snapshot %d' % i)
            for j in range(len(vtu_fields)):
                vtu_field = vtu_fields[j]
                if not vtu_field in vtu_file.point_data.keys():
                   raise ValueError(F'{vtu_field} not avaliable in {vtu_file.point_data.keys()} for {file_prefix} %d {file_format}' % i)
                field = vtu_file.point_data[vtu_field]
                if j == 0:
                   if field.ndim == 1: field = field.reshape(field.shape[0], 1)
                   data[i - start] = field
                else:
                   if field.ndim == 1: field = field.reshape(field.shape[0], 1)
                   data[i - start] = np.hstack((data[i - start], field))
            cnt_progress +=1
            bar.update(cnt_progress)
        bar.finish()
        whole_data = torch.from_numpy(np.array(data)).float()
        
        # get rid of zero components
        zero_compos = 0
        for i in range(whole_data.shape[-1]):
            if whole_data[..., i].max() - whole_data[..., i].min() < 1e-8:
               zero_compos += 1
               whole_data[..., i:-1] = whole_data[..., i + 1:]
        if zero_compos > 0 : whole_data = whole_data[..., :-zero_compos]

        if write_out:
           print("\nWriting Tensors......\n")
           bar=progressbar.ProgressBar(maxval=num_data)
           bar.start()
           cnt = 0
           for i in range(whole_data.shape[0]):
               torch.save(whole_data[i, :].clone(), 'tensor_%d.pt'%i)
               cnt += 1
               bar.update(cnt)
           bar.finish()
        
        return whole_data, coords, cells    

    elif (file_format == ".txt" or file_format == ".dat"):
        print("Read in txt/dat Data......")
        bar=progressbar.ProgressBar(maxval=num_data)
        data = []
        for i in range(num_data):
            data[i] = torch.from_numpy(np.loadtxt('{file_prefix} %d {file_format}' % i)).float()
            cnt_progress +=1
            bar.update(cnt_progress)
        bar.finish()
        return torch.cat(data, -1)

def get_simulation_index(num, simulation):
    '''
    This function returns the indexes for a square grid simulation that implemented in advection_block_analytical.py.

    Input:
    ---
    num: [int] the number of the simulation.
    simulation: [int] the run_simulation_advection class object defined in advection_block_analytical.py

    Output:
    ---
    indexes: [1d-array] the indexes for a certain simulation
    '''
    return np.arange(num * (simulation.steps + 1), (num + 1) * (simulation.steps + 1))

def read_parameters(setting_file = 'parameters.ini'):
    '''
    This function reads all the parameter settings in a setting file 'parameters.ini', interact with command_train.py, used for command line training on HPC.
    setting_file
    '''
    f = open(setting_file, 'r')
    lines = f.readlines()
    # create a dicitionary to store the parameters
    list_p = {}
 
    for line in lines[1:]:
        line = line.strip('\n')
        ss = re.split('=', line)
        list_p[ss[0].strip()] = ss[-1].strip()
    f.close()
    
    return list_p  

def normalize_tensor(tensor):
    '''
    This function normalize a torch.tensor with the operation channel-wisely. x = (x - mu) / sigma, where mu is the mean, sigma is std.
    
    Input: 
    ---
    tensor: [torch.FloatTensor] tensor input, last dimension represents channel.

    Output:
    ---
    3-tuple: [torch.FloatTensor] normalised tensor, [torch.FloatTensor] mean for each channel, [torch.FloatTensor] variance for each channel.

    '''
    if tensor.ndim > 2:
       t_mean = torch.zeros(tensor.shape[-1])
       t_std = torch.zeros(tensor.shape[-1])
       for i in range(tensor.shape[-1]):
          t_mean[i] = tensor[..., i].mean()
          t_std[i] = tensor[..., i].std()
          tensor[...,i] -= t_mean[i]
          tensor[...,i] /= t_std[i]
       return tensor, t_mean, t_std
    else:
        t_mean = torch.mean(tensor)
        t_std = torch.std(tensor)
        return (tensor - t_mean)/t_std, t_mean, t_std

def standardlize_tensor(tensor, lower = -1, upper = 1):
    '''
    This function maps a torch.tensor to a interval [lower, upper] channel-wisely.
    
    Input: 
    ---
    tensor: [torch.FloatTensor] tensor input, last dimension represents channel.

    Output:
    ---
    3-tuple: [torch.FloatTensor] standardlized tensor, [torch.FloatTensor] tk for each channel, [torch.FloatTensor] tb for each channel.
    where standardlized tensor is belong to [lower, upper] for each channel

    '''
    if tensor.ndim > 2:
       tk = torch.zeros(tensor.shape[-1])
       tb = torch.zeros(tensor.shape[-1])
       for i in range(tensor.shape[-1]):
          tk[i] = (upper - lower) /(tensor[..., i].max() - tensor[..., i].min())
          tb[i] = (tensor[..., i].max() * lower - tensor[..., i].min() * upper) /(tensor[..., i].max() - tensor[..., i].min())
          tensor[...,i] *= tk[i]
          tensor[...,i] += tb[i]
       return tensor, tk, tb
    else:
        tk = (upper - lower) / (tensor.max() - tensor.min())
        tb = (tensor.max() * lower - tensor.min() * upper) / (tensor.max() - tensor.min())
        return tensor * tk + tb, tk, tb

def denormalize_tensor(tensor, t_mean, t_std):
    '''
    This function denormalize a tensor from normalisation channel-wisely.

    Input:
    ---
    tensor:  [torch.FloatTensor] tensor input, last dimension represents channel.
    t_mean: [torch.FloatTensor] the mean value for each channel, got from function normalize_tensor()
    t_std: [torch.FloatTensor] the variance value for each channel, got from function normalize_tensor()

    Output:
    ---
    tensor: [torch.FloatTensor] denormalised tensor
    '''
    if tensor.ndim > 2:
       for i in range(tensor.shape[-1]):
           tensor[...,i] *= t_std[i]
           tensor[...,i] += t_mean[i]
       else:
          tensor *= t_std
          tensor += t_mean
    return tensor

def destandardlize_tensor(tensor, tk, tb):
    '''
    This function destandardlize a tensor from standardlisation channel-wisely.

    Input:
    ---
    tensor:  [torch.FloatTensor] tensor input, last dimension represents channel.
    tk: [torch.FloatTensor] the mean value for each channel, got from function standardlize_tensor()
    tb: [torch.FloatTensor] the variance value for each channel, got from function standardlize_tensor()

    Output:
    ---
    tensor: [torch.FloatTensor] destandardlized tensor
    '''
    if tensor.ndim > 2:
       for i in range(tensor.shape[-1]):
           tensor[...,i] -= tb[i]
           tensor[...,i] /= tk[i]
    else:
        tensor -= tb
        tensor /= tk
    return tensor

def get_path_data(data_path, indexes):
    '''
    This function would return a path list for data with a arbitary indice.

    Input:
    ---
    data_path: [string] the path for the data, vtu or txt files.
    indexes: [1d-array] the indice we want to select for the data.

    Output:
    ---
    path_list: [list of strings] the path list of corresponding data, used for np.loadtxt()/ meshio.read()
    '''
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[:-1]
    file_prefix = ''.join(file_prefix)
    file_prefix = file_prefix.split('_')[:-1]
    file_prefix = ''.join(file_prefix) + '_'
    file_format = data[0].split('.')[-1]
    file_format = '.' + file_format
    path_data = []
    for i in range(len(indexes)):
        path_data.append(F'{file_prefix}%d{file_format}' % indexes[i])
    return path_data



class MyTensorDataset(Dataset):
    '''
    This class defines a custom dataset used for command line training, covert all your data to .pt files snapshot by snapshot before using it.

    ___init__:
       Input:
       ---
       path_dataset: [string] the data where holds the .pt files
       lower: [float] the lower bound for standardlisation
       upper: [float] the upper bound for standardlisation
       tk: [torch.FloatTensor] pre-load tk numbers, if we have got it for the dataset, default is None.
       tb: [torch.FloatTensor] pre-load tb numbers, if we have got it for the dataset, default is None.
       set_bound: [1d-array of list] of shape (2,) used for volume_fraction for slugflow dataset, bound [0, 1]
    
    __getitem__(i):
       Returns on call:
       ---
       self.dataset[i]: a single snapshot after standardlisation.

    __len__:
       Returns on call:
       ---
       len: [int] the length of dataset, equal number of time steps/ snapshots


    '''
    def __init__(self, path_dataset, lower, upper, tk = None, tb = None, set_bound = False):
        self.dataset = path_dataset
        self.length = len(path_dataset)
        self.bounded = set_bound
        t_max = torch.load(self.dataset[0]).max(0).values.unsqueeze(0)
        t_min = torch.load(self.dataset[0]).min(0).values.unsqueeze(0)
        cnt_progress = 0

        # find tk and tb for the dataset.
        if tk is None or tb is None:
            print("Computing min and max......\n")
            bar=progressbar.ProgressBar(maxval=self.length)
            bar.start()
            for i in range(1, self.length):
              t_max = torch.cat((t_max, torch.load(self.dataset[i]).max(0).values.unsqueeze(0)), 0)
              t_min = torch.cat((t_min, torch.load(self.dataset[i]).min(0).values.unsqueeze(0)), 0)
              cnt_progress +=1
              bar.update(cnt_progress)
            bar.finish()
            self.t_max = t_max.max(0).values
            self.t_min = t_min.min(0).values
            self.tk = (upper - lower) / (self.t_max - self.t_min)
            self.tb = (self.t_max * lower - self.t_min * upper) / (self.t_max - self.t_min)
        else: # jump that process, if we have got tk and tb.
            self.tk = tk
            self.tb = tb
        print('tk: ', self.tk, '\n')
        print('tb: ', self.tb, '\n')

    def __getitem__(self, index):
        tensor = torch.load(self.dataset[index])
        tensor = (tensor * self.tk + self.tb).float()
        if self.bounded: 
           tensor[..., 0][tensor[..., 0] > 1] = 1
           tensor[..., 0][tensor[..., 0] < 0] = 0
        return tensor
      
    def __len__(self):
        return self.length

####################################################  Plotting functions for unstructured mesh ######################################################################      

def plot_path_grid_cube(size, ordering, point_color = 'red', line_color = 'blue', show_blocks = True, mark_numbers = True, linewidth = 5, levels = None):    
    '''
    This function will generate path plot (with block background) of some sfcs in 3D cube grids, as well as show the node numbering.

    Input:
    ---
    size: [int] the length of the cube.
    ordering: [1d-array] the ordering of the grids in the cube, length stricted equal to size ^ 3.
    point_color: [str] the color of the Nodes.
    line_color: [str] the color of the lines.
    show_blocks: [bool] whether the grids are showed in the plot.
    mark_numbers: [bool] whether the numbering of the grids are annotated in the plot.
    linewidth: [float] the width of the SFC.
    levels: [int or NoneType] whether color of different levels (w.r.t ordering) are distinguished.

    Output:
    ---
    A 3D plot.
    '''
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    mesh = np.arange(0, size + 1)
    mesh = (mesh[:-1] + mesh[1:]) / 2
    x, y, z = np.meshgrid(mesh, mesh, mesh, indexing='ij')

    # Create axis
    axes = [size] * 3
  
    # Create Data
    data = np.ones(axes, dtype=np.bool)
  
    # Controll Tranperency
    alpha = 0.2
    alpha_2 = 0.2
  
    # Control colour
    colors = np.empty(axes + [4], dtype=np.float32)
    edges = np.empty(axes + [4], dtype=np.float32)
    colors = [1, 1, 1, alpha]
    edges = [0, 0, 0, alpha_2]
  
    # Voxels is used
    if show_blocks: ax.voxels(data, facecolors=colors, edgecolors=edges)

    # plot ordering in 3d
    x = x.flatten()[ordering]
    y = y.flatten()[ordering]
    z = z.flatten()[ordering]
    if levels is None: 
       ax.plot(x, y, z, color = line_color, linewidth = linewidth)
       ax.scatter(x, y, z, c = point_color)
    else: 
        cuts = np.linspace(0, size ** 3, levels + 1).astype(np.int32)
        for i in range(levels): 
            start = cuts[i]
            end = cuts[i + 1]
            if i != levels - 1: end += 1
            ax.plot(x[cuts[i]:cuts[i+1]], y[cuts[i]:cuts[i+1]], z[cuts[i]:cuts[i+1]], '-', linewidth = linewidth)
            ax.scatter(x[cuts[i]:cuts[i+1]], y[cuts[i]:cuts[i+1]], z[cuts[i]:cuts[i+1]], '-')

    labs = (np.arange(size ** 3) + 1).tolist()

    #use for loop to add annotations to each point in plot
    if mark_numbers: 
      for i, txt in enumerate(labs):
        disp = [0.05, 0.02, 0.02]
        ax.text(x[i] + disp[0], y[i] + disp[1], z[i] + disp[2], txt, None, color = point_color, fontfamily = 'sans-serif', fontsize=15, fontweight='bold')

    ax.axis('off')
    plt.show()

def get_sfc_curves_from_coords(coords, num):
    '''
    This functions generate space-filling orderings for a coordinate input of a Discontinuous Galerkin unstructured mesh.

    Input:
    ---
    coords: [2d-array] coordinates of mesh, read from meshio.read().points or vtktools.vtu().GetLocations(),  of shape(number of Nodes, 3)
    num: [int] the number of (orthogonal) space-filling curves you want.

    Output:
    ---
    curve_lists: [list of 1d-arrays] the list of space-filling curves, each element of shape [number of Nodes, ]
    inv_lists: [list of 1d-arrays] the list of inverse space-filling curves, each element of shape [number of Nodes, ]
    '''
    findm, colm, ncolm = sfc.form_spare_matric_from_pts(coords, coords.shape[0])
    colm = colm[:ncolm]
    curve_lists = []
    inv_lists = []
    ncurve = num
    graph_trim = -10  # has always been set at -10
    starting_node = 0 # =0 do not specifiy a starting node, otherwise, specify the starting node
    whichd, space_filling_curve_numbering = sfc.ncurve_python_subdomain_space_filling_curve(colm, findm, starting_node, graph_trim, ncurve, coords.shape[0], ncolm)
    for i in range(space_filling_curve_numbering.shape[-1]):
        curve_lists.append(np.argsort(space_filling_curve_numbering[:,i]))
        inv_lists.append(np.argsort(np.argsort(space_filling_curve_numbering[:,i])))

    return curve_lists, inv_lists

def get_sfc_curves_from_coords_CG(coords, ncurves, template_vtu):
    '''
    get inspiration from Claire's Code, this functions generate space-filling orderings for a coordinate input of a Continuous Galerkin unstructured mesh.

    Input:
    ---
    coords: [2d-array] coordinates of mesh, read from meshio.read().points or vtktools.vtu().GetLocations(),  of shape(number of Nodes, 3)
    num: [int] the number of (orthogonal) space-filling curves you want.
    template_vtu: [vtu file] a template vtu file, use for reading Node-connectivities.

    Output:
    ---
    curve_lists: [list of 1d-arrays] the list of space-filling curves, each element of shape [number of Nodes, ]
    inv_lists: [list of 1d-arrays] the list of inverse space-filling curves, each element of shape [number of Nodes, ]
    '''
    ncolm=0
    colm=[]
    findm=[0]
    for nod in range(coords.shape[0]):
        nodes = template_vtu.GetPointPoints(nod)
        nodes2 = np.sort(nodes) #sort_assed(nodes) 
        colm.extend(nodes2[:]) 
        nlength = nodes2.shape[0]
        ncolm=ncolm+nlength
        findm.append(ncolm)

    colm = np.array(colm)
    colm = colm + 1
    findm = np.array(findm)
    findm = findm + 1

    curve_lists = []
    inv_lists = []
    graph_trim = -10  # has always been set at -10
    starting_node = 0 # =0 do not specifiy a starting node, otherwise, specify the starting node
    whichd, space_filling_curve_numbering = sfc.ncurve_python_subdomain_space_filling_curve(colm, findm, starting_node, graph_trim, ncurves, coords.shape[0], ncolm)
    for i in range(space_filling_curve_numbering.shape[-1]):
       curve_lists.append(np.argsort(space_filling_curve_numbering[:,i]))
       inv_lists.append(np.argsort(np.argsort(space_filling_curve_numbering[:,i])))

    return curve_lists, inv_lists

def plot_trace_vtu_2D(coords, levels, save = False, width = None):
    '''
    This function plots the node connection of a 2D unstructured mesh based on a coordinate sequence.

    Input:
    ---
    coords: [2d-array] of shape(number of Nodes, 2/3), suggest to combine it with space-filling orderings, e.g. coords[sfc_ordering].
    levels: [int] the levels of colormap for the plot.

    Output:
    ---
    NoneType: the plot.
    '''
    x_left = coords[:, 0].min()
    x_right = coords[:, 0].max()
    y_bottom = coords[:, 1].min()
    y_top = coords[:, 1].max()
    y_scale = (y_top - y_bottom) / (x_right - x_left)
    fig, ax = plt.subplots(figsize=(40, 40 * y_scale))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    cuts = np.linspace(0, coords.shape[0], levels + 1).astype(np.int32)
    for i in range(levels):
        start = cuts[i]
        end = cuts[i + 1]
        if i != levels - 1: end += 1
        if width is not None: ax.plot(coords[cuts[i]:cuts[i+1], 0], coords[cuts[i]:cuts[i+1], 1], '-', linewidth = width)
        else: ax.plot(coords[cuts[i]:cuts[i+1], 0], coords[cuts[i]:cuts[i+1], 1], '-')
    plt.axis('off')
    if save: plt.savefig('curve_vtu_fields_2D.png', dpi = 200)
    else:
      plt.show()

def countour_plot_vtu_2D(coords, levels, mask=True, values=None, cmap = None, save = False):
    '''
    This function plots the contour of a 2D unstructured mesh based on a coordinate sequence.

    Input:
    ---
    coords: [2d-array] of shape(number of Nodes, 2/3), suggest to combine it with space-filling orderings, e.g. coords[sfc_ordering].
    levels: [int] the levels of colormap for the plot.
    mask: [bool] mask the cylinder, only turn it on for the 'Flow Past Cylinder' Case.
    Values: [1d-array] of shape(number of Nodes, ), default is Node indexing, suggest you will use this for plotting scalar field? Not suggested, too slow.
    cmap: [camp object] a custom cmap like 'cmocean.cm.ice' or an official colormap like 'inferno'.

    Output:
    ---
    NoneType: the plot.
    '''
    x = coords[:, 0]
    y = coords[:, 1]
    x_left = x.min()
    x_right = x.max()
    y_bottom = y.min()
    y_top = y.max()
    y_scale = (y_top - y_bottom) / (x_right - x_left)
    fig, ax = plt.subplots(figsize=(40, 40 * y_scale))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    
    triang = tri.Triangulation(x, y)

    if values == None:
        values=np.arange(coords.shape[0])
    
    if mask:
       min_radius = 0.05
       # Mask off unwanted triangles for the FPC case.
       xmid = x[triang.triangles].mean(axis=1)
       ymid = y[triang.triangles].mean(axis=1)
       mask = np.where((xmid - 0.2)**2 + (ymid - 0.2)**2 <= min_radius*min_radius, 1, 0)
       triang.set_mask(mask)
    
    plt.tricontourf(triang, values, levels = levels, cmap = cmap)    
    plt.axis('off')
    if save: plt.savefig('contour_vtu_fields_2D.png', dpi = 250)
    else:
      plt.show()  

class anim_vtu_fields_2D():
    '''
    This class is implemented to generated to animate the fields on a 2D FPC case, but abandoned at last because of slow speed.

    __init__:
      Input:
      ---
      coords: [2d-array] of shape(number of Nodes, 2/3), suggest to combine it with space-filling orderings, e.g. coords[sfc_ordering].
      levels: [int] the levels of colormap for the plot.
      Values: [1d-array] of shape(number of Nodes, ), default is Node indexing.
      cmap: [camp object] a custom cmap like 'cmocean.cm.ice' or an official colormap like 'inferno'.
      steps: [int] the number of time levels/ snapshots for the simulation.

    __update_grid__(step):
      Updates the animation.

    __generate_anime__:
      Returns:
      ---
      A matplotlib.animation Object.
    '''
    def __init__(self, coords, values=None, levels = 15, cmap = None, steps = None, min_radius = 0.05, mask_x = 0.2, mask_y = 0.2):
       # initialize location of mesh
       self.x = coords[:, 0]
       self.y = coords[:, 1]
       self.x_left = self.x.min()
       self.x_right = self.x.max()
       self.y_bottom = self.y.min()
       self.y_top = self.y.max()
       self.fig, self.ax = plt.subplots(figsize=(40,8))
       self.ax.set_xlim(self.x_left, self.x_right)
       self.ax.set_ylim(self.y_bottom, self.y_top)
       self.triang = tri.Triangulation(self.x, self.y)
       self.cmap = cmap
       self.levels = levels
       self.values = np.array(values)

       if steps is None: 
           self.steps = self.values.shape[0]
       else: self.steps = steps
    
       self.min_radius = min_radius
       self.mask_x = mask_x
       self.mask_y = mask_y
       # Mask off unwanted triangles.
       xmid = self.x[self.triang.triangles].mean(axis=1)
       ymid = self.y[self.triang.triangles].mean(axis=1)
       mask = np.where((xmid - self.mask_x)**2 + (ymid - self.mask_y)**2 <= self.min_radius**2, 1, 0)
       self.triang.set_mask(mask)
       self.image = self.ax.tricontourf(self.triang, self.values[0], levels = self.levels, cmap = self.cmap)    
       self.ax.axis('off')

    def update_grid(self, n_step: int):
       self.image = self.ax.tricontourf(self.triang, self.values[n_step], levels = self.levels, cmap = self.cmap)
       print('frame %d saved.' % n_step)
       return self.image,
   
    def generate_anime(self):
       return animation.FuncAnimation(self.fig, self.update_grid, frames = np.arange(0, self.steps))


#################################################### Extension functions for SFC_CAE module ######################################################################

def find_plus_neigh(ordering):
    '''
    This function returns the upper neighbour for a sfc ordering, see thesis

    Input:
    ---
    ordering: [1d-array] the (sfc) ordering of the Nodes.
    
    Return:
    ---
    plus_neigh: [1d-array] the upper-neighbour ordering.
    '''
    plus_neigh = np.zeros_like(ordering)
    plus_neigh[:-1] = ordering[1:]
    plus_neigh[-1] = ordering[-1]
    return plus_neigh

def find_minus_neigh(ordering):
    '''
    This function returns the lower neighbour for a sfc ordering, see thesis

    Input:
    ---
    ordering: [1d-array] the (sfc) ordering of the Nodes.
    
    Return:
    ---
    minus_neigh: [1d-array] the lower-neighbour ordering.
    '''
    minus_neigh = np.zeros_like(ordering)
    minus_neigh[1:] = ordering[:-1]
    minus_neigh[0] = ordering[0]
    return minus_neigh

def gen_neighbour_keys(ndim, direct_neigh = False):
    '''
    Generate keys for create neighbours in multi-dimension,
    where -1 represents minus neigh, 0 represents no shift, 1 represents plus neigh.

    Input:
    ---
    ndim: [int] dimension for NN.
    direct_neigh: [bool] whether we are only considering the direct neighbours, for example, in 2d, they are (-1, 0), (0, -1), (1, 0), (0, 1).
    
    Return:
    ---
    C: [list of ndim-tuples] indicating the neighbours in md.    
    '''
    keys = (np.arange(3) - 1).astype('int')
    C = list(itertools.product(keys, repeat=ndim))
    if ndim == 1:
       C.remove((0,) * ndim)
       return C
    else: 
       if not direct_neigh:
          C.remove((0,) * ndim)
          return C
       else:
          C = []
          for i in range(ndim):
              uppers = (0,) * (i) + (1,) + (0,) * (ndim - i-1)
              lowers = (0,) * (i) + (-1,) + (0,) * (ndim - i-1)
              C.append(uppers)
              C.append(lowers)      
          return C     

def get_neighbour_index(ordering, tuple_i):
    '''
    Get neighbours for a sfc in multi-dimension,
    corresponding to a md neighbour key generated from 'gen_neighbour_keys' function.

    Input:
    ---
    ordering: [numpy.ndarray] multi-dimensional sfc.
    tuple_i: [tuple] indicates the respective place of this neighbour, see 'gen_neighbour_keys' function.
    
    Return:
    ---
    neigh_ordering: [numpy.ndarray] the neighbours in md, same shape to \{ordering\}.
    '''
    ndim = len(tuple_i)
    neigh_ordering = copy.deepcopy(ordering)
    indices_from = {}
    indices_to = {}
    for i in range(ndim):
        if tuple_i[i] == 1:
           indices_from.update({i: slice(1, None)})
           indices_to.update({i: slice(None, -1)})
        elif tuple_i[i] == -1:
           indices_from.update({i: slice(None, -1)})
           indices_to.update({i: slice(1, None)})

    idx_from = tuple([indices_from.get(dim, slice(None)) for dim in range(ndim)])
    idx_to = tuple([indices_to.get(dim, slice(None)) for dim in range(ndim)])
    
    neigh_ordering[idx_to] = ordering[idx_from]

    return neigh_ordering

# def get_neighbourhood_md(x, Ax, ordering = False):
#     '''
#     This function returns the neighbourhood for a sfc ordering/ tensor variable in multi-dimension.

#     Input:
#     ---
#     x: [numpy.ndarray or torch.Tensor] the sfc ordering/tensor variable.
#     Ax: [list of tuples] the neighbour_keys generated by 'gen_neighbour_keys' function.
#     ordering: [bool] indicating the input tensor is an ordering (int array), or not.
    
#     Return:
#     ---
#     neighbourhood: [tuple of numpy.ndarray or torch.Tensor] neighbourhood in multi-dimension.
#     '''
#     if ordering: x = x.long()
#     order_list = (x.flatten(), )
#     # size = x.flatten().shape[0]
#     for i, tuple_i in enumerate(Ax):
#         order_list += (get_neighbour_index(x, tuple_i).flatten(), )
#     order_list = torch.stack(order_list, 0)
#     # , order_list + size
#     # + (i+1) * size
#     # if channels > 1: 
#     #    for i in range(1, channels):
#     #        order_list = torch.cat((order_list, order_list + size), -1)
#     return order_list

# def get_concat_list_md(x, ordering_list):
#     '''
#     get the concat list of a tensor input x according to some ordering list of size (size of neighborhood, total nodes in x)

#     Input:
#     ---
#     x: [torch.Tensor] the tensor variable.
#     ordering_list: [(normally) numpy.ndarray] 1-D array represents sfc_ordering after applying MD-NN, of size [size of neighborhood * total nodes in x].
    
#     Return:
#     ---
#     ordered_tensor: [torch.Tensor] ordered neighbourhood tensor in md, input of 'NearestNeighbouring_md'.
#     '''
#     num_neigh = ordering_list.shape[0]
#     xx = copy.deepcopy(x[..., ordering_list[0]]).unsqueeze(-1)
#     for i in range(1, num_neigh): 
#         xx = torch.cat((xx, x[..., ordering_list[i]].unsqueeze(-1)), -1)
#     return xx

def get_neighbourhood_md(x, Ax, ordering = False):
    '''
    This function returns the neighbourhood for a sfc ordering/ tensor variable in multi-dimension.

    Input:
    ---
    x: [numpy.ndarray or torch.Tensor] the sfc ordering/tensor variable.
    Ax: [list of tuples] the neighbour_keys generated by 'gen_neighbour_keys' function.
    ordering: [bool] indicating the input tensor is an ordering (int array), or not.
    
    Return:
    ---
    neighbourhood: [tuple of numpy.ndarray or torch.Tensor] neighbourhood in multi-dimension.
    '''
    if ordering: x = x.long()
    order_list = (x.flatten(), )
    size = x.flatten().shape[0]
    for i, tuple_i in enumerate(Ax):
        order_list += (get_neighbour_index(x, tuple_i).flatten() + (i+1) * size, )
    order_list = torch.cat(order_list, 0)
    return order_list

def torch_reshape_fortran(x, shape):
    '''
    Fortran-like reshaping for pytorch, same to numpy.reshape(..., order = 'F').
    source: https://stackoverflow.com/questions/63960352/reshaping-order-in-pytorch-fortran-like-index-ordering.
    '''
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def get_concat_list_md(x, ordering_list, num_neigh_md, self_concat=1):
    '''
    get the concat list of a tensor input x according to some ordering list of size (size of neighborhood, total nodes in x)

    Input:
    ---
    x: [torch.Tensor] the tensor variable.
    ordering_list: [(normally) numpy.ndarray] 1-D array represents sfc_ordering after applying MD-NN, of size [size of neighborhood * total nodes in x].
    
    Return:
    ---
    ordered_tensor: [torch.Tensor] ordered neighbourhood tensor in md, input of 'NearestNeighbouring_md'.
    '''
    target_shape = x.shape + (num_neigh_md,)
    xx = (x.repeat(((1,) * (x.ndim - 1) + (num_neigh_md,))))[..., ordering_list]
    xx = torch_reshape_fortran(xx, target_shape)
    if self_concat > 1: xx = torch.cat(torch.chunk(xx, self_concat, dim = 1), -1)      
    return xx

class NearestNeighbouring_md(nn.Module):
    '''
    This class defines the "Neareast Neighbouring" Layer in the multi-dimension form, i.e 2D or 3D.
    
    __init__:
      Input:
      ---
      size: [tuple] the shape of the input image, 2D or 3D.
      initial_weight: [int] the initial weights for neighbourhoods, an intuiative value is defined in sfc_cae.py
      num_neigh: [int] the number of neighbours plus self, default is 3, but can be a larger number if self_concat > 1.

    __forward__(tensor_list):
      Input:
      ---
      tensor_list: [torch.FloatTensor] the concat list of our variable x and its neighbours, concatenate at the last dimension, 
                   of shape [number of time steps, number of Nodes, number of channels, number of neighbours]

      Returns:
      ---
      The element-wise (hadamard) product and addition: Σ(w_i * x_i) for x_i ɛ {neighbourhood of x}  
    '''
    def __init__(self, shape, initial_weight=None, channels = 1, num_neigh_md = 3, self_concat = 1):
        super(NearestNeighbouring_md, self).__init__()
        self.size = np.prod(shape)
        # self.dim = len(shape)
        self.channels = channels
        self.num_neigh_md = num_neigh_md
        self.self_concat = self_concat
        if initial_weight is None: initial_weight = 1/ (self.num_neigh_md * self_concat)
        self.weight_shape = (self.size, self.num_neigh_md * self.self_concat)
        if self.channels > 1: self.weight_shape = (self.channels,) + self.weight_shape
        self.weights = nn.Parameter(torch.ones(self.weight_shape) * initial_weight)
        self.bias = nn.Parameter(torch.zeros(self.size))

    def forward(self, tensor_list):
        tensor_list *= self.weights
        return torch.sum(tensor_list, -1) + self.bias

# def expand_snapshot_for_structured_backward(x, num_diff_nodes):
#     '''
#     fill the node number difference from unstructured and (virtual) structured grids.
#     '''
#     filled = torch.flip(x, (-1,))[..., 1:num_diff_nodes + 1]
#     return torch.cat((x, filled), -1)

def gen_filling_paras(unstructured_size, structured_size):
    '''
    Return filling indexes for a unstructured grid -> structured grid, with
    backward-forward approach.

    Input:
    ---
    unstructured_size: [int] the number of Nodes in the actual unstructured grid
    structured_size: [int] the number of Nodes in our artifical structured grid

    Output:
    ---
    n_fold: [int] the folds of x
    flip_time: [int] the times for filpping (copying/reversing).
    end_backward: [int] does this expand sfc ending in a inverse order?
    remainder: [int] the remaining nodes, if flip_time = 0, this is simply {structured_size - unstructured_size}.
    '''
    assert structured_size >= unstructured_size, 'Make sure the virtual structured grid you are constructing have more nodes than the original unstructured mesh!'
    unstructured_size -= 1
    n_fold = structured_size // unstructured_size
    remainder = structured_size % unstructured_size
    flip_time = n_fold // 2
    end_backward = bool(n_fold % 2)
    return n_fold, flip_time, end_backward, remainder

def expand_snapshot_backward_connect(x, n_fold, flip_time, end_backward, remainder):
    '''
    Fill the node number difference from unstructured and (virtual) structured grids.

    Input:
    ---
    x: [Torch.Tensor] the fluid snapshots, in batch.

    ## Next three parameters see function 'gen_filling_paras()' ##
    n_fold: [int] the folds of x
    flip_time: [int] the times for filpping (copying/reversing).
    end_backward: [int] does this expand sfc ending in a inverse order?
    remainder: [int] the remaining nodes, if flip_time = 0, this is simply {structured_size - unstructured_size}.

    Output:
    ---  
    xx: [Torch.Tensor] expand snapshot on structured grid.
    '''

    num_nodes = x.shape[-1] - 1
    forward_x = x[..., :num_nodes]
    backward_x = torch.flip(x, (-1,))[..., :num_nodes]
    if flip_time > 0:
       flipped = torch.cat((forward_x, backward_x), -1)
       if flip_time > 1: flipped = flipped.repeat((1,) * (x.ndim - 1) + (flip_time,))
    else: flipped = None
    if end_backward:
       remain =  torch.cat((forward_x, backward_x[..., :remainder]), -1)
    else:
       remain = forward_x[..., :remainder]
    if flipped is not None: return torch.cat((flipped, remain), -1)
    else: return remain

def reduce_expanded_snapshot(xx, unstructured_size, n_fold, flip_time, end_backward, remainder, scheme='mean'):
    '''
    Collect the results from the expanded structured grid.

    Input:
    ---
    xx: [Torch.Tensor] the expanded fluid snapshots by 'expand_snapshot_backward_connect()', in batch.

    ## Next three parameters see function 'gen_filling_paras()' ##
    flip_time: [int] the times for filpping (copying/reversing).
    end_backward: [int] does this expand sfc ending in a inverse order?
    remainder: [int] the remaining nodes, if flip_time = 0, this is simply {structured_size - unstructured_size}.
    scheme: [string] the reduce scheme, default is 'mean' (taking average), 'truncate' is also avaliable.

    Output:
    ---  
    reduced_x: [Torch.Tensor] reduced snapshot on unstructured grid.
    '''
    if scheme=='mean':
      xx = xx.float()
      remain = xx[..., -remainder:]
      folded = xx[..., :xx.shape[-1] - remainder]

      if end_backward: 
         remain = torch.flip(remain, (-1,)) 

      folded = folded.reshape(folded.shape[:-1] + (n_fold, unstructured_size - 1))

      forward_part = folded[..., ::2, :]
      forward_duplicates = forward_part.shape[-2]
      backward_part = folded[..., 1::2, :]
      backward_duplicates = backward_part.shape[-2]
      
      forward_part = forward_part.sum(-2)
      backward_part = backward_part.sum(-2)
      backward_part = torch.flip(backward_part, (-1,))

      reduced_x  = torch.cat((forward_part[..., 0].unsqueeze(-1), forward_part[..., 1:] + backward_part[..., :-1], backward_part[..., -1].unsqueeze(-1)), -1)
      if end_backward: 
         reduced_x[..., -remainder:] += remain
         reduced_x[..., -remainder:-1] /= n_fold + 1
         reduced_x[..., -1] /= backward_duplicates + 1
         reduced_x[..., 1:remainder] /= n_fold
         reduced_x[..., 0] /= forward_duplicates
      else:
         reduced_x[..., :remainder] += remain
         reduced_x[..., 1:remainder] /= n_fold + 1
         reduced_x[..., 0] /= forward_duplicates + 1
         reduced_x[..., -remainder:-1] /= n_fold
         reduced_x[..., -1] /= backward_duplicates  

      return reduced_x

    elif scheme=='truncate': return xx[..., :unstructured_size]

def ordering_tensor(tensor, ordering):
    '''
    This function orders the tensor in the 0-axis with a provided ordering.

    Input:
    ---
    tensor: [torch.FloatTensor] the simulation tensor, of shape [number of time steps, number of Nodes, number of channels]
    ordering: [1d-array] the (sfc) ordering of the Nodes.

    Output:
    ---
    tensor: [torch.FloatTensor] the ordered simulation tensor.
    '''
    return tensor[..., ordering]

class NearestNeighbouring(nn.Module):
    '''
    This class defines a custom Pytorch Layer, known as "Neareast Neighbouring", see Thesis
    
    __init__:
      Input:
      ---
      size: [int] the number of Nodes of each snapshot
      initial_weight: [int] the initial weights for w, w+, and w-, an intuiative value is defined in sfc_cae.py
      num_neigh: [int] the number of neighbours plus self, default is 3, but can be a larger number if self_concat > 1.

    __forward__(tensor_list):
      Input:
      ---
      tensor_list: [torch.FloatTensor] the concat list of our variable x and its neighbours, concatenate at the last dimension, 
                   of shape [number of time steps, number of Nodes, number of channels, number of neighbours]

      Returns:
      ---
      The element-wise (hadamard) product and addition: (w^-) * (x^-) + w * x + (w^+) * (x^+) + b
    '''
    def __init__(self, size, initial_weight, num_neigh = 3):
        super(NearestNeighbouring, self).__init__()
        self.size = size
        self.num_neigh = num_neigh
        self.weights = nn.Parameter(torch.ones(size, num_neigh) * initial_weight)
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, tensor_list):
        tensor_list *= self.weights
        return torch.sum(tensor_list, -1) + self.bias

def expend_SFC_NUM(sfc_ordering, partitions):
    '''
    This function construct a extented_sfc for components > 1.

    Input:
    ---
    sfc_ordering: [1d-array] the (sfc) ordering of the Nodes.  
    partitions: [int] the number of components/channels we have, equal to x.shape[-1]

    Output:
    ---
    sfc_ext: [int] the extended sfc ordering.
    '''
    size = len(sfc_ordering)
    sfc_ext = np.zeros(size * partitions, dtype = 'int')
    for i in range(partitions):
        sfc_ext[i * size : (i+1) * size] = i * size + sfc_ordering
    return sfc_ext

def find_size_conv_layers_and_fc_layers(size, kernel_size, padding, stride, dims_latent, sfc_nums, input_channel, increase_multi, num_final_channels, ndim=1):
    '''
    This function contains the algorithm for finding 1D convolutional layers and fully-connected layers depend on the input, see thesis

    Input:
    ---
    size: [int] the number of Nodes in each snapshot.
    kernel_size: [int] the constant kernel size throughout all filters.
    padding: [int] the constant padding throughout all filters.
    stride: [int] the constant stride throughout all filters.
    dims_latent: [int] the dimension of latent varible we are compressed to.
    sfc_nums: [int] the number of space-filling curves we use.
    input_channel: [int] the number of input_channels of the tensor, equals to components * self_concat, see 'sfc_cae.py'
    increase_multi: [int] an muliplication factor we have for consecutive 1D Conv Layers.
    num_final_channels: [int] the maximum number we defined for all Layers.
    ndim: [int] the dimension of ConvLayers, only used when second sfc(s) is used.

    Output:
    ---
    conv_size: [1d-array] the shape n_H at the 0-axis of training tensor x after each 1D Conv layer, first one is original shape.
    len(channels) - 1: [int] the number of 1D Conv layers
    size_fc: [1d-array] the sizes of the fully-connected layers
    channels: [1d-array] the number of channels/filters in each layer, first one is input_channel.
    inv_conv_start: [int] the size of the penultimate fully-connected layer, equals to size_fc[-2], just before dims_latent.
    np.array(output_paddings[::-1][1:]): [1d-array] the output_paddings, used for the Decoder.
    '''
    channels = [input_channel]
    output_paddings = [(size + 2 * padding - kernel_size) % stride]
    conv_size = [size]

    # find size of convolutional layers 
    while size ** ndim * num_final_channels * sfc_nums > 4000: # a intuiative value of 4000 is hard-coded here, to prohibit large size of FC layers, which would lead to huge memory cost.
        size = (size + 2 * padding - kernel_size) // stride + 1 # see the formula for computing shape for 1D conv layers
        conv_size.append(size)
        if num_final_channels >= input_channel * increase_multi: 
            input_channel *= increase_multi
            output_paddings.append((size + 2 * padding - kernel_size) % stride)
            channels.append(input_channel)
        else: 
            channels.append(num_final_channels)
            output_paddings.append((size + 2 * padding - kernel_size) % stride)
       
    # find size of fully-connected layers 
    size = size ** ndim
    inv_conv_start = size
    size = size * sfc_nums * num_final_channels
    size_fc = [size]
    if stride < 4: stride = 8
    # an intuiative value 1.5 of exponential is chosen here, because we want the size_after_decrease > dims_latent * (stride ^ 0.5), which is not too close to dims_latent.
    while size // (stride ** 1.5) > dims_latent:  
        size //= stride
        if size * stride < 100 and size < 50: break # we do not not want more than two FC layers with size < 100, also we don't want too small size at the penultimate layer.
        size_fc.append(size)
    size_fc.append(dims_latent)

    return conv_size, len(channels) - 1, size_fc, channels, inv_conv_start, np.array(output_paddings[::-1][1:])


#################################################### Extension functions for data post-processing ######################################################################


def vtu_compress(data_path, save_path, vtu_fields, autoencoder, tk, tb, start_index = None, end_index = None, model_device = torch.device('cpu')):
    '''
    This function would compress the specified fields of vtu files based on a trained SFC_CAE Autoencoder defined in sfc_cae.py, 
    to .pt files snapshot by snapshot.
    
    Input:
    ---
    data_path: [string] the path (with '/') for the vtu datas.
    save_path: [string] the saving path (no '/') for the compressed variables (.pt files)
    vtu_fields: [list] the list of vtu_fields if read in vtu files, the last dimension of the tensor, e.g. ['Velocity', 'Pressure']
    autoencoder: [SFC_CAE object] the trained SFC_(V)CAE.
    tk: [torch.FloatTensor] the tk coeffcients for the dataset of standardlisation, of shape [number of components,]
    tb: [torch.FloatTensor] the tb coeffcients for the dataset of standardlisation, of shape [number of components,]
    start_index: [int] the start_index of the time level, default None, will be set as the first snapshot.
    end_index: [int] the end_index of the time level, default None, will be set as the last snapshot.
    model_device: [torch.device] compute the autoencoder on GPU or CPU.

    Output:
    ---
    Compressed .pt files in {save_path}.
    '''
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[0].split('_')
    file_prefix.pop(-1)
    if len(file_prefix) != 1: file_prefix = '_'.join(file_prefix) + "_"
    else: file_prefix = file_prefix[0] + "_"
    file_format = '.vtu'
    print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
    point_data = {''}
    variational = autoencoder.encoder.variational
    dimension = autoencoder.encoder.dimension
    cnt_progress = 0
    print("Compressing vtu Data......\n")
    bar=progressbar.ProgressBar(maxval=num_data)
    bar.start()
    start = 0
    while(True):
        if not os.path.exists(F'{file_prefix}%d{file_format}' % start):
            print(F'{file_prefix}%d{file_format} not exist, starting number switch to {file_prefix}%d{file_format}' % (start, start+1))
            start += 1
        else: break
    if start_index is None: start_index = start
    if end_index is None: end_index = num_data + start
    os.system(F'mkdir -p {save_path}') 
    save_path += '/'
    for i in range(start_index, end_index):
            point_data = {}
            field_spliter = [0]
            vtu_file = meshio.read(F'{file_prefix}%d{file_format}' % i)
            coords = vtu_file.points
            cells = vtu_file.cells_dict         
            filename = F'{save_path}reconstructed_%d{file_format}' % i
            for j in range(len(vtu_fields)):
                vtu_field = vtu_fields[j]
                field = vtu_file.point_data[vtu_field]
                # see if last dimension is zero
                if dimension == 2 and field.shape[-1] > 2: field = field[..., :-1]
                vari_tensor = torch.from_numpy(field)
                if vari_tensor.ndim == 1: vari_tensor = vari_tensor.unsqueeze(-1)
                if j == 0: tensor = vari_tensor.unsqueeze(0)
                else: tensor = torch.cat((tensor, vari_tensor.unsqueeze(0)), -1)
                field_spliter.append(tensor.shape[-1])
            tensor = tensor.float()
            for k in range(tensor.shape[-1]):
                tensor[...,k] *= tk[k]
                tensor[...,k] += tb[k] 
            tensor = tensor.to(model_device)
            if variational: compressed_tensor, _ = autoencoder.encoder(tensor)
            else: compressed_tensor = autoencoder.encoder(tensor)
            compressed_tensor = compressed_tensor.to('cpu') 
            print('compressing snapshot %d, shape:' % i, compressed_tensor.shape)
            torch.save(compressed_tensor, save_path +'compressed_%d.pt' % i)
            cnt_progress +=1
            bar.update(cnt_progress)
    bar.finish()
    print('\n Finished compressing vtu files.')

def read_in_compressed_tensors(data_path, start_index = None, end_index = None):
    '''
    This function would read the compressed variables from the outcome of vtu_compress(),  to a tensor. 
    It is implemented for experiments over the latent space e.g. Noise experiments, create t-SNE plots. 
    
    Input:
    ---
    data_path: [string] the path (with '/') for the vtu datas.
    start_index: [int] the start_index of the time level, default None, will be set as the first snapshot.

    Output:
    ---
    latent_tensor: [torch.FloatTensor] tensor of all latent variables, of shape [number of snapshots, dims_latent]
    '''
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[0].split('_')
    file_prefix.pop(-1)
    if len(file_prefix) != 1: file_prefix = '_'.join(file_prefix) + "_"
    else: file_prefix = file_prefix[0] + "_"
    file_format = '.pt'
    print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
    point_data = {''}
    cnt_progress = 0
    print("Reading in compressed Data......\n")
    bar=progressbar.ProgressBar(maxval=num_data)
    bar.start()
    start = 0
    while(True):
        if not os.path.exists(F'{file_prefix}%d{file_format}' % start):
            print(F'{file_prefix}%d{file_format} not exist, starting number switch to {file_prefix}%d{file_format}' % (start, start+1))
            start += 1
        else: break
    if start_index is None: start_index = start
    if end_index is None: end_index = num_data + start   
    for i in range(start_index, end_index):
        print('read in compressed data %d ...' % i)
        if i == start_index:
           full_tensor = torch.load(F'{file_prefix}%d{file_format}' % i)
        else:
           full_tensor = torch.cat((full_tensor, torch.load(F'{file_prefix}%d{file_format}' % i)), 0)
        print(full_tensor.shape)
        bar.update(cnt_progress)
    bar.finish() 
    return full_tensor  

def decompress_to_vtu(full_tensor, tamplate_vtu, save_path, vtu_fields, field_spliter, autoencoder, tk, tb, start_index = None, end_index = None, model_device = torch.device('cpu')):
    '''
    This function would decompress the full latent-variables to vtu files based on a trained SFC_CAE Autoencoder defined in sfc_cae.py, snapshot by snapshot.
    
    Input:
    ---
    full_tensor: [torch.FloatTensor] tensor of all latent variables, of shape [number of snapshots, dims_latent]
    tamplate_vtu: [vtu file] a tamplate vtu file from the {data_path} to read the coords and cell_dict from.
    save_path: [string] the saving path (no '/') for the compressed variables (.pt files)
    vtu_fields: [list] the list of vtu_fields if read in vtu files, the last dimension of the tensor, e.g. ['Velocity', 'Pressure']    
    field_spliter: [1d-array] the start point of different vtu_fields, similar to Intptr() of a CSRMatrix, see doc of Scipy.Sparse.CSRMatrix.
    autoencoder: [SFC_CAE object] the trained SFC_(V)CAE.
    tk: [torch.FloatTensor] the tk coeffcients for the dataset of standardlisation, of shape [number of components,]
    tb: [torch.FloatTensor] the tb coeffcients for the dataset of standardlisation, of shape [number of components,]
    start_index: [int] the start_index of the time level, default None, will be set as the first snapshot.
    end_index: [int] the end_index of the time level, default None, will be set as the last snapshot.
    model_device: [torch.device] compute the autoencoder on GPU or CPU.

    Output:
    ---
    Deompressed .vtu files in {save_path}.
    '''
    file_format = '.vtu'
    point_data = {''}
    coords = tamplate_vtu.points
    cells = tamplate_vtu.cells_dict 
    variational = autoencoder.encoder.variational
    dimension = autoencoder.encoder.dimension
    cnt_progress = 0
    print("Write vtu Data......\n")
    bar=progressbar.ProgressBar(maxval=full_tensor.shape[0])
    bar.start()
    start = 0
    if start_index is None: start_index = 0
    if end_index is None: end_index = full_tensor.shape[0]
    os.system(F'mkdir -p {save_path}') 
    save_path += '/'
    for i in range(start_index, end_index):
            point_data = {}
            tensor = full_tensor[i]    
            print("Reconstructing vtu %d ......\n" % i)
            filename = F'{save_path}reconstructed_from_latent_%d{file_format}' % i
            tensor = tensor.to(model_device)
            reconsturcted_tensor = autoencoder.decoder(tensor)
            reconsturcted_tensor = reconsturcted_tensor.to('cpu') 
            for k in range(reconsturcted_tensor.shape[-1]):
                reconsturcted_tensor[...,k] -= tb[k]
                reconsturcted_tensor[...,k] /= tk[k]       
            reconsturcted_tensor = reconsturcted_tensor.squeeze(0)    
            print(reconsturcted_tensor.shape)
            for j in range(len(vtu_fields)):
                vtu_field = vtu_fields[j]
                field = (reconsturcted_tensor[..., field_spliter[j] : field_spliter[j + 1]]).detach().numpy()
                point_data.update({vtu_field: field})
            mesh = meshio.Mesh(coords, cells, point_data)
            mesh.write(filename)
            cnt_progress +=1
            bar.update(cnt_progress)
    bar.finish()
    print('\n Finished decompressing vtu files.')

def result_vtu_to_vtu(data_path, save_path, vtu_fields, autoencoder, tk, tb, start_index = None, end_index = None, model_device = torch.device('cpu')):
    '''
    This function provides a simple reconstruction with a trained autoencoder directly to .vtu files, 
    especially useful for experiment purpose: directly view the reconstruction performance.
    
    Input:
    ---
    data_path: [string] the path (with '/') for the vtu datas.
    save_path: [string] the saving path (no '/') for the compressed variables (.pt files)
    vtu_fields: [list] the list of vtu_fields if read in vtu files, the last dimension of the tensor, e.g. ['Velocity', 'Pressure']    
    autoencoder: [SFC_CAE object] the trained SFC_(V)CAE.
    tk: [torch.FloatTensor] the tk coeffcients for the dataset of standardlisation, of shape [number of components,]
    tb: [torch.FloatTensor] the tb coeffcients for the dataset of standardlisation, of shape [number of components,]
    start_index: [int] the start_index of the time level, default None, will be set as the first snapshot.
    end_index: [int] the end_index of the time level, default None, will be set as the last snapshot.
    model_device: [torch.device] compute the autoencoder on GPU or CPU.

    Output:
    ---
    Reconstructed .vtu files in {save_path}.
    '''
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[0].split('_')
    file_prefix.pop(-1)
    if len(file_prefix) != 1: file_prefix = '_'.join(file_prefix) + "_"
    else: file_prefix = file_prefix[0] + "_"
    file_format = '.vtu'
    print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
    point_data = {''}
    variational = autoencoder.encoder.variational
    dimension = autoencoder.encoder.dimension
    cnt_progress = 0
    print("Write vtu Data......\n")
    bar=progressbar.ProgressBar(maxval=num_data)
    bar.start()
    start = 0
    while(True):
        if not os.path.exists(F'{file_prefix}%d{file_format}' % start):
            print(F'{file_prefix}%d{file_format} not exist, starting number switch to {file_prefix}%d{file_format}' % (start, start+1))
            start += 1
        else: break
    if start_index is None: start_index = start
    if end_index is None: end_index = num_data + start
    os.system(F'mkdir -p {save_path}') 
    save_path += '/'
    for i in range(start_index, end_index):
            point_data = {}
            field_spliter = [0]
            vtu_file = meshio.read(F'{file_prefix}%d{file_format}' % i)
            coords = vtu_file.points
            cells = vtu_file.cells_dict         
            filename = F'{save_path}reconstructed_%d{file_format}' % i
            for j in range(len(vtu_fields)):
                vtu_field = vtu_fields[j]
                field = vtu_file.point_data[vtu_field]
                # see if last dimension is zero
                if dimension == 2 and field.shape[-1] > 2: field = field[..., :-1]
                vari_tensor = torch.from_numpy(field)
                if vari_tensor.ndim == 1: vari_tensor = vari_tensor.unsqueeze(-1)
                if j == 0: tensor = vari_tensor.unsqueeze(0)
                else: tensor = torch.cat((tensor, vari_tensor.unsqueeze(0)), -1)
                field_spliter.append(tensor.shape[-1])
            tensor = tensor.float()
            for k in range(tensor.shape[-1]):
                tensor[...,k] *= tk[k]
                tensor[...,k] += tb[k] 
            tensor = tensor.to(model_device)
            if variational: reconsturcted_tensor, _ = autoencoder(tensor)
            else: reconsturcted_tensor = autoencoder(tensor)
            print('Reconstruction MSE error for snapshot %d: %f' % (i, nn.MSELoss()(tensor, reconsturcted_tensor).item()))
            reconsturcted_tensor = reconsturcted_tensor.to('cpu') 
            for k in range(tensor.shape[-1]):
                reconsturcted_tensor[...,k] -= tb[k]
                reconsturcted_tensor[...,k] /= tk[k]       
            reconsturcted_tensor = reconsturcted_tensor.squeeze(0)    
            print(reconsturcted_tensor.shape)
            for j in range(len(vtu_fields)):
                vtu_field = vtu_fields[j]
                field = (reconsturcted_tensor[..., field_spliter[j] : field_spliter[j + 1]]).detach().numpy()
                point_data.update({vtu_field: field})
            mesh = meshio.Mesh(coords, cells, point_data)
            mesh.write(filename)
            cnt_progress +=1
            bar.update(cnt_progress)
    bar.finish()
    print('\n Finished reconstructing vtu files.')
