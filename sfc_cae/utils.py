"""
This module contains extra functions for training.py/ sfc_cae.py as supplement.
Author: Pozzetti Andrea
Github handle: acse-ap2920
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import space_filling_decomp_new as sfc
import x_conv_fixed_length as sfc_interpolate
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
import meshio
import re
import wandb

# create an animation
from matplotlib import animation
from IPython.display import HTML
# custom colormap
import cmocean

import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, TensorDataset, Dataset


#################################################### Functions for data pre-processing / data loading ######################################################################

def read_in_files(data_path, file_format='vtu', vtu_fields=None):
    '''
    This function reads in the vtu/txt files in a {data_path} as tensors, of shape [snapshots, number of Nodes, Channels]

    Input:
    ---
    data_path: [string] the data_path which holds vtu/txt files, no other type of files are accepted!!!
    file_format: [string] 'vtu' or 'txt', the format of the file.
    vtu_fields: [list] the list of vtu_fields if read in vtu files, the last dimension of the tensor, e.g. ['Velocity', 'Pressure']

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

def read_in_files_adaptive(data_path, file_format='vtu', vtu_fields=["Velocity"]):
  data = os.listdir(data_path) #glob.glob(data_path + "*")
  num_data = len(data)
  file_prefix = data_path + "fpc_cg_"
  file_format = '.vtu'
  print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
  cnt_progress = 0
  if (file_format == ".vtu"):
      print("Read in vtu Data......\n")
      bar=progressbar.ProgressBar(maxval=num_data)
      bar.start()
      data = []
      coords = [0]
      cells = []
      start = 0
      while(True):
          if not os.path.exists(F'{file_prefix}%d{file_format}' % start):
              print(F'{file_prefix}%d{file_format} not exist, starting number switch to {file_prefix}%d{file_format}' % (start, start+1))
              start += 1
          else: break
      for i in range(start, num_data + start):
          data.append([])
          print(F'{file_prefix}%d{file_format}' % i)
          vtu_file = meshio.read(F'{file_prefix}%d{file_format}' % i)
          if not np.array_equal(coords[-1],vtu_file.points):
              coords.append(vtu_file.points)
              cells.append(vtu_file.cells_dict)
              print('mesh adapted at snapshot %d' % i)
          for j in range(len(vtu_fields)):
              vtu_field = vtu_fields[j]
              if not vtu_field in vtu_file.point_data.keys():
                  raise ValueError(F'{vtu_field} not avaliable in {vtu_file.point_data.keys()} for {file_prefix} %d {file_format}' % i)
              field = vtu_file.point_data[vtu_field]
              if j == 0:
                  if field.ndim == 1: field = field.reshape(field.shape[0], 1)
                  print(field.shape,coords[i+1][:,0].shape)
                  data[i - start] = torch.from_numpy(np.concatenate((field[:,:2],coords[i+1][:,:2]),axis=1)).permute((1,0))
          # print(data[i - start].shape)
          cnt_progress +=1
          bar.update(cnt_progress)
      bar.finish()
      coords.pop(0)
  return data, coords, cells


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

def standardlize_listoftensors_adaptive(tensors, lower = -1, upper = 1):
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
    
    tk = torch.zeros(4)
    tb = torch.zeros(4)
    for i in range(4):
      currentmax = 0
      currentmin = 0
      for j in range(len(tensors)):
        tmax = tensors[j][i,...].max()
        tmin = tensors[j][i,...].min()
        if currentmax<tmax:
          currentmax = tmax
        if currentmin>tmin:
          currentmin = tmin
      tk[i] = (upper - lower) /(currentmax - currentmin)
      tb[i] = (currentmax * lower - currentmin * upper) /(currentmax - currentmin)
      for j in range(len(tensors)):     
        tensors[j][i,...] *= tk[i]
        tensors[j][i,...] += tb[i]
    
    return tensors, tk, tb

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
    return tensor[:, ordering]

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
        super().__init__()
        self.size = size
        self.num_neigh = num_neigh
        self.weights = nn.Parameter(torch.ones(size, num_neigh) * initial_weight)
        self.bias = nn.Parameter(torch.zeros(size))

    def __repr__(self):
      return "NearestNeighbouring(Size="+str(self.size)+", num_neigh="+str(self.num_neigh)+")"

    def forward(self, tensor_list):
        tensor_list *= self.weights
        return tensor_list.sum(-1) + self.bias

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

def find_size_conv_layers_and_fc_layers(size, kernel_size, padding, stride, dims_latent, sfc_nums, input_channel, increase_multi, num_final_channels, nfclayers = 0):
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
    output_paddings = [size % stride]
    conv_size = [size]

    intval = dims_latent

    if nfclayers>0:
      intval = min(4000, (stride**0.5)*dims_latent*((stride)**(nfclayers)))
    
    if num_final_channels * sfc_nums > intval:
      intval = num_final_channels * sfc_nums
      print("Compression to", dims_latent, "variables was not possible through convolutional layers alone! Stopping at", intval , "dims")

    # find size of convolutional layers 
    while size * num_final_channels * sfc_nums > intval: # a intuiative value of 4000 is hard-coded here, to prohibit large size of FC layers, which would lead to huge memory cost.
        # print("target:", intval)
        # print("size:", size)
        # print("actual size:", size * num_final_channels * sfc_nums)
        size = (size + 2 * padding - kernel_size) // stride + 1 # see the formula for computing shape for 1D conv layers
        # print("after size:", size)
        # print("after actual size:", size * num_final_channels * sfc_nums)
        conv_size.append(size)
        if num_final_channels >= input_channel * increase_multi: 
            input_channel *= increase_multi
            output_paddings.append(size % stride)
            channels.append(input_channel)
        else: 
            channels.append(num_final_channels)
            output_paddings.append(size % stride)
        
    # find size of fully-connected layers 
    if nfclayers>0:
      inv_conv_start = size
      size *= sfc_nums * num_final_channels
      size_fc = [size]
      # an intuiative value 1.5 of exponential is chosen here, because we want the size_after_decrease > dims_latent * (stride ^ 0.5), which is not too close to dims_latent.
      while size // (stride ** 1.5) > dims_latent:  
          size //= stride
          if size * stride < 100 and size < 50: break # we do not not want more than two FC layers with size < 100, also we don't want too small size at the penultimate layer.
          size_fc.append(size)
      size_fc.append(dims_latent)
    else:
      size_fc = [size*sfc_nums*num_final_channels]
      inv_conv_start = size

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
            else:
              print(tensor.shape)
              reconsturcted_tensor = autoencoder(tensor)
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

def interpolate(x, conc, nonods_l):
  nscalar = conc.shape[0]
  nonods = conc.shape[1]
  ndim = x.shape[0]
  x_l = torch.zeros((2,4000))
  conc_l =  torch.zeros((2,4000))
  
  x_l,conc_l = sfc_interpolate.x_conv_fixed_length(x,conc,nonods_l,nonods,ndim,nscalar)

  return x_l, conc_l