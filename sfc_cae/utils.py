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
import meshio
import re

# create an animation
from matplotlib import animation
from IPython.display import HTML
import cmocean

import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, TensorDataset, Dataset

def read_in_files(data_path, file_format='vtu', vtu_fields=None):
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[-2].split('_')
    file_prefix.pop(-1)
    if len(file_prefix) != 1: file_prefix = '_'.join(file_prefix) + "_"
    else: file_prefix = file_prefix[0] + "_"
    # file_format = data[0].split('.')[-1]
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
                # if field.ndim > 1 and field[..., -1].max() - field[..., -1].min() < 1e-8: field = field[...,0:-1] # get rid of zero coords
                if j == 0:
                   if field.ndim == 1: field = field.reshape(field.shape[0], 1)
                   data[i - start] = field
                else:
                   if field.ndim == 1: field = field.reshape(field.shape[0], 1)
                   data[i - start] = np.hstack((data[i - start], field))
            # print(data[i - start].shape)
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

def get_simulation_index(num, simulation):
    return np.arange(num * (simulation.steps + 1), (num + 1) * (simulation.steps + 1))

def read_parameters(setting_file = 'parameters.ini'):
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
    if tensor.ndim > 2:
       for i in range(tensor.shape[-1]):
           tensor[...,i] *= t_std[i]
           tensor[...,i] += t_mean[i]
       else:
          tensor *= t_std
          tensor += t_mean
    return tensor

def destandardlize_tensor(tensor, tk, tb):
    if tensor.ndim > 2:
       for i in range(tensor.shape[-1]):
           tensor[...,i] -= tb[i]
           tensor[...,i] /= tk[i]
    else:
        tensor -= tb
        tensor /= tk
    return tensor

def find_min_and_max(data_path, only_get_names = False):
    data = glob.glob(data_path + "*")
    num_data = len(data)
    print(num_data)
    # file_prefix = data[0].split('.')[:-1]
    # file_prefix = ''.join(file_prefix)
    # file_prefix = file_prefix.split('_')[:-1]
    # file_prefix = ''.join(file_prefix) + '_'
    # file_format = data[0].split('.')[-1]
    # file_format = '.' + file_format
    # print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
    cnt_progress = 0
    print("Loading Data......\n")
    bar=progressbar.ProgressBar(maxval=num_data)
    bar.start()
    # while(True):
    #     if not os.path.exists(F'{file_prefix}%d{file_format}' % start):
    #         print(F'{file_prefix}%d{file_format} not exist, starting number switch to {file_prefix}%d{file_format}' % (start, start+1))
    #         start += 1
    #     else: break
    for i in range(num_data):
        filename = data[i] # F'{file_prefix}%d{file_format}' % i
        if not only_get_names:
           tensor = torch.load(filename)
           if i == 0:
              t_min = tensor.min(0).values.unsqueeze(-1)
              t_max = tensor.max(0).values.unsqueeze(-1)
           else:
              t_min = torch.cat((t_min, tensor.min(0).values.unsqueeze(-1)), -1)
              t_max = torch.cat((t_max, tensor.max(0).values.unsqueeze(-1)), -1)
        # data.append(filename)
        cnt_progress +=1
        bar.update(cnt_progress)
    bar.finish()
    if not only_get_names:
        t_min = t_min.min(-1).values
        t_max = t_max.max(-1).values        
        np.savetxt('./t_max.txt', t_max)
        np.savetxt('./t_min.txt', t_min)
    return data

def get_path_data(data_path, indexes):
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
      def __init__(self, path_dataset, lower, upper):
          self.dataset = path_dataset
          self.length = len(path_dataset)
          t_max = torch.load(self.dataset[0]).max(0).values.unsqueeze(0)
          t_min = torch.load(self.dataset[0]).min(0).values.unsqueeze(0)
          cnt_progress = 0
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
        #   self.t_max = np.loadtxt('./t_max.txt')
        #   self.t_min = np.loadtxt('./t_min.txt')
          self.tk = (upper - lower) / (self.t_max - self.t_min)
          self.tb = (self.t_max * lower - self.t_min * upper) / (self.t_max - self.t_min)

          print('tk: ', self.tk, '\n')
          print('tb: ', self.tb, '\n')

      def __getitem__(self, index):
          tensor = torch.load(self.dataset[index])
          return (tensor * self.tk + self.tb).float()
      
      def __len__(self):
          return self.length
          

def get_sfc_curves_from_coords(coords, num):
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
    '''copied from Claire's Code'''
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

def plot_trace_vtu_2D(coords, levels):
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
        ax.plot(coords[cuts[i]:cuts[i+1], 0], coords[cuts[i]:cuts[i+1], 1], '-')
    plt.axis('off')
    plt.show() 

def countour_plot_vtu_2D(coords, levels, mask=True, values=None, cmap = None):
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
    plt.show()  

class anim_vtu_fields_2D():
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
       # plt.show() 

    def update_grid(self, n_step: int):
       # self.image.set_array(self.values[n_step])
       self.image = self.ax.tricontourf(self.triang, self.values[n_step], levels = self.levels, cmap = self.cmap)
    #    print(n_step, self.values[n_step])
       print('frame %d saved.' % n_step)
       return self.image,
   
    def generate_anime(self):
       # anim = animation.FuncAnimation(self.fig, self.update_grid, frames = np.arange(0, self.steps))
       return animation.FuncAnimation(self.fig, self.update_grid, frames = np.arange(0, self.steps))

def find_plus_neigh(ordering):
    plus_neigh = np.zeros_like(ordering)
    plus_neigh[:-1] = ordering[1:]
    plus_neigh[-1] = ordering[-1]
    return plus_neigh

def find_minus_neigh(ordering):
    minus_neigh = np.zeros_like(ordering)
    minus_neigh[1:] = ordering[:-1]
    minus_neigh[0] = ordering[0]
    return minus_neigh

def ordering_tensor(tensor, ordering):
    return tensor[:, ordering]

class NearestNeighbouring(nn.Module):
    def __init__(self, size, initial_weight, num_neigh = 3):
        super().__init__()
        self.size = size
        self.num_neigh = num_neigh
        self.weights = nn.Parameter(torch.ones(size, num_neigh) * initial_weight)
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, tensor_list):
        tensor_list *= self.weights
        return tensor_list.sum(-1) + self.bias

def expend_SFC_NUM(sfc_ordering, partitions):
    size = len(sfc_ordering)
    sfc_ext = np.zeros(size * partitions, dtype = 'int')
    for i in range(partitions):
        sfc_ext[i * size : (i+1) * size] = i * size + sfc_ordering
    return sfc_ext

def find_size_conv_layers_and_fc_layers(size, stride, dims_latent, sfc_nums, input_channel, increase_multi, num_final_channels):
       channels = [input_channel]
       output_paddings = [size % stride]
       conv_size = [size]
       while size * num_final_channels * sfc_nums > 4000:
          size = size // stride + 1
          conv_size.append(size)
          if num_final_channels >= input_channel * increase_multi: 
              input_channel *= increase_multi
              output_paddings.append(size % stride)
              channels.append(input_channel)
          else: 
              channels.append(num_final_channels)
              output_paddings.append(size % stride)
    
       inv_conv_start = size
       size *= sfc_nums * num_final_channels
       size_fc = [size]
       while size // (stride ** 1.5) > dims_latent: 
          size //= stride
          if size * stride < 100 and size < 50: break
          size_fc.append(size)
       size_fc.append(dims_latent)

       return conv_size, len(channels) - 1, size_fc, channels, inv_conv_start, np.array(output_paddings[::-1][1:])


def result_to_vtu_unadapted(data_path, coords, cells, tensor, vtu_fields, field_spliters):
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[0].split('_')
    file_prefix.pop(-1)
    if len(file_prefix) != 1: file_prefix = '_'.join(file_prefix) + "_"
    else: file_prefix = file_prefix[0] + "_"
    file_format = '.vtu'
    print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
    point_data = {''}
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
    for i in range(start, num_data + start):
            point_data = {}
            filename = F'../reconstructed/reconstructed_%d{file_format}' % i
            for j in range(len(vtu_fields)):
                vtu_field = vtu_fields[j]
                field = tensor[i][..., field_spliters[j] : field_spliters[j + 1]].detach().numpy()
                point_data.update({vtu_field: field})
            mesh = meshio.Mesh(coords, cells, point_data)
            mesh.write(filename)
            cnt_progress +=1
            bar.update(cnt_progress)
    bar.finish()
    print('\n Finished writing vtu files.')


def vtu_compress(data_path, save_path, vtu_fields, autoencoder, tk, tb, variational = False, start_index = None, end_index = None, model_device = torch.device('cpu'), dimension = 3):
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[0].split('_')
    file_prefix.pop(-1)
    if len(file_prefix) != 1: file_prefix = '_'.join(file_prefix) + "_"
    else: file_prefix = file_prefix[0] + "_"
    file_format = '.vtu'
    print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
    point_data = {''}
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
            if variational: compressed_tensor = autoencoder.encoder(tensor)[0]
            else: compressed_tensor = autoencoder.encoder(tensor)
            compressed_tensor = compressed_tensor.to('cpu') 
            print('compressing snapshot %d, shape:' % i, compressed_tensor.shape)
            torch.save(compressed_tensor, save_path +'compressed_%d.pt' % i)
            cnt_progress +=1
            bar.update(cnt_progress)
    bar.finish()
    print('\n Finished compressing vtu files.')

def read_in_compressed_tensors(data_path, start_index = None, end_index = None):
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

def decompress_to_vtu(full_tensor, tamplate_vtu, save_path, vtu_fields, field_spliter, autoencoder, tk, tb, variational = False, start_index = None, end_index = None, model_device = torch.device('cpu'), dimension = 3):
    file_format = '.vtu'
    point_data = {''}
    coords = tamplate_vtu.points
    cells = tamplate_vtu.cells_dict 
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
            if variational: reconsturcted_tensor = autoencoder(tensor)[0]
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
    print('\n Finished writing vtu files.')
