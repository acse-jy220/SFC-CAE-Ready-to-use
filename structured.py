from simple_hilbert import *
from advection_block_analytical import *
import space_filling_decomp_new as sfc
import numpy as np  # Numpy
import scipy.sparse.linalg as spl
import scipy.linalg as sl
import scipy.sparse as sp
import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torchvision import transforms  # Transforms from torchvision
from util import *

def loadsimulation(simulaion_steps, simulaion_num, reshape = False):
    for i in range(simulaion_steps + 1):
        iter_data = np.loadtxt(F'{DATADIR}_%d/step_%d.txt'% (simulaion_num, i))
        if reshape: 
            size = np.sqrt(iter_data.shape[0]).astype('int')
            iter_data = iter_data.reshape((size, size))
        if i != 0: tensor = torch.cat((tensor, torch.unsqueeze(torch.from_numpy(iter_data), 0)), 0)
        else: 
           tensor = torch.unsqueeze(torch.from_numpy(iter_data), 0)
      
    return tensor

def load_tensor(simulation_indexes):
    total = len(simulation_indexes)
    cnt_progress = 0
    bar=progressbar.ProgressBar(maxval=total)
    tensor = loadsimulation(simulaion_steps, simulation_indexes[0])
    cnt_progress+=1
    bar.update(cnt_progress)    
    for i in range(1, total):
        tensor = torch.cat((tensor, loadsimulation(simulaion_steps, simulation_indexes[i])))
        cnt_progress+=1
        bar.update(cnt_progress)          
    bar.finish()
    return tensor

def index_split(train_ratio, valid_ratio, test_ratio, total_num = 500):
    if train_ratio + valid_ratio + test_ratio != 1:
        raise ValueError("The sum of three input ratios should be 1!")
    total_index = np.arange(1, total_num + 1)
    rng = np.random.default_rng()
    total_index = rng.permutation(total_index)
    knot_1 = int(total_num * train_ratio)
    knot_2 = int(total_num * valid_ratio) + knot_1
    train_index, valid_index, test_index = np.split(total_index, [knot_1, knot_2])
    return train_index, valid_index, test_index

def normalize_tensor(tensor):
    t_mean = torch.mean(tensor)
    t_std = torch.std(tensor)
    return (tensor - t_mean) / t_std, t_mean, t_std

def denormalize_tensor(normalized_tensor, t_mean, t_std):
    return normalized_tensor * t_std + t_mean

def sparse_square_grid(N):
    n = N ** 2
    
    offsets = [-N, -1, 0, 1, N]
    diags = []
    # coefficient in front of u_{i-N}:
    diags.append(np.ones(n-N))
    # coefficient in front of u_{i-1}:
    diags.append(np.ones(n-1))
    # main diagonal, zero for centre difference in space
    diags.append(np.ones(n))
    # coefficient in front of u_{i+1}:
    diags.append(diags[1])
    # coefficient in front of u_{i+N}:
    diags.append(diags[0])
    
    K = sp.diags(diags, offsets, format='csr')
    
    # loop over left-most column in grid (except first row)
    for i in range(N, n, N):
        K[i, i-1] = 0
        K[i-1, i] = 0
    K.eliminate_zeros()
    
    return K.indptr + 1, K.indices + 1, K.getnnz()

def get_hilbert_curves(size, num):
    Hilbert_index = hilbert_space_filling_curve(size)
    invert_Hilbert_index = np.argsort(Hilbert_index)
    if num == 1: return [Hilbert_index], [invert_Hilbert_index]
    elif num == 2:
        Hilbert_index_2 = Hilbert_index.reshape(size, size).T.flatten()
        invert_Hilbert_index_2 = np.argsort(Hilbert_index_2)
        return [Hilbert_index, Hilbert_index_2], [invert_Hilbert_index, invert_Hilbert_index_2]

def get_MFT_RNN_curves_structured(size, num):
    findm, colm, ncolm  = sparse_square_grid(size)
    curve_lists = []
    inv_lists = []
    ncurve = num
    graph_trim = -10  # has always been set at -10
    starting_node = 0 # =0 do not specifiy a starting node, otherwise, specify the starting node
    whichd, space_filling_curve_numbering = sfc.ncurve_python_subdomain_space_filling_curve(colm, findm, starting_node, graph_trim, ncurve, size**2, ncolm)
    for i in range(space_filling_curve_numbering.shape[-1]):
        curve_lists.append(np.argsort(space_filling_curve_numbering[:,i]))
        inv_lists.append(np.argsort(np.argsort(space_filling_curve_numbering[:,i])))

    return curve_lists, inv_lists

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
        super(NearestNeighbouring, self).__init__()
        self.size = size
        self.num_neigh = num_neigh
        self.weights = nn.Parameter(torch.ones(size, num_neigh) * initial_weight)
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, tensor_list):
        return torch.mul(tensor_list, self.weights).sum(-1) + self.bias


class SFC_CAE_structured_Encoder(nn.Module): 
  def __init__(self, input_size = 128**2, nearest_neighbouring = True, dims_latent = 16, space_filling_orderings = None, activaton = nn.ReLU()):
    '''
    Class contains the Encoder (snapshot -> latent).
    '''

    super(SFC_CAE_structured_Encoder, self).__init__()
    self.NN = nearest_neighbouring
    self.activate = activaton
    self.dims_latent = dims_latent
    self.input_size = input_size
    self.orderings = []
    self.sfc_plus = []
    self.sfc_minus = []
    self.sps = []
    self.sfc_nums = len(space_filling_orderings)
    for i in range(self.sfc_nums):
          self.orderings.append(space_filling_orderings[i])
          if nearest_neighbouring:
             self.sfc_plus.append(find_plus_neigh(space_filling_orderings[i]))
             self.sfc_minus.append(find_minus_neigh(space_filling_orderings[i]))
             self.sps.append(NearestNeighbouring(size = input_size, initial_weight= 1/3))
             self.register_parameter(name='sp%d_weights'%(i + 1), param=self.sps[i].weights)
             self.register_parameter(name='sp%d_bias'%(i + 1), param=self.sps[i].bias)

    self.c3 = nn.Conv1d(1, 2, kernel_size=32, stride= 4, padding=16) 
    self.c4 = nn.Conv1d(2, 4, kernel_size=32, stride= 4, padding=16)
    self.c5 = nn.Conv1d(4, 8, kernel_size=32, stride= 4, padding=16) 
    self.c6 = nn.Conv1d(8, 16, kernel_size=32, stride= 4, padding=16)
    self.fcs = []
    fc_in_num = (input_size // (4 **4) + 1)* 2**4 * self.sfc_nums
    out_neurons = 2 ** np.log2(fc_in_num).astype('int') // 4
    self.decoder_start = out_neurons * 4
    while (out_neurons >= dims_latent * 4):
        self.fcs.append(nn.Linear(fc_in_num, out_neurons))
        fc_in_num = out_neurons
        out_neurons //= 4
    self.fcs.append(nn.Linear(fc_in_num, dims_latent))
    self.fc_num = len(self.fcs)
    for i in range(self.fc_num):
        self.register_parameter(name='fc%d_weights'%(i + 1), param=self.fcs[i].weight)
        self.register_parameter(name='fc%d_bias'%(i + 1), param=self.fcs[i].bias)

  def get_concat_list(self, x, num_sfc):
      if x.is_cuda:
         return torch.cat((ordering_tensor(x, self.sfc_minus[num_sfc]).unsqueeze(-1).to(device), 
                           ordering_tensor(x, self.orderings[num_sfc]).unsqueeze(-1),
                           ordering_tensor(x, self.sfc_plus[num_sfc]).unsqueeze(-1).to(device)), -1)
      else:
         return torch.cat((ordering_tensor(x, self.sfc_minus[num_sfc]).unsqueeze(-1), 
                           ordering_tensor(x, self.orderings[num_sfc]).unsqueeze(-1),
                           ordering_tensor(x, self.sfc_plus[num_sfc]).unsqueeze(-1)), -1)


  def forward(self, x):  # Custom pytorch modules should follow this structure 
    '''
    x: [float] the block-advection snapshot
    '''
    xs = []
    for i in range(self.sfc_nums):
        if self.NN: 
            tensor_list = self.get_concat_list(x, i)
            xs.append(self.activate(self.sps[i](tensor_list)).unsqueeze(1))
        else:
            xs.append(util.ordering_tensor(x, self.orderings[i]).unsqueeze(1))
        # if self.NN: xs[i] = self.activate(self.sps[i](self.get_concat_list(xs[i], i)))
        # print(xs[i].shape)
        xs[i] = self.activate(self.c3(xs[i]))
        # print(xs[i].shape)
        xs[i] = self.activate(self.c4(xs[i])) 
        # print(xs[i].shape)
        xs[i] = self.activate(self.c5(xs[i]))
        # print(xs[i].shape)
        xs[i] = self.activate(self.c6(xs[i]))
        xs[i] = xs[i].view(-1, xs[i].size(1)*xs[i].size(2))
        # print(xs[i].shape)
    if self.NN: x = torch.cat(xs, -1)
    else: x = xs[0]
    # fully connect layers
    for i in range(len(self.fcs)): x = self.activate(self.fcs[i](x))
    return x

class SFC_CAE_structured_Decoder(nn.Module):
  def __init__(self, Encoder, invert_space_filling_orderings = None):
    '''
    Class contains the Decoder (latent -> snapshot).
    '''

    super(SFC_CAE_structured_Decoder, self).__init__()
    self.activate = nn.ReLU()
    self.NN = Encoder.NN
    self.input_size = Encoder.input_size
    self.activate = Encoder.activate
    self.orderings = []
    self.sfc_plus = []
    self.sfc_minus = []
    self.sps = []
    self.sfc_nums = len(invert_space_filling_orderings)
    self.sp_final = NearestNeighbouring(size = self.input_size, initial_weight= 1/self.sfc_nums, num_neigh= self.sfc_nums)
    # inverting fcs
    for i in range(self.sfc_nums):
          self.orderings.append(invert_space_filling_orderings[i])
          if self.NN:
             self.sfc_plus.append(find_plus_neigh(invert_space_filling_orderings[i]))
             self.sfc_minus.append(find_minus_neigh(invert_space_filling_orderings[i]))
             self.sps.append(NearestNeighbouring(size = self.input_size, initial_weight= 1/3))  
             self.register_parameter(name='sp%d_weights'%(i + 1), param=self.sps[i].weights)
             self.register_parameter(name='sp%d_bias'%(i + 1), param=self.sps[i].bias)
    self.fcs = []
    self.fcs.append(nn.Linear(Encoder.dims_latent, Encoder.fcs[-1].in_features))
    for i in range(2, Encoder.fc_num):
        self.fcs.append(nn.Linear(Encoder.fcs[-i].out_features, Encoder.fcs[-i].in_features))
    self.fcs.append(nn.Linear(Encoder.fcs[0].out_features, Encoder.decoder_start))
    self.avg_num = Encoder.decoder_start // self.sfc_nums

    for i in range(Encoder.fc_num):
        self.register_parameter(name='fc%d_weights'%(i + 1), param=self.fcs[i].weight)
        self.register_parameter(name='fc%d_bias'%(i + 1), param=self.fcs[i].bias)

     # convTranspose
    self.c13 = nn.ConvTranspose1d(16, 8, kernel_size=32, stride= 4, padding=14) 
    self.c14 = nn.ConvTranspose1d(8, 4, kernel_size=32, stride= 4, padding=14)
    self.c15 = nn.ConvTranspose1d(4, 2, kernel_size=32, stride= 4, padding=14) 
    self.c16 = nn.ConvTranspose1d(2, 1, kernel_size=32, stride= 4, padding=14)

  def get_concat_list(self, x, num_sfc):
      if x.is_cuda:
         return torch.cat((ordering_tensor(x, self.sfc_minus[num_sfc]).unsqueeze(-1).to(device), 
                           ordering_tensor(x, self.orderings[num_sfc]).unsqueeze(-1),
                           ordering_tensor(x, self.sfc_plus[num_sfc]).unsqueeze(-1).to(device)), -1)
      else:
         return torch.cat((ordering_tensor(x, self.sfc_minus[num_sfc]).unsqueeze(-1), 
                           ordering_tensor(x, self.orderings[num_sfc]).unsqueeze(-1),
                           ordering_tensor(x, self.sfc_plus[num_sfc]).unsqueeze(-1)), -1)

  def forward(self, z):
    '''
    z: [float] a sample from the latent
    '''
    for i in range(len(self.fcs)): z = self.activate(self.fcs[i](z))
    zs = []
    for i in range(self.sfc_nums):
        zs.append(z[:, i * self.avg_num: (i+1) * self.avg_num].unsqueeze(1))
        zs[i] = zs[i].view(-1, 16, self.avg_num // 16)
        # print(zs[i].shape)
        zs[i] = self.activate(self.c13(zs[i]))
        # print(zs[i].shape)
        zs[i] = self.activate(self.c14(zs[i]))
        # print(zs[i].shape) 
        zs[i] = self.activate(self.c15(zs[i]))
        # print(zs[i].shape)
        zs[i] = self.activate(self.c16(zs[i])).flatten(-1).squeeze(1)
        # print(zs[i].shape)
        if self.NN:
           tensor_list = self.get_concat_list(zs[i], i)
           zs[i] = self.activate(self.sps[i](tensor_list))
        else: zs[i] = ordering_tensor(x, self.orderings[num_sfc])
        if self.sfc_nums > 1: zs[i] = zs[i].unsqueeze(-1)
    if self.sfc_nums > 1: 
        z = torch.cat(zs, -1)
        z = self.activate(self.sp_final(z))
    else: z = zs[0]
    # print(z.shape)
    return z

class SFC_CAE_structured_Autoencoder(nn.Module):
  def __init__(self,
               size = 128 ** 2, 
               dims_latent = 16, 
               nearest_neighbouring = True,
               space_filling_orderings = None, 
               invert_space_filling_orderings = None,
               activation = nn.ReLU()):
    '''
    Class combines the Encoder and the Decoder with an Autoencoder latent space.

    dims_latent: [int] the dimension of (number of nodes in) the mean-field gaussian latent variable
    '''

    super(SFC_CAE_structured_Autoencoder, self).__init__()
    self.encoder = SFC_CAE_structured_Encoder(input_size = size, 
                                              dims_latent = dims_latent,
                                              nearest_neighbouring = nearest_neighbouring,
                                              space_filling_orderings = space_filling_orderings, 
                                              activaton = activation)
    self.decoder = SFC_CAE_structured_Decoder(self.encoder, invert_space_filling_orderings = invert_space_filling_orderings)


  def forward(self, x):
    '''
    x - [float] A batch of images from the data-loader
    '''

    z = self.encoder(x)
    z = self.decoder(z)
    return z  # Return the output of the decoder (1-D, the predicted image)












