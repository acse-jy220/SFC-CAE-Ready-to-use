"""
This module contains the of a space-filling convolutional autoencoder with multi-dimensional Conv/Sparse layers
Author: Jin Yu
Github handle: acse-jy220
"""

import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from .utils import *


###############################################################   Encoder Part ###################################################################

class SFC_CAE_Encoder_md(nn.Module): 
  def __init__(self, 
               input_size,
               dimension,
               components,
               structured,
               self_concat,
               nearest_neighbouring, 
               dims_latent, 
               space_filling_orderings,
               activation,
               variational,
               force_initialising_param=None,
               sfc_mapping_to_structured=None,
               **kwargs):
    '''
    Class contains the Encoder (snapshot -> latent).

    Input:
    ---
    input_size: [int] the number of Nodes in each snapshot.
    dimension: [int] the dimension of the problem, 2 for 2D and 3 for 3D.
    components: [int] the number of components we are compressing.
    structured: [bool] whether the mesh is structured or not.
    self_concat: [int] a channel copying operation, of which the input_channel of the 1D Conv Layers would be components * self_concat
    nearest_neighbouring: [bool] whether the sparse layers are added to the ANN or not.
    dims_latent: [int] the dimension of (number of nodes in) the mean-field gaussian latent variable
    space_filling_orderings: [list of 1D-arrays] the space-filling curves, of shape [number of curves, number of Nodes]
    activation: [torch.nn.functional] the activation function, ReLU() and Tanh() are usually used.
    variational: [bool] whether this is a variational autoencoder or not.
    force_initialising_param: [1d-array or 1d-list] a interval to initialize the parameters of the 1D Conv/TransConv, Fully-connected Layers, e.g. [a, b]

    Output:
    ---
    CASE -- (SFC-CAE-md): 
         x: the compressed latent variable, of shape (batch_size, dims_latent)

    CASE -- (SFC-VCAE-md):
         x: the compressed latent variable, of shape (batch_size, dims_latent)
         kl_div: the KL-divergence of the latent distribution to a standard Gaussian N(0, 1)
    '''

    super(SFC_CAE_Encoder_md, self).__init__()
    self.NN = nearest_neighbouring
    self.dims_latent = dims_latent
    self.dimension = dimension
    self.input_size = input_size
    self.components = components
    self.self_concat = self_concat
    self.input_channel = components * self_concat
    self.num_final_channels = 16
    self.variational = variational
    self.orderings = torch.tensor(space_filling_orderings).long()
    self.sfc_nums = len(space_filling_orderings)
    # self.NN_neighs = []
    self.num_neigh = 3
    # for i in range(self.sfc_nums):self.NN_neighs.append(get_neighbourhood_md(self.orderings[i], gen_neighbour_keys(1), ordering = True))
    self.NN_neigh_1d = get_neighbourhood_md(torch.arange(self.input_size).long(), gen_neighbour_keys(1), ordering = True)
    self.second_sfc = sfc_mapping_to_structured
    if 'direct_neigh' in kwargs.keys():
        self.direct_neigh = kwargs['direct_neigh']
    else: self.direct_neigh = False

    self.structured = structured
    if self.structured: 
       if activation is None:
          self.activate = nn.ReLU()
       else:
          self.activate = activation     
    else: 
       if activation is None:
          self.activate = nn.Tanh()
       else:
          self.activate = activation

    if  force_initialising_param is not None and len(force_initialising_param) != 2:
      raise ValueError("the input size of 'force_initialising_param' must be 2 !!!")
    else:
      self.init_param = force_initialising_param  

    if sfc_mapping_to_structured is not None:
         self.structured_size_input = self.second_sfc.shape[-1]
         self.diff_nodes = self.structured_size_input - self.input_size
         self.structured_size = np.round(np.power(self.structured_size_input, (1/self.dimension))).astype('int')
         self.shape = (self.structured_size,) * self.dimension
        #  self.num_neigh_md = 3 ** self.dimension

    if sfc_mapping_to_structured is None:
       
      if dimension == 2: 
        self.kernel_size = 32
        self.stride = 4
        self.increase_multi = 2
      elif dimension == 3:
        self.kernel_size = 176
        self.stride = 8
        self.increase_multi = 4

      self.padding = self.kernel_size//2

      self.shape = (self.input_size,)

      # find size of convolutional layers and fully-connected layers, see the funtion 'find_size_conv_layers_and_fc_layers()' in utils.py
      self.conv_size, self.size_conv, self.size_fc, self.channels, self.inv_conv_start, self.output_paddings \
      = find_size_conv_layers_and_fc_layers(self.input_size, self.kernel_size, self.padding, self.stride, self.dims_latent, self.sfc_nums, self.input_channel, self.increase_multi,  self.num_final_channels)
    
    elif sfc_mapping_to_structured is not None: 
         if 'kernel_size' in kwargs.keys():
            self.kernel_size = kwargs['kernel_size']
         else: self.kernel_size = 5

         if 'stride' in kwargs.keys():
                self.stride = kwargs['stride']
         else: self.stride = 2

         if 'padding' in kwargs.keys():
                self.padding = kwargs['padding']
         else: self.padding = 2

         if 'increase_multi' in kwargs.keys():
                self.increase_multi = kwargs['increase_multi']
         else: self.increase_multi = 4

        #  if dimension == 2: 
        #    self.increase_multi = 2
        #  elif dimension == 3:
        #    self.increase_multi = 4      

         # find size of convolutional layers and fully-connected layers, see the funtion 'find_size_conv_layers_and_fc_layers()' in utils.py
         self.conv_size, self.size_conv, self.size_fc, self.channels, self.inv_conv_start, self.output_paddings \
         = find_size_conv_layers_and_fc_layers(self.structured_size, self.kernel_size, self.padding, self.stride, self.dims_latent, self.sfc_nums, self.input_channel, self.increase_multi,  self.num_final_channels, self.dimension)
         
         self.Ax = gen_neighbour_keys(ndim=self.dimension, direct_neigh=self.direct_neigh)
        #  self.neigh_md = get_neighbourhood_md(self.second_sfc.reshape(self.shape), self.Ax, ordering = True)
         self.num_neigh_md = len(self.Ax) + 1
         self.neigh_md = get_neighbourhood_md((torch.arange(self.structured_size_input).long()).reshape(self.shape), self.Ax, ordering = True)

         # parameters for expand snapshots
         self.expand_paras = gen_filling_paras(self.input_size, self.structured_size_input)

    # set up convolutional layers, fully-connected layers and sparse layers
    self.fcs = []
    self.convs = []

    #If NN, add a sparse layer 
    if self.NN: self.sps = []
    for i in range(self.sfc_nums):
       self.convs.append([])
       for j in range(self.size_conv):
           if sfc_mapping_to_structured is None: 
              self.convs[i].append(nn.Conv1d(self.channels[j], self.channels[j+1], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
           else:
              if self.dimension == 2:
                  self.convs[i].append(nn.Conv2d(self.channels[j], self.channels[j+1], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
              elif self.dimension == 3:
                  self.convs[i].append(nn.Conv3d(self.channels[j], self.channels[j+1], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
           if self.init_param is not None: 
              self.convs[i][j].weight.data.uniform_(self.init_param[0], self.init_param[1])
              self.convs[i][j].bias.data.fill_(0.001)
       self.convs[i] = nn.ModuleList(self.convs[i])
       if self.NN:
        #   if sfc_mapping_to_structured is None:
        #     self.sps.append(NearestNeighbouring(size = self.input_size * self.input_channel, initial_weight= (1/3), num_neigh = 3))
        #   else:
            self.sps.append(NearestNeighbouring_md(shape = self.shape, initial_weight= None, num_neigh_md = self.num_neigh_md)) 
    self.convs = nn.ModuleList(self.convs)
    if self.NN: self.sps = nn.ModuleList(self.sps)
    for i in range(len(self.size_fc) - 2):
       self.fcs.append(nn.Linear(self.size_fc[i], self.size_fc[i+1]))
       if self.init_param is not None: 
            self.fcs[i].weight.data.uniform_(self.init_param[0], self.init_param[1])
            self.fcs[i].bias.data.fill_(0.001)
    
    if self.variational:
       self.layerMu = nn.Linear(self.size_fc[-2], self.size_fc[-1])
       self.layerSig = nn.Linear(self.size_fc[-2], self.size_fc[-1])
       self.Normal01 = torch.distributions.Normal(0, 1)
       if self.init_param is not None: 
            self.layerMu.weight.data.uniform_(self.init_param[0], self.init_param[1])
            self.layerMu.bias.data.fill_(0.001)
            self.layerSig.weight.data.uniform_(self.init_param[0], self.init_param[1])
            self.layerSig.bias.data.fill_(0.001)
    else:
       self.fcs.append(nn.Linear(self.size_fc[-2], self.size_fc[-1]))
       if self.init_param is not None: 
            self.fcs[-1].weight.data.uniform_(self.init_param[0], self.init_param[1])
            self.fcs[-1].bias.data.fill_(0.001)
    self.fcs = nn.ModuleList(self.fcs)

  def forward(self, x):
    '''
    x: [float] the fluid data snapshot, could have multiple components, but 
    the last dimension should always represent the component index.
    '''
    xs = []
    # if self.components > 1: 
    #     x = x.permute(0, -1, -2)
    #     x = x.reshape(-1, x.shape[-2] * x.shape[-1])
    if self.self_concat > 1: 
        if x.ndim == 2: x = x.unsqueeze(1)
        x = torch.cat([x] * self.self_concat, 1)
    # print(x.shape)
    # 1D or MD Conv Layers
    for i in range(self.sfc_nums):
        a = x[..., self.orderings[i]]
        # print(a.shape)
        # a = ordering_tensor(x, self.orderings[i]) 
        if self.second_sfc is not None: 
            a = expand_snapshot_backward_connect(a, *self.expand_paras)
            a = a[..., self.second_sfc]
            if self.NN:
               tt_list = get_concat_list_md(a, self.neigh_md, self.num_neigh_md)
            #    print(tt_list.shape)
               tt_nn = self.sps[i](tt_list)
               a = self.activate(tt_nn)
               del tt_list
               del tt_nn
            a = a.reshape(a.shape[:-1] + self.shape)
        else: 
            if self.NN:
               tt_list = get_concat_list_md(a, self.NN_neigh_1d, self.num_neigh)
               tt_nn = self.sps[i](tt_list)
               a = self.activate(tt_nn)
               del tt_list
               del tt_nn   
            # a = a.reshape((a.shape[0], self.input_channel, self.input_size)) 
        # if self.input_channel > 1: a = a.view(-1, self.input_channel, self.input_size)
        # else: a = a.unsqueeze(1)
        for j in range(self.size_conv):
            a = self.activate(self.convs[i][j](a))
        # xs.append(a.view(-1, a.size(1)*a.size(2)))
        a = a.reshape(a.shape[0], -1)
        xs.append(a)
        # print(a.shape)
        del a
    del x
    if self.sfc_nums > 1: x = torch.cat(xs, 1)
    else: x = xs[0]
    # x = x.reshape(x.shape[0], -1)
    for i in range(self.sfc_nums): del xs[0] # clear memory 

    # fully connect layers
    for i in range(len(self.fcs)): x = self.activate(self.fcs[i](x))

    # variational sampling
    if self.variational:
      mu = self.layerMu(x)
      sigma = torch.exp(self.layerSig(x))
      sample = self.Normal01.sample(mu.shape).to(x.device)
      x = mu + sigma * sample
      kl_div = ((sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()) / (mu.shape[0] * self.input_size * self.components)
      return x, kl_div
    else: return x

###############################################################   Decoder Part ###################################################################

class SFC_CAE_Decoder_md(nn.Module): 
  def __init__(self, encoder, inv_space_filling_orderings, output_linear = False, reduce_strategy = 'truncate'):
    '''
    Class contains the Decoder for SFC_CAE (latent -> reconstructed snapshot).

    Input:
    ---
    encoder: [SFC_CAE_Encoder object] the SFC_CAE_Encoder class, we want to nearly 'invert' the operation, so we just inherit most parameters from the Encoder.
    inv_space_filling_orderings: [list of 1D-arrays] the inverse space-filling curves, of shape [number of curves, number of Nodes]
    output_linear: [bool] default is false, if turned on, a linear activation will be applied at the output.

    Output:
    ---
    CASE -- (SFC-CAE-md): 
         z: the reconstructed batch of snapshots, in 1D, of shape (batch_size, number of Nodes, number of components)

    CASE -- (SFC-VCAE-md):
         z: the reconstructed batch of snapshots, in 1D, of shape (batch_size, number of Nodes, number of components)
         kl_div: the KL-divergence of the latent distribution to a standard Gaussian N(0, 1)
    '''

    super(SFC_CAE_Decoder_md, self).__init__()

    # pass parameters from the encoder
    self.NN = encoder.NN
    self.variational = encoder.variational
    self.activate = encoder.activate
    self.dims_latent = encoder.dims_latent
    self.dimension = encoder.dimension
    self.kernel_size = encoder.kernel_size
    self.stride = encoder.stride
    self.padding = encoder.padding
    self.increase_multi = encoder.increase_multi
    self.input_size = encoder.input_size
    self.components = encoder.components
    self.self_concat = encoder.self_concat
    self.num_final_channels = encoder.num_final_channels
    self.output_linear = output_linear
    self.size_conv = encoder.size_conv
    self.inv_conv_start = encoder.inv_conv_start
    self.input_channel = self.components * self.self_concat
    self.sfc_nums = encoder.sfc_nums
    self.orderings = torch.tensor(inv_space_filling_orderings).long()
    self.shape = encoder.shape

    self.reduce = reduce_strategy

    # self.NN_neighs = []
    self.num_neigh = encoder.num_neigh
    # for i in range(self.sfc_nums):self.NN_neighs.append(get_neighbourhood_md(self.orderings[i], gen_neighbour_keys(1), ordering = True))
    self.NN_neigh_1d = encoder.NN_neigh_1d

    # md Decoder
    if encoder.second_sfc is None: 
        self.inv_second_sfc = None
        self.init_convTrans_shape = (encoder.num_final_channels, ) + (encoder.conv_size[-1], )
    else: 
        self.inv_second_sfc = np.argsort(encoder.second_sfc)  
        self.structured_size_input = self.inv_second_sfc.shape[-1]
        self.diff_nodes = encoder.diff_nodes
        self.structured_size = encoder.structured_size
        self.shape = encoder.shape
        self.num_neigh_md = encoder.num_neigh_md   
        self.neigh_md = encoder.neigh_md   
        self.init_convTrans_shape = (encoder.num_final_channels, ) + (encoder.conv_size[-1], ) * self.dimension
        self.expand_paras = encoder.expand_paras
    self.fcs = []
    # set up fully-connected layers
    for k in range(1, len(encoder.size_fc)):
       self.fcs.append(nn.Linear(encoder.size_fc[-k], encoder.size_fc[-k-1]))
       if encoder.init_param is not None: 
            self.fcs[k - 1].weight.data.uniform_(encoder.init_param[0], encoder.init_param[1])
            self.fcs[k - 1].bias.data.fill_(0.001) 
    self.fcs = nn.ModuleList(self.fcs)

    # set up convolutional layers, fully-connected layers and sparse layers
    self.convTrans = []
    self.sps = []
    for i in range(self.sfc_nums):
       self.convTrans.append([])
       for j in range(1, encoder.size_conv + 1):
           if encoder.second_sfc is None:
              self.convTrans[i].append(nn.ConvTranspose1d(encoder.channels[-j], encoder.channels[-j-1], kernel_size=self.kernel_size, stride=self.stride, padding=encoder.padding, output_padding = encoder.output_paddings[j - 1]))
           else:
              if self.dimension == 2:
                  self.convTrans[i].append(nn.ConvTranspose2d(encoder.channels[-j], encoder.channels[-j-1], kernel_size=self.kernel_size, stride=self.stride, padding=encoder.padding, output_padding = encoder.output_paddings[j - 1]))
              elif self.dimension == 3:
                  self.convTrans[i].append(nn.ConvTranspose3d(encoder.channels[-j], encoder.channels[-j-1], kernel_size=self.kernel_size, stride=self.stride, padding=encoder.padding, output_padding = encoder.output_paddings[j - 1]))               
           if encoder.init_param is not None: 
              self.convTrans[i][j - 1].weight.data.uniform_(encoder.init_param[0], encoder.init_param[1])
              self.convTrans[i][j - 1].bias.data.fill_(0.001)       
       self.convTrans[i] = nn.ModuleList(self.convTrans[i])
       if self.NN:
        #   if encoder.second_sfc is None:
        #     self.sps.append(NearestNeighbouring(size = self.input_size * self.input_channel, initial_weight= (1/3), num_neigh = 3))
        #   else:
            self.sps.append(NearestNeighbouring_md(self.shape, None, self.num_neigh_md, self.self_concat)) 

    self.convTrans = nn.ModuleList(self.convTrans)
    self.sps = nn.ModuleList(self.sps)         

    self.split = encoder.size_fc[0] // self.sfc_nums

    # final sparse layer combining SFC outputs, those two approaches are not as good as the simple [tensor_list].sum(-1)
    # if self.sfc_nums > 1: self.final_sp = nn.Parameter(torch.ones(self.sfc_nums) / self.sfc_nums)
    # if self.sfc_nums > 1: self.final_sp = NearestNeighbouring(size = self.input_size * self.components, initial_weight= 1 / self.sfc_nums, num_neigh = self.sfc_nums)

    # final linear activate (shut down it if you have standardlized your data first)
   #  if output_linear:
   #    self.out_linear_weights = []
   #    self.out_linear_bias = []
   #    for i in range(self.components):
   #        self.out_linear_weights.append(nn.Parameter(torch.ones(self.input_size)))
   #        self.out_linear_bias.append(nn.Parameter(torch.zeros(self.input_size)))
   #    self.out_linear_weights = nn.ModuleList(self.out_linear_weights)
   #    self.out_linear_bias = nn.ModuleList(self.out_linear_bias)
    if output_linear:
      self.out_linear_weights = nn.Parameter(torch.ones(self.components))
      self.out_linear_bias = nn.Parameter(torch.zeros(self.components))

  def forward(self, x):  # Custom pytorch modules should follow this structure 
    '''
    z: [float] the fluid data snapshot, could have multiple components, but 
    the last dimension should always represent the component index.
    '''   

    for i in range(len(self.fcs)):
        x = self.activate(self.fcs[i](x))
    # revert torch.cat
    if self.sfc_nums > 1: x = torch.chunk(x, chunks=self.sfc_nums, dim=1)
    zs = []
    for i in range(self.sfc_nums):
        # if self.inv_second_sfc is not None: 
        b = x[i].reshape((x[i].shape[0],) + self.init_convTrans_shape)
        # else: 
        #     b = x[..., i].view(-1, self.num_final_channels, self.inv_conv_start)
        for j in range(self.size_conv):
            b = self.activate(self.convTrans[i][j](b))
        if self.inv_second_sfc is not None: 
            b = b.reshape(b.shape[:2] + (self.structured_size_input, ))
            b = b[..., self.inv_second_sfc]
            if self.NN:
               tt_list = get_concat_list_md(b, self.neigh_md, self.num_neigh_md, self.self_concat)
               tt_nn = self.sps[i](tt_list)
               b = self.activate(tt_nn)
               del tt_list 
               del tt_nn  
            b = reduce_expanded_snapshot(b, self.input_size, *self.expand_paras, scheme=self.reduce) # truncate or mean
            # b = b[..., :self.input_size] # simple truncate
            b = b[..., self.orderings[i]] # backward order refer to first sfc(s).         
        else: 
            # b = b[..., self.orderings[i]] # backward order refer to first sfc(s).
            # b = b.reshape(b.shape[:2] + (self.input_size, ))
            if self.NN:
               tt_list = get_concat_list_md(b, self.NN_neigh_1d, self.num_neigh, self.self_concat)
               tt_nn = self.sps[i](tt_list)
               b = self.activate(tt_nn)
               del tt_list
               del tt_nn
            else: 
              if self.self_concat > 1: b = sum(torch.chunk(b, chunks=self.self_concat, dim=1))
            b = b[..., self.orderings[i]] # backward order refer to first sfc(s).
        # if self.self_concat > 1:
        #    b = sum(torch.chunk(b, chunks=self.self_concat, dim=1))
        zs.append(b.unsqueeze(-1))
    z = torch.cat(zs, -1).sum(-1)
    # if self.inv_second_sfc is not None: return z[..., :self.input_size]
    # else: 
    return self.activate(z)

###############################################################   AutoEncoder Wrapper ###################################################################

class SFC_CAE_md(nn.Module):
  def __init__(self, 
               input_size,
               dimension,
               components,
               structured,
               self_concat,
               nearest_neighbouring, 
               dims_latent, 
               space_filling_orderings,
               inv_space_filling_orderings,
               activation,
               variational,
               force_initialising_param=None,
               sfc_mapping_to_structured=None,
               output_linear = False,
               reduce_strategy = 'truncate',
               **kwargs):
    '''
    SFC_CAE Class combines the SFC_CAE_Encoder and the SFC_CAE_Decoder with an Autoencoder latent space.

    Input:
    ---
    size: [int] the number of Nodes in each snapshot.
    dimension: [int] the dimension of the problem, 2 for 2D and 3 for 3D.
    components: [int] the number of components we are compressing.
    structured: [bool] whether the mesh is structured or not.
    self_concat: [int] a channel copying operation, of which the input_channel of the 1D Conv Layers would be components * self_concat
    nearest_neighbouring: [bool] whether the sparse layers are added to the ANN or not.
    dims_latent: [int] the dimension of (number of nodes in) the mean-field gaussian latent variable
    space_filling_orderings: [list of 1D-arrays] the space-filling curves, of shape [number of curves, number of Nodes]
    invert_space_filling_orderings: [list of 1D-arrays] the inverse space-filling curves, of shape [number of curves, number of Nodes]
    activation: [torch.nn.functional] the activation function, ReLU() and Tanh() are usually used.
    variational: [bool] whether this is a variational autoencoder or not.
    force_initialising_param: [1d-array or 1d-list] a interval to initialize the parameters of the 1D Conv/TransConv, Fully-connected Layers, e.g. [a, b]
    output_linear: [bool] default is false, if turned on, a linear activation will be applied at the output.

    Output:
    ---
    CASE -- (SFC-CAE-md): 
         self.decoder(z): the reconstructed batch of snapshots, in 1D, of shape (batch_size, number of Nodes, number of components)

    CASE -- (SFC-VCAE-md):
         self.decoder(z): the reconstructed batch of snapshots, in 1D, of shape (batch_size, number of Nodes, number of components)
         kl_div: the KL-divergence of the latent distribution to a standard Gaussian N(0, 1)
    '''

    super(SFC_CAE_md, self).__init__()
    self.encoder = SFC_CAE_Encoder_md(input_size,
               dimension,
               components,
               structured,
               self_concat,
               nearest_neighbouring, 
               dims_latent, 
               space_filling_orderings,
               activation,
               variational,
               force_initialising_param,
               sfc_mapping_to_structured,
               **kwargs)
    self.decoder = SFC_CAE_Decoder_md(self.encoder, inv_space_filling_orderings, output_linear, reduce_strategy)
   
    # specify name of the activation
    if isinstance(self.encoder.activate, type(nn.ReLU())):
      self.activate = 'ReLU'
    elif isinstance(self.encoder.activate, type(nn.Tanh())):
      self.activate = 'Tanh'
    elif isinstance(self.encoder.activate, type(nn.SELU())):
      self.activate = 'SELU'

  def forward(self, x):
   '''
   x - [Torch.Tensor.float] A batch of fluid snapshots from the data-loader
   '''
   # return value for VAE 
   if self.encoder.variational:
      z, kl_div = self.encoder(x) # encoder, compress each image to 1-D data of size {dims_latent}, as well as record the KL divergence.
      return self.decoder(z), kl_div
   # return value for normal AE
   else:
      z = self.encoder(x) # encoder, compress each image to 1-D data of size {dims_latent}.
      return self.decoder(z)  # Return the output of the decoder (1-D, the predicted image)

      