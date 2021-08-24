"""
This module contains the main class of a space-filling convolutional autoencoder.
Author: Jin Yu
Github handle: acse-jy220
"""

import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from sfc_cae.utils import *



###############################################################   Encoder Part ###################################################################

class SFC_CAE_Encoder(nn.Module): 
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
               force_initialising_param):
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
    CASE -- (SFC-CAE): 
         x: the compressed latent variable, of shape (dims_latent, )

    CASE-- (SFC-VCAE):
         x: the compressed latent variable, of shape (dims_latent, )
         kl_div: the KL-divergence of the latent distribution to a standard Gaussian N(0, 1)
    '''

    super(SFC_CAE_Encoder, self).__init__()
    self.NN = nearest_neighbouring
    self.dims_latent = dims_latent
    self.dimension = dimension
    self.input_size = input_size
    self.components = components
    self.self_concat = self_concat
    self.input_channel = components * self_concat
    self.variational = variational
    self.orderings = []
    self.sfc_plus = []
    self.sfc_minus = []
    self.sfc_nums = len(space_filling_orderings)
    if  force_initialising_param is not None and len(force_initialising_param) != 2:
      raise ValueError("the input size of 'force_initialising_param' must be 2 !!!")
    else:
      self.init_param = force_initialising_param

    for i in range(self.sfc_nums):
        if self.input_channel > 1:
           self.orderings.append(expend_SFC_NUM(space_filling_orderings[i], self.input_channel))
           if self.NN:
              self.sfc_plus.append(expend_SFC_NUM(find_plus_neigh(space_filling_orderings[i]), self.input_channel))
              self.sfc_minus.append(expend_SFC_NUM(find_minus_neigh(space_filling_orderings[i]), self.input_channel))
        else:
           self.orderings.append(space_filling_orderings[i])
           self.sfc_plus.append(find_plus_neigh(space_filling_orderings[i]))
           self.sfc_minus.append(find_minus_neigh(space_filling_orderings[i]))          
       

    if dimension == 2: 
        self.kernel_size = 32
        self.stride = 4
        self.increase_multi = 2
    elif dimension == 3:
        self.kernel_size = 176
        self.stride = 8
        self.increase_multi = 4

    self.padding = self.kernel_size//2
    self.num_final_channels = 16

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
    
    # find size of convolutional layers and fully-connected layers, see the funtion 'find_size_conv_layers_and_fc_layers()' in utils.py
    self.conv_size, self.size_conv, self.size_fc, self.channels, self.inv_conv_start, self.output_paddings \
    = find_size_conv_layers_and_fc_layers(self.input_size, self.kernel_size, self.padding, self.stride, self.dims_latent, self.sfc_nums, self.input_channel, self.increase_multi,  self.num_final_channels)
    
    # set up convolutional layers, fully-connected layers and sparse layers
    self.fcs = []
    self.convs = []

    #If NN, add a sparse layer 
    if self.NN: self.sps = []
    for i in range(self.sfc_nums):
       self.convs.append([])
       for j in range(self.size_conv):
           self.convs[i].append(nn.Conv1d(self.channels[j], self.channels[j+1], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
           if self.init_param is not None: 
              self.convs[i][j].weight.data.uniform_(self.init_param[0], self.init_param[1])
              self.convs[i][j].bias.data.fill_(0.001)
       self.convs[i] = nn.ModuleList(self.convs[i])
       if self.NN:
          self.sps.append(NearestNeighbouring(size = self.input_size * self.input_channel, initial_weight= (1/3), num_neigh = 3))
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

  def get_concat_list(self, x, num_sfc):
      self_t = ordering_tensor(x, self.orderings[num_sfc]).unsqueeze(-1)
      minus_neigh = ordering_tensor(x, self.sfc_minus[num_sfc]).unsqueeze(-1)
      plus_neigh = ordering_tensor(x, self.sfc_plus[num_sfc]).unsqueeze(-1)
      tensor_list = torch.cat((minus_neigh, self_t, plus_neigh), -1)
      del self_t
      del minus_neigh
      del plus_neigh
      return tensor_list


  def forward(self, x):
    '''
    x: [float] the fluid data snapshot, could have multiple components, but 
    the last dimension should always represent the component index.
    '''
    xs = []
    if self.components > 1: 
        x = x.permute(0, -1, -2)
        x = x.reshape(-1, x.shape[-2] * x.shape[-1])
    if self.self_concat > 1: x = torch.cat([x] * self.self_concat, -1)
    
    # 1D Conv Layers
    for i in range(self.sfc_nums):
        if self.NN:
           tt_list = self.get_concat_list(x, i)
           tt_nn = self.sps[i](tt_list)
           a = self.activate(tt_nn)
           del tt_list
           del tt_nn
        else:
           a = ordering_tensor(x, self.orderings[i])
        if self.input_channel > 1: a = a.view(-1, self.input_channel, self.input_size)
        else: a = a.unsqueeze(1)
        for j in range(self.size_conv):
            a = self.activate(self.convs[i][j](a))
        xs.append(a.view(-1, a.size(1)*a.size(2)))
        del a
    del x
    if self.sfc_nums > 1: x = torch.cat(xs, -1)
    else: x = xs[0]
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


class SFC_CAE_Decoder(nn.Module): 
  def __init__(self, encoder, inv_space_filling_orderings, output_linear = False):
    '''
    Class contains the Decoder for SFC_CAE (latent -> reconstructed snapshot).

    Input:
    ---
    encoder: [SFC_CAE_Encoder object] the SFC_CAE_Encoder class, we want to nearly 'invert' the operation, so we just inherit most parameters from the Encoder.
    inv_space_filling_orderings: [list of 1D-arrays] the inverse space-filling curves, of shape [number of curves, number of Nodes]
    output_linear: [bool] default is false, if turned on, a linear activation will be applied at the output.

    Output:
    ---
    CASE -- (SFC-CAE): 
         z: the reconstructed batch of snapshots, in 1D, of shape (batch_size, number of Nodes, number of components)

    CASE-- (SFC-VCAE):
         z: the reconstructed batch of snapshots, in 1D, of shape (batch_size, number of Nodes, number of components)
         kl_div: the KL-divergence of the latent distribution to a standard Gaussian N(0, 1)
    '''

    super(SFC_CAE_Decoder, self).__init__()

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

    # initialize sfc_orderings/ neighbours
    self.orderings = []
    self.sfc_plus = []
    self.sfc_minus = []
    self.sfc_nums = len(inv_space_filling_orderings)
    
    # expend sfc_indexs
    for i in range(self.sfc_nums):
        if self.input_channel > 1:
           self.orderings.append(expend_SFC_NUM(inv_space_filling_orderings[i], self.input_channel))
           if self.NN:
              self.sfc_plus.append(expend_SFC_NUM(find_plus_neigh(inv_space_filling_orderings[i]), self.input_channel))
              self.sfc_minus.append(expend_SFC_NUM(find_minus_neigh(inv_space_filling_orderings[i]), self.input_channel))
        else:
           self.orderings.append(inv_space_filling_orderings[i])
           self.sfc_plus.append(find_plus_neigh(inv_space_filling_orderings[i]))
           self.sfc_minus.append(find_minus_neigh(inv_space_filling_orderings[i]))          
    
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
           self.convTrans[i].append(nn.ConvTranspose1d(encoder.channels[-j], encoder.channels[-j-1], kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size//2, output_padding = encoder.output_paddings[j - 1]))
           if encoder.init_param is not None: 
              self.convTrans[i][j - 1].weight.data.uniform_(encoder.init_param[0], encoder.init_param[1])
              self.convTrans[i][j - 1].bias.data.fill_(0.001)       
       self.convTrans[i] = nn.ModuleList(self.convTrans[i])
       if self.NN:
          self.sps.append(NearestNeighbouring(size = self.input_size * self.components, initial_weight= (1/3) / self.self_concat, num_neigh = 3 * self.self_concat))  
       else:
          if self.self_concat > 1:
             self.sps.append(NearestNeighbouring(size = self.input_size * self.components, initial_weight= 1 / self.self_concat, num_neigh = self.self_concat))

    self.convTrans = nn.ModuleList(self.convTrans)
    self.sps = nn.ModuleList(self.sps)         

    self.split = encoder.size_fc[0] // self.sfc_nums

    # final sparse layer combining SFC outputs, those two approaches are not as good as the simple [tensor_list].sum(-1)
    # if self.sfc_nums > 1: self.final_sp = nn.Parameter(torch.ones(self.sfc_nums) / self.sfc_nums)
    # if self.sfc_nums > 1: self.final_sp = NearestNeighbouring(size = self.input_size * self.components, initial_weight= 1 / self.sfc_nums, num_neigh = self.sfc_nums)

    # final linear activate (shut down it if you have standardlized your data first)
    if output_linear:
      self.out_linear_weights = []
      self.out_linear_bias = []
      for i in range(self.components):
          self.out_linear_weights.append(nn.Parameter(torch.ones(self.input_size)))
          self.out_linear_bias.append(nn.Parameter(torch.zeros(self.input_size)))
      self.out_linear_weights = nn.ModuleList(self.out_linear_weights)
      self.out_linear_bias = nn.ModuleList(self.out_linear_bias)
      

  def get_concat_list(self, x, num_sfc):
    if self.self_concat > 1:
       return torch.cat((ordering_tensor(x, self.sfc_minus[num_sfc]).view(-1, self.self_concat, self.input_size * self.components).permute(0, -1, -2), 
                           ordering_tensor(x, self.orderings[num_sfc]).view(-1, self.self_concat, self.input_size * self.components).permute(0, -1, -2),
                           ordering_tensor(x, self.sfc_plus[num_sfc]).view(-1, self.self_concat, self.input_size * self.components).permute(0, -1, -2)), -1)
    else:
      self_t = ordering_tensor(x, self.orderings[num_sfc]).unsqueeze(-1)
      minus_neigh = ordering_tensor(x, self.sfc_minus[num_sfc]).unsqueeze(-1)
      plus_neigh = ordering_tensor(x, self.sfc_plus[num_sfc]).unsqueeze(-1)
      tensor_list = torch.cat((minus_neigh, self_t, plus_neigh), -1)
      del self_t
      del minus_neigh
      del plus_neigh
      return tensor_list

  def forward(self, x):  # Custom pytorch modules should follow this structure 
    '''
    z: [float] the fluid data snapshot, could have multiple components, but 
    the last dimension should always represent the component index.
    '''
    for i in range(len(self.fcs)):
        x = self.activate(self.fcs[i](x))
    
    x = x.view(-1, self.split, self.sfc_nums)
    zs = []
    for i in range(self.sfc_nums):
        b = x[..., i].view(-1, self.num_final_channels, self.inv_conv_start)
        for j in range(self.size_conv):
            b = self.activate(self.convTrans[i][j](b))
        b = b.view(-1, self.input_size * self.input_channel)
        if self.NN:
           tt_list = self.get_concat_list(b, i)
           tt_nn = self.sps[i](tt_list)
           b = self.activate(tt_nn)
           del tt_list
           del tt_nn
        else:
           if self.self_concat > 1:
              b = ordering_tensor(b, self.orderings[i]).view(-1, self.self_concat, self.components * self.input_size).permute(0, -1, -2)
              b = self.activate(self.sps[i](b))
           else:
              b = ordering_tensor(b, self.orderings[i])
        zs.append(b.unsqueeze(-1))
        del b
    del x
    if self.sfc_nums > 1: 
        tt_list = torch.cat(zs, -1).sum(-1)
      #   f_nn = self.final_sp(tt_list)
        # tt_list *= self.final_sp
      #  tt_list = tt_list.sum(-1)
      #   del tt_list
        z = self.activate(tt_list)
      #   del f_nn
        del tt_list
    else: z = zs[0].squeeze(-1)
    for i in range(self.sfc_nums): del zs[0]
    if self.components > 1: 
        z = z.view(-1, self.components, self.input_size).permute(0, -1, -2)
        if self.output_linear: 
            ts = [] 
            for i in range(self.components):
                t = z[..., i]
                t = t * self.out_linear_weights[i] + self.out_linear_bias[i]
                ts.append(t.unsqueeze(-1))
            z = torch.cat(ts, -1)
            del ts
        return z
    else: 
        if self.output_linear:
            z = self.out_linear_weights[0] * z + self.out_linear_bias[0]
        return z


###############################################################   AutoEncoder Wrapper ###################################################################

class SFC_CAE(nn.Module):
  def __init__(self,
               size,
               dimension,
               components,
               structured,
               self_concat,
               nearest_neighbouring,
               dims_latent,
               space_filling_orderings, 
               invert_space_filling_orderings,
               activation = None,
               variational = False,
               force_initialising_param = None,
               output_linear = False):
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
    CASE -- (SFC-CAE): 
         self.decoder(z): the reconstructed batch of snapshots, in 1D, of shape (batch_size, number of Nodes, number of components)

    CASE-- (SFC-VCAE):
         self.decoder(z): the reconstructed batch of snapshots, in 1D, of shape (batch_size, number of Nodes, number of components)
         kl_div: the KL-divergence of the latent distribution to a standard Gaussian N(0, 1)
    '''

    super(SFC_CAE, self).__init__()
    self.encoder = SFC_CAE_Encoder(size,
                          dimension,
                          components,
                          structured,
                          self_concat,
                          nearest_neighbouring, 
                          dims_latent, 
                          space_filling_orderings,
                          activation,
                          variational,
                          force_initialising_param)
    self.decoder = SFC_CAE_Decoder(self.encoder, invert_space_filling_orderings, output_linear)

  def output_structure(self):
    '''
    This function is a automated LaTeX table generater for the autoencoder.

    Usage:
    ---
    Once the SFC_CAE object is initialized, you could simply type 'autoencoder.parameters' to see the layers in Python, 
    Or you can type 'autoencoder.output_structure()', which calls this function, and a table in LaTeX format will be writed in 'LatexTable.txt'.
    '''
    with open('LatexTable.txt', 'w') as f:
      f.write('\\begin{table}[!htbp]\n')
      f.write('\\resizebox{\\columnwidth}{!}{%\n')
      f.write('\\small%\n')
      f.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}\n')
      f.write('\\hline\n')
      f.write('layers & input size \\& ordering & kernel size & channels & stride & padding & output padding & output size \\& ordering & activation\\\\\n')
      f.write('\\hline\n')
    
      # initialize numbers inside the cells
      size = self.encoder.input_size
      components = self.encoder.components
      sfc_set = '\{1'
      for i in range(1, self.encoder.sfc_nums):
         sfc_set += F', {i + 1}'
      sfc_set += '\}'
    
      components_set = '\{1'
      for i in range(1, self.encoder.components):
          components_set += F', {i + 1}'
      components_set += '\}'
    
      if isinstance(self.encoder.activate, type(nn.ReLU())):
         activate = 'ReLU'
      elif isinstance(self.encoder.activate, type(nn.Tanh())):
         activate = 'Tanh'
    
      if self.encoder.structured:
         cell_11 = '1-Grid'
         type_m = 'Grid'
         cell_end = '-Grid'
      else: 
         cell_11 = '1-FEM (ANN input)'
         cell_end = '-FEM (ANN output)'
         type_m = 'FEM'
    
      first_line = F'{cell_11} & ({size}, {components}, {type_m}) & 1 Identity & {components} & 1 & 0 & 0 & ({size}, {components}, SFC$\\mathcal{{C}}$)  $\\forall \\; \\mathcal{{C}} \\in {sfc_set}$ & Identity\\\\\n'
      f.write(first_line)
      f.write('\\hline\n')
    

      f.write('\\multicolumn{9}{|c|}{\\textbf{Encoder}}\\\\\n')
      f.write('\\hline\n')
    
      layer_count = 1
    
      if self.encoder.NN:
        if self.encoder.self_concat > 1:
            f.write(F'\\multicolumn{{9}}{{|c|}}{{Copying channels and concatenate at the 2nd dimension to form ({size}, {components * self.encoder.self_concat}, SFC$\\mathcal{{C}}$), flatten it to 1D.}} \\\\\n')     
        else:
            f.write(F'\\multicolumn{{9}}{{|c|}}{{Flatten ({size}, {components}, SFC$\\mathcal{{C}}$) to 1D $\\forall \\; \\mathcal{{C}} \\in {sfc_set}$}} \\\\\n')
            
        f.write('\\hline\n')  
        layer_count += 1
        f.write(F'{layer_count}-ExpandNN-SFC$\\mathcal{{C}}$ & ({size * components * self.encoder.self_concat}, 1, SFC$\\mathcal{{C}}$) & 3 Variable (3 $\\times$ {size * components * self.encoder.self_concat}) & 1 & 1 & 0 & 0 & ({size * components * self.encoder.self_concat}, 1, SFC$\\mathcal{{C}}$) & {activate}\\\\\n')
        f.write('\\hline\n') 
        f.write(F'\\multicolumn{{9}}{{|c|}}{{Reshape ({size * components * self.encoder.self_concat}, 1, SFC$\\mathcal{{C}}$) to form ({size}, {components * self.encoder.self_concat}, SFC$\\mathcal{{C}}$)}} \\\\\n')
        f.write('\\hline\n')
      else:
           if self.encoder.self_concat > 1:
              f.write(F'\\multicolumn{{9}}{{|c|}}{{Copying channels and concatenate at the 2nd dimension to form ({size}, {components * self.encoder.self_concat}, SFC$\\mathcal{{C}}$)}} \\\\\n')
              f.write('\\hline\n')
    
      for i in range(len(self.encoder.conv_size) - 1):
        conv_f = self.encoder.conv_size[i]
        conv_n = self.encoder.conv_size[i + 1]
        layer_count += 1
        f.write(F'{layer_count}-Conv1d-SFC$\\mathcal{{C}}$ & ({conv_f}, {self.encoder.channels[i]}, SFC$\\mathcal{{C}}$) & {self.encoder.kernel_size} & {self.encoder.channels[i + 1]} & {self.encoder.stride} & {self.encoder.padding} & 0 & ({conv_n}, {self.encoder.channels[i + 1]}, SFC$\\mathcal{{C}}$) & {activate}\\\\\n')
        f.write('\\hline\n')
    
      for i in range(len(self.encoder.size_fc) - 2):
        layer_count += 1
        fc_f = self.encoder.size_fc[i]
        fc_n = self.encoder.size_fc[i + 1]
        if i == 0:
            f.write(F'{layer_count}-FC & {fc_f} ($= {self.encoder.conv_size[-1]} \\times {self.encoder.channels[-1]} \\times {self.encoder.sfc_nums}$) & \\multicolumn{{5}}{{c|}}{{}} & {fc_n} & {activate}\\\\\n')
        else:
            f.write(F'{layer_count}-FC & {fc_f} & \multicolumn{{5}}{{c|}}{{}} & {fc_n} & {activate}\\\\\n')
        f.write('\\hline\n')
      
      # Whether variational decide the format in the middle
      if self.encoder.variational: 
         layer_count += 1
         f.write('\\multicolumn{9}{|c|}{\\textbf{Variational Reparametrization}}\\\\\n')
         f.write('\\hline\n') 
         f.write(F'{layer_count}-FC-$\\sigma$ & {self.encoder.size_fc[-2]} & \multicolumn{{5}}{{c|}}{{}} & {self.encoder.size_fc[-1]} & {activate}\\\\\n')
         f.write('\\hline\n') 
         f.write(F'{layer_count}-FC-$\\mu$ & {self.encoder.size_fc[-2]} & \multicolumn{{5}}{{c|}}{{}} & {self.encoder.size_fc[-1]} & {activate}\\\\\n')
         f.write('\\hline\n') 
         layer_count += 1
         f.write(F'{layer_count}-Sampling & {self.encoder.size_fc[-1]} & 3 Variable (3 $\\times${self.encoder.size_fc[-1]}) & 1 & 1 & 0 & 0 & {self.encoder.size_fc[-1]} & Identity\\\\\n')
         layer_count += 1
      else: 
         f.write(F'{layer_count}-FC & {self.encoder.size_fc[-2]} & \multicolumn{{5}}{{c|}}{{}} & {self.encoder.size_fc[-1]} & {activate}\\\\\n') 
         layer_count += 1    
      f.write('\\hline\n')
        
      f.write('\\multicolumn{9}{|c|}{\\textbf{Decoder}}\\\\\n')
      f.write('\\hline\n')  
    
      for i in range(1, len(self.encoder.size_fc)):
        layer_count += 1
        fc_f = self.encoder.size_fc[-i]
        fc_n = self.encoder.size_fc[-i - 1]
        f.write(F'{layer_count}-FC & {fc_f} & \multicolumn{{5}}{{c|}}{{}} & {fc_n} & {activate}\\\\\n')
        f.write('\\hline\n')

      f.write(F'\multicolumn{{9}}{{|c|}}{{Split the data into {self.encoder.sfc_nums} sequences as the input of layer {layer_count + 1}-TransConv1d-SFC$\\mathcal{{C}}$ $\\forall \\; \\mathcal{{C}} \\in {sfc_set}$, convert from {self.encoder.size_fc[0] // self.encoder.sfc_nums} to ({self.encoder.conv_size[-1]}, {self.encoder.num_final_channels})}} \\\\\n')
      f.write('\\hline\n')    
    
      for i in range(1, len(self.encoder.conv_size)):
        conv_f = self.encoder.conv_size[-i]
        conv_n = self.encoder.conv_size[-i - 1]
        layer_count += 1
        f.write(F'{layer_count}-TransConv1d-SFC$\\mathcal{{C}}$ & ({conv_f}, {self.encoder.channels[-i]}, SFC$\\mathcal{{C}}$) & {self.encoder.kernel_size} & {self.encoder.channels[-i]} & {self.encoder.stride} & {self.encoder.padding} & {self.encoder.output_paddings[i - 1]} & ({conv_n}, {self.encoder.channels[-i - 1]}, SFC$\\mathcal{{C}}$) & {activate}\\\\\n')
        f.write('\\hline\n')
    
      if self.encoder.NN:
        f.write(F'\multicolumn{{9}}{{|c|}}{{Apply inverse SFC orderings to ({size}, {components}, SFC$\\mathcal{{C}}$) $\\forall \\; \\mathcal{{C}} \\in {sfc_set}$,  flatten into ({size * components * self.encoder.self_concat}, 1, {type_m}$\mathcal{{C}}$)}} \\\\\n')
        f.write('\\hline\n')
        layer_count += 1
        f.write(F'{layer_count}-ExpandNN-{type_m}$\\mathcal{{C}}$ & ({size * components * self.encoder.self_concat}, 1, {type_m}$\\mathcal{{C}}$) & 3 Variable (3 $\\times$ {size * components * self.encoder.self_concat}) & 1 & 1 & 0 & 0 & ({size * components * self.encoder.self_concat}, 1, {type_m}$\\mathcal{{C}}$) & {activate}\\\\\n')
        f.write('\\hline\n')    
        f.write(F'\multicolumn{{9}}{{|c|}}{{Reshape ({size * components * self.encoder.self_concat}, 1, {type_m}$\\mathcal{{C}}$), and separate self-concat channels, to form ({size}, {components * self.encoder.self_concat}, {type_m}$\\mathcal{{C}}$) $\\times$ {self.encoder.self_concat}}} \\\\\n')
        f.write('\\hline\n')         
      else:
        f.write(F'\multicolumn{{9}}{{|c|}}{{Apply inverse SFC orderings to ({size}, {self.encoder.channels[0]}, SFC$\\mathcal{{C}}$) $\\forall \\; \\mathcal{{C}} \\in {sfc_set}$, and separate self-concat channels, to form ({size}, {components}, {type_m}$\\mathcal{{C}}$) $\\times$ {self.encoder.self_concat}}} \\\\\n')
        f.write('\\hline\n')
    
      layer_count += 1
      f.write(F'{layer_count}{cell_end} & ({size}, {components}, {type_m}$\\mathcal{{C}}$) $\\times$ {self.encoder.self_concat * self.encoder.sfc_nums} & $\\sum\\limits_{{i = 1}}^{{{self.encoder.self_concat * self.encoder.sfc_nums}}} \\; ${type_m}$i$ & {components} & 1 & 0 & 0 & ({size}, {components}, {type_m}$\\mathcal{{C}}$) & {activate} \\\\\n')
      f.write('\\hline\n')
    
      f.write('\\end{tabular}%\n')
      f.write('}\n')
      f.write('\\end{table}\n')
      f.close()
      print("The LaTeX script for the table structure of the SFC-CAE has been written to 'LatexTable.txt', please copy it to a LaTeX compiler environment, e.g. overleaf.")

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

