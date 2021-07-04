from simple_hilbert import *
from advection_block_analytical import *
import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torchvision import transforms  # Transforms from torchvision
from util import *

class SFC_CAE_Encoder(nn.Module): 
  def __init__(self, 
               input_size,
               dimension,
               components,
               structured,
               self_concat,
               nearest_neighbouring, 
               dims_latent, 
               space_filling_orderings):
    '''
    Class contains the Encoder (snapshot -> latent).
    '''

    super(SFC_CAE_Encoder, self).__init__()
    self.NN = nearest_neighbouring
    self.dims_latent = dims_latent
    self.dimension = dimension
    self.input_size = input_size
    self.components = components
    self.self_concat = self_concat
    self.input_channel = components * self_concat
    self.orderings = []
    self.sfc_plus = []
    self.sfc_minus = []
    self.sfc_nums = len(space_filling_orderings)

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
    elif dimension == 3:
        self.kernel_size = 176
        self.stride = 8
    self.padding = self.kernel_size//2

    self.structured = structured
    if self.structured: 
        self.increase_multi = 2
        self.activate = nn.ReLU()
    else: 
        self.increase_multi = 4
        self.activate = nn.Tanh()

    self.num_final_channels = 16  # default

    self.size_conv, self.size_fc, self.channels, self.inv_conv_start, self.output_paddings \
    = find_size_conv_layers_and_fc_layers(self.input_size, self.stride, self.dims_latent, self.sfc_nums, self.input_channel, self.increase_multi,  self.num_final_channels)
    
    # set up convolutional layers, fully-connected layers and sparse layers
    self.fcs = []
    self.convs = []
    if self.NN: self.sps = []
    for i in range(self.sfc_nums):
       self.convs.append([])
       for j in range(self.size_conv):
           self.convs[i].append(nn.Conv1d(self.channels[j], self.channels[j+1], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
           self.register_parameter(name='conv%d%d_weights'%(i + 1, j + 1), param=self.convs[i][j].weight)
           self.register_parameter(name='conv%d%d_bias'%(i + 1, j + 1), param=self.convs[i][j].bias)
       if self.NN:
          self.sps.append(NearestNeighbouring(size = self.input_size * self.input_channel, initial_weight= (1/3), num_neigh = 3))
          self.register_parameter(name='sp%d_weights'%(i + 1), param=self.sps[i].weights)
          self.register_parameter(name='sp%d_bias'%(i + 1), param=self.sps[i].bias)   
    for i in range(len(self.size_fc) - 1):
       self.fcs.append(nn.Linear(self.size_fc[i], self.size_fc[i+1]))
       self.register_parameter(name='fc%d_weights'%(i + 1), param=self.fcs[i].weight)
       self.register_parameter(name='fc%d_bias'%(i + 1), param=self.fcs[i].bias)

    self.xs = torch.zeros(self.sfc_nums, self.num_final_channels * self.inv_conv_start)

  def get_concat_list(self, x, num_sfc):
    return torch.cat((ordering_tensor(x, self.sfc_minus[num_sfc]).unsqueeze(-1), 
                      ordering_tensor(x, self.orderings[num_sfc]).unsqueeze(-1),
                      ordering_tensor(x, self.sfc_plus[num_sfc]).unsqueeze(-1)), -1)


  def forward(self, x):  # Custom pytorch modules should follow this structure 
    '''
    x: [float] the fluid data snapshot, could have multiple components, but 
    the last dimension should always represent the component index.
    '''
    xs = []
    if self.components > 1: 
        x = x.permute(0, -1, -2)
        x = x.reshape(-1, x.shape[-2] * x.shape[-1])
    if self.self_concat > 1: x = torch.cat([x] * self.self_concat, -1)

    for i in range(self.sfc_nums):
        if self.NN:
           tt_list = self.get_concat_list(x, i)
           a = self.activate(self.sps[i](tt_list))
           del tt_list
        else:
           a = ordering_tensor(x, self.orderings[i])
        if self.input_channel > 1: a = a.view(-1, self.input_channel, self.input_size)
        else: a = a.unsqueeze(1)
        for j in range(self.size_conv):
            # print(a.shape)
            a = self.activate(self.convs[i][j](a))
        xs.append(a.view(-1, a.size(1)*a.size(2)))
        # print(xs[i].shape)
    del x
    if self.sfc_nums > 1: x = torch.cat(xs, -1)
    else: x = xs[0]
    for i in range(self.sfc_nums): del xs[0] # clear memory 
    # fully connect layers
    for i in range(len(self.fcs)): x = self.activate(self.fcs[i](x))
    return x

class SFC_CAE_Decoder(nn.Module): 
  def __init__(self, encoder, inv_space_filling_orderings):
    '''
    Class contains the Decoder (snapshot -> latent).
    '''

    super(SFC_CAE_Decoder, self).__init__()
    self.NN = encoder.NN
    self.activate = encoder.activate
    self.dims_latent = encoder.dims_latent
    self.dimension = encoder.dimension
    self.input_size = encoder.input_size
    self.components = encoder.components
    self.self_concat = encoder.self_concat
    self.num_final_channels = encoder.num_final_channels
    self.size_conv = encoder.size_conv
    self.inv_conv_start = encoder.inv_conv_start
    self.input_channel = self.components * self.self_concat
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
       

    if self.dimension == 2: 
        self.kernel_size = 32
        self.stride = 4
    elif self.dimension == 3:
        self.kernel_size = 176
        self.stride = 8
    self.padding = self.kernel_size//2
    if encoder.structured: self.increase_multi = 2
    else: self.increase_multi = 4
    
    self.fcs = []
    # set up fully-connected layers
    for k in range(1, len(encoder.size_fc)):
       self.fcs.append(nn.Linear(encoder.size_fc[-k], encoder.size_fc[-k-1]))
       self.register_parameter(name='fc%d_weights'%(k), param=self.fcs[k - 1].weight)
       self.register_parameter(name='fc%d_bias'%(k), param=self.fcs[k - 1].bias)


    # set up convolutional layers, fully-connected layers and sparse layers
    self.convTrans = []
    self.sps = []
    for i in range(self.sfc_nums):
       self.convTrans.append([])
       for j in range(1, encoder.size_conv + 1):
           self.convTrans[i].append(nn.ConvTranspose1d(encoder.channels[-j], encoder.channels[-j-1], kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size//2, output_padding = encoder.output_paddings[j - 1]))
           self.register_parameter(name='convTrans%d%d_weights'%(i + 1, j), param=self.convTrans[i][j - 1].weight)
           self.register_parameter(name='convTrans%d%d_bias'%(i + 1, j), param=self.convTrans[i][j - 1].bias)
       if self.NN:
          self.sps.append(NearestNeighbouring(size = self.input_size * self.components, initial_weight= (1/3) / self.self_concat, num_neigh = 3 * self.self_concat))
          self.register_parameter(name='sp%d_weights'%(i + 1), param=self.sps[i].weights)
          self.register_parameter(name='sp%d_bias'%(i + 1), param=self.sps[i].bias)   
       else:
          if self.self_concat > 1:
             self.sps.append(NearestNeighbouring(size = self.input_size * self.components, initial_weight= 1 / self.self_concat, num_neigh = self.self_concat))
             self.register_parameter(name='sp%d_weights'%(i + 1), param=self.sps[i].weights)
             self.register_parameter(name='sp%d_bias'%(i + 1), param=self.sps[i].bias)            

    self.split = encoder.size_fc[0] // self.sfc_nums

    # final sparse layer combining SFC outputs
    self.final_sp = NearestNeighbouring(size = self.input_size * self.components, initial_weight= 1 / self.sfc_nums, num_neigh = self.sfc_nums)
    self.register_parameter(name='final_sp_weights', param=self.final_sp.weights)
    self.register_parameter(name='final_sp_bias', param=self.final_sp.bias)   

  def get_concat_list(self, x, num_sfc):
    if self.self_concat > 1:
       return torch.cat((ordering_tensor(x, self.sfc_minus[num_sfc]).view(-1, self.self_concat, self.input_size * self.components).permute(0, -1, -2), 
                           ordering_tensor(x, self.orderings[num_sfc]).view(-1, self.self_concat, self.input_size * self.components).permute(0, -1, -2),
                           ordering_tensor(x, self.sfc_plus[num_sfc]).view(-1, self.self_concat, self.input_size * self.components).permute(0, -1, -2)), -1)
    else:
         return torch.cat((ordering_tensor(x, self.sfc_minus[num_sfc]).unsqueeze(-1), 
                           ordering_tensor(x, self.orderings[num_sfc]).unsqueeze(-1),
                           ordering_tensor(x, self.sfc_plus[num_sfc]).unsqueeze(-1)), -1)        


  def forward(self, x):  # Custom pytorch modules should follow this structure 
    '''
    z: [float] the fluid data snapshot, could have multiple components, but 
    the last dimension should always represent the component index.
    '''
    for i in range(len(self.fcs)):
        x = self.activate(self.fcs[i](x))
    
    x = x.view(-1, self.split, self.sfc_nums)
    # print(x.shape)
    zs = []
    for i in range(self.sfc_nums):
        b = x[..., i].view(-1, self.num_final_channels, self.inv_conv_start)
        for j in range(self.size_conv):
            b = self.activate(self.convTrans[i][j](b))
            # print(b.shape)
        b = b.view(-1, self.input_size * self.input_channel)
        # print(b.shape)
        if self.NN:
           b = self.get_concat_list(b, i)
        #    print(tt_list.shape)
           b = self.activate(self.sps[i](b))
        else:
           if self.self_concat > 1:
              b = ordering_tensor(b, self.orderings[i]).view(-1, self.self_concat, self.components * self.input_size).permute(0, -1, -2)
            #   print(tt_list.shape)
              b = self.activate(self.sps[i](b))
           else:
              b = ordering_tensor(b, self.orderings[i])
        zs.append(b.unsqueeze(-1))
    del x
    if self.sfc_nums > 1: 
        z = torch.cat(zs, -1)
        # print(z.shape)
        z = self.activate(self.final_sp(z))
    else: z = zs[0].squeeze(-1)
    for i in range(self.sfc_nums): del zs[0]
    if self.components > 1: return z.view(-1, self.components, self.input_size).permute(0, -1, -2)
    else: return z


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
               invert_space_filling_orderings):
    '''
    Class combines the Encoder and the Decoder with an Autoencoder latent space.

    dims_latent: [int] the dimension of (number of nodes in) the mean-field gaussian latent variable
    '''

    super(SFC_CAE, self).__init__()
    self.encoder = SFC_CAE_Encoder(size,
                          dimension,
                          components,
                          structured,
                          self_concat,
                          nearest_neighbouring, 
                          dims_latent, 
                          space_filling_orderings)
    self.decoder = SFC_CAE_Decoder(self.encoder, invert_space_filling_orderings)


  def forward(self, x):
    '''
    x - [float] A batch of images from the data-loader
    '''

    z = self.encoder(x)
    return self.decoder(z)  # Return the output of the decoder (1-D, the predicted image)
