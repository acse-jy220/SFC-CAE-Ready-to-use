"""
This script contains the main functions for loading/training/saving the autoencoder.
Author: Jin Yu
Github handle: acse-jy220
"""

import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn
from torch.nn.parallel.data_parallel import DataParallel  # Function module
from torch.utils.data import DataLoader
from livelossplot import PlotLosses
import random 
import numpy as np
from sfc_cae.utils import *
# for other custom Pytorch Optimizers
from timm import optim as tioptim
# Distributed Data Parallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import distributed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True

    return True

def setup_DDP(rank, world_size):
    '''
    Setup Distributed Data Parallel.
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_DDP():
    '''
    Destroy Distributed Data Parallel.
    '''
    dist.destroy_process_group()

def relative_MSE(x, y, epsilon = 0):
    '''
    Compute the relative MSE
    x: [tensor] prediction
    y: [tensor] true_value
    '''

    assert x.shape == y.shape, 'the input tensors should have the same shape!'
    return ((x - y) ** 2).sum() / (y ** 2).sum()     

def save_model(model, optimizer, check_gap, n_epoches, save_path, dict_only = False):
    '''
    Save model and parameters of the optimizer as pth file, for continuous training.
    ---
    model: the neutral network
    optimizer: the current optimizer in training process
    n_epoches: the finishing epoches for this run
    save_path: the path for saving this module

    '''
    model_name = F"{save_path}.pth" 
    model_dictname = F"{save_path}_dict.pth"

    # save the model_dict (as well as the learning rate and epoches)
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'check_gap':check_gap,
            'epoch_start':n_epoches
            }, model_dictname)
    
    if not dict_only:
      # save the pure model (for direct evaluation)
      torch.save(model, model_name)
      print('model saved to', model_name)
    print('model_dict saved to', model_dictname)

def train(autoencoder, variational, optimizer, criterion, other_metric, dataloader, parallel_mode):
  '''
  This function is implemented for training the model.

  Input:
  ---
  autoencoder: [SFC_CAE object] the trained SFC_(V)CAE.
  variational: [bool] whether this is a variational autoencoder or not.
  optimizer: [torch.optim.Optimizer] Pytorch optimizer object for the model
  criterion: [torch.nn.MSELoss() or other] the obejctive function for training.
  other_metric: [reMSELoss() or other] other metric for evaluation
  dataloader: [torch.utils.data.DataLoader] the dataloader for evaluation.

  Output:
  ---
  train_loss / data_length: [torch.float with no grad] train loss for the batch
  train_loss_other/ data_length:  [torch.float with no grad] other metric train loss for the batch

  (variational Optional)
  whole_MSE/ data_length: [torch.float with no grad] MSE loss for the batch
  whole_KL/ data_length: [torch.float with no grad] KL divergence loss for the batch
  '''
  autoencoder.train()
  train_loss, train_loss_other, data_length = 0, 0, len(dataloader.dataset)
  if variational: 
     whole_KL = 0
     whole_MSE = 0
  count = 0
  for batch in dataloader:
      count += batch.size(0)
      if not isinstance(autoencoder, DDP): batch = batch.to(device)  # Send batch of images to the GPU
      optimizer.zero_grad()  # Set optimiser grad to 0
      if variational:
        x_hat, KL = autoencoder(batch)
        MSE = criterion(batch, x_hat)
        if torch.cuda.device_count() > 1: KL = KL.sum()
        whole_KL += KL.detach().cpu().numpy() * batch.size(0)
        whole_MSE += MSE.item() * batch.size(0)
        Loss = MSE.add_(KL) # MSE loss plus KL divergence
      else:
        x_hat = autoencoder(batch)
        Loss = criterion(batch, x_hat)  # Calculate MSE loss
      with torch.no_grad(): other_MSE = other_metric(batch, x_hat)  # Calculate (may be) relative loss
      Loss.backward()  # Back-propagate
      optimizer.step()
      train_loss += Loss * batch.size(0)
      train_loss_other += other_MSE * batch.size(0)
      del x_hat
      del batch
      del Loss
      del other_MSE
  if variational: return train_loss / data_length, train_loss_other/ data_length, whole_MSE/ data_length, whole_KL/ data_length  # Return Loss, MSE, KL separately.
  else: return train_loss / data_length, train_loss_other/ data_length  # Return MSE

def validate(autoencoder, variational, optimizer, criterion, other_metric, dataloader, parallel_mode):
  '''
  This function is implemented for validating the model.

  Input:
  ---
  autoencoder: [SFC_CAE object] the trained SFC_(V)CAE.
  variational: [bool] whether this is a variational autoencoder or not.
  optimizer: [torch.optim.Optimizer] Pytorch optimizer object for the model
  criterion: [torch.nn.MSELoss() or other] the obejctive function for training.
  other_metric: [reMSELoss() or other] other metric for evaluation
  dataloader: [torch.utils.data.DataLoader] the dataloader for evaluation.

  Output:
  ---
  validation_loss / data_length: [torch.float with no grad] validation loss for the batch
  valid_loss_other/ data_length:  [torch.float with no grad] other metric validation loss for the batch

  (variational Optional)
  whole_MSE/ data_length: [torch.float with no grad] MSE loss for the batch
  whole_KL/ data_length: [torch.float with no grad] KL divergence loss for the batch
  '''
  autoencoder.eval()
  validation_loss, valid_loss_other, data_length = 0, 0, len(dataloader.dataset)
  if variational: 
    whole_KL = 0
    whole_MSE = 0
  count = 0
  for batch in dataloader:
    with torch.no_grad():
      count += batch.size(0)
      if not isinstance(autoencoder, DDP): batch = batch.to(device)  # Send batch of images to the GPU
      if variational:
          x_hat, KL = autoencoder(batch)
          MSE = criterion(batch, x_hat)
          if torch.cuda.device_count() > 1: KL = KL.sum()
          whole_KL += KL.detach().cpu().numpy() * batch.size(0)
          whole_MSE += MSE.item() * batch.size(0)
          Loss = MSE.add_(KL) # MSE loss plus KL divergence
      else:
          x_hat = autoencoder(batch)
          Loss = criterion(batch, x_hat)  # Calculate MSE loss
      other_MSE = other_metric(batch, x_hat)
      validation_loss += Loss * batch.size(0)
      valid_loss_other += other_MSE * batch.size(0)
      del batch
      del x_hat
      del Loss
      del other_MSE

  if variational: return validation_loss / data_length, valid_loss_other/ data_length, whole_MSE/ data_length, whole_KL/ data_length  # Return Loss, MSE, KL separately.
  else: return validation_loss / data_length, valid_loss_other/ data_length  # Return MSE

# main function for training, returns a trained model as well as the final loss function value and accuracy for the train, valid, test sets.
def train_model(autoencoder,
                train_loader, 
                valid_loader,
                test_loader,
                lr = 1e-4,
                n_epochs = 100,
                seed = 41,
                save_path = None,
                dict_only = False,
                visualize = True,
                parallel_mode = 'DP',
                optimizer_type = 'Adam',
                state_load = None,
                varying_lr = False, 
                check_gap = 3,
                weight_decay = 0, 
                criterion_type = 'MSE',
                rank = None):
  '''
  This function is main function for loading, training, and saving the model.

  Input:
  ---
  autoencoder: [SFC_CAE object] the untrained SFC_(V)CAE.
  train_loader: [torch.utils.data.DataLoader] the DataLoader for train set.
  valid_loader: [torch.utils.data.DataLoader] the DataLoader for valid set.
  test_loader: [torch.utils.data.DataLoader] the DataLoader for test set.
  optimizer: [string] 'Adam' or 'Adamax', the optimizer for the model
  state_load: [string] the state_dict for the SFC_(V)CAE object.
  n_epochs: [int] total number of epoches to train.
  varying_lr: [bool] if turned on, the learning rate will automatically decrease if
              a detection of 'stucking' happenes. 
  check_gap: [int] the number of initial check gap of 'varying_lr'.
  lr: [float] the initial learning rate 
  weight_decay: [float] the coefficient for L2 regularization.
  criterion_type: [string] 'MSE' or 'relative_MSE'
  visualize: [bool] whether do a liveloss plot.
  seed: [int] the random seed from cuda kernels
  save_path: [string] the path to save the training txt files and model/model_dict.
  dict_only: [bool] only save the model_dict to save memory of the disk for large models.
  parallel_mode: [string] 'DP' or 'DDP', 'DP' represents Data Parallel, 'DDP' represents Distributed Data Parallel.

  Output:
  ---
  autoencoder: [SFC_CAE object] the trained SFC_(V)CAE.  
  '''
  set_seed(seed)
  if isinstance(autoencoder, DDP): 
     variational = autoencoder.module.encoder.variational
     # device = rank
  else: variational = autoencoder.encoder.variational
  
  print('torch device num:', torch.cuda.device_count(),'\n')
  if not isinstance(autoencoder, DDP): autoencoder.to(device)
  if torch.cuda.device_count() > 1 and parallel_mode == 'DP':
     print("Let's use", torch.cuda.device_count(), "GPUs!")
     autoencoder = torch.nn.DataParallel(autoencoder)

  # see if continue training happens
  if state_load is not None:
     if torch.cuda.device_count() > 1 and parallel_mode == 'DP': state_load = torch.load(state_load, map_location='cuda:0')
     else: state_load = torch.load(state_load)
     check_gap = state_load['check_gap']
     epoch_start = state_load['epoch_start']
     if torch.cuda.device_count() > 1 and parallel_mode == 'DP': autoencoder.module.load_state_dict(state_load['model_state_dict'])
     else: autoencoder.load_state_dict(state_load['model_state_dict'])
     optimizer_state_dict = state_load['optimizer_state_dict']
  else: epoch_start = 0
  
  if optimizer_type == 'Adam': optimizer = torch.optim.Adam(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)
  elif optimizer_type == 'Adamax': optimizer = torch.optim.Adamax(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)
  elif optimizer_type == 'SGD': optimizer = torch.optim.SGD(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)
  elif optimizer_type == 'Adagrad': optimizer = torch.optim.Adagrad(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)
  elif optimizer_type == 'AdamW': optimizer = torch.optim.AdamW(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)
  # other custom Pytorch optimizers, from https://github.com/rwightman/pytorch-image-models/tree/master/timm/optim
  elif optimizer_type == 'Nadam': optimizer = tioptim.Nadam(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)
  elif optimizer_type == 'AdamP': optimizer = tioptim.AdamP(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)
  elif optimizer_type == 'RAdam': optimizer = tioptim.RAdam(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)
  elif optimizer_type == 'madgrad': optimizer = tioptim.MADGRAD(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)
  elif optimizer_type == 'RMSprop': optimizer = tioptim.RMSpropTF(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)

  if state_load is not None: optimizer.load_state_dict(optimizer_state_dict)

  if criterion_type == 'MSE': 
      criterion = nn.MSELoss()
      other_metric = relative_MSE
  elif criterion_type == 'relative_MSE': 
      other_metric = nn.MSELoss()
      criterion = relative_MSE

  train_MSEs = []
  valid_MSEs = []
  re_train_MSEs = []
  re_valid_MSEs = []
  
  total_time_start = time.time()

  # do livelossplot if visualize turned-on
  if visualize:
      liveloss = PlotLosses()
  
  # initialize some parameters before training
  old_loss = 1
  decrease_rate = 0
  lr_list = [lr]
  lr_change_epoches = [int(epoch_start)]
  n_epochs += epoch_start

  for epoch in range(epoch_start, n_epochs):
    print("epoch %d starting......"%(epoch))
    time_start = time.time()
    if variational:
      train_loss, train_loss_other, real_train_MSE, train_KL = train(autoencoder, variational, optimizer, criterion, other_metric, train_loader, parallel_mode) 
      valid_loss, valid_loss_other, real_valid_MSE, valid_KL = validate(autoencoder, variational, optimizer, criterion, other_metric, valid_loader, parallel_mode)
    else:
      train_loss, train_loss_other = train(autoencoder, variational, optimizer, criterion, other_metric, train_loader, parallel_mode)
      valid_loss, valid_loss_other = validate(autoencoder, variational, optimizer, criterion, other_metric, valid_loader, parallel_mode)

    if criterion_type == 'MSE':
        train_MSE_re = train_loss_other.cpu().numpy()
        valid_MSE_re = valid_loss_other.cpu().numpy()
        train_MSE = train_loss.cpu().detach().numpy()
        valid_MSE = valid_loss.cpu().numpy()
    elif criterion_type == 'relative_MSE':
        train_MSE = train_loss_other.cpu().numpy()
        valid_MSE = valid_loss_other.cpu().numpy()
        train_MSE_re = train_loss.cpu().detach().numpy()
        valid_MSE_re = valid_loss.cpu().numpy()
    
    # do livelossplot if visualize turned-on 
    if visualize: 
      logs = {}
                 
      logs['' + 'log loss'] = train_MSE
      logs['val_' + 'log loss'] = valid_MSE

      logs['' + 'log loss (relative)'] = train_MSE_re
      logs['val_' + 'log loss (relative)'] = valid_MSE_re          
      
      liveloss.update(logs)
      liveloss.draw()

    time_end = time.time()
    train_MSEs.append(train_MSE)
    valid_MSEs.append(valid_MSE)
    re_train_MSEs.append(train_MSE_re)
    re_valid_MSEs.append(valid_MSE_re)

    if variational:
        print('Epoch: ', epoch, '| train loss: %e' % train_MSE, '| train MSE: %e' % real_train_MSE, '| train KL: %e' % train_KL, '\n       \t'  
        '| valid loss: %e' % valid_MSE, '| valid MSE: %e' % real_valid_MSE, '| valid KL: %e' % valid_KL, 
        '\n      \t| train loss (relative): %e' % train_MSE_re, '| valid loss (relative): %e' % valid_MSE_re,
          '\nEpoch %d use: %.2f second.\n' % (epoch, time_end - time_start))
    else:
        print('Epoch: ', epoch, '| train loss: %e' % train_MSE, '| valid loss: %e' % valid_MSE,
          '\n      \t| train loss (relative): %e' % train_MSE_re, '| valid loss (relative): %e' % valid_MSE_re,
          '\nEpoch %d use: %.2f second.\n' % (epoch, time_end - time_start))
    
    if varying_lr:
      print("Current learning rate: %.0e"% optimizer.param_groups[0]['lr'])
      this_loss = train_MSE
      decrease_rate += old_loss - this_loss
      if epoch % check_gap == 0: 
        # print(F'check at epoch {epoch}')
        digits = -np.floor(np.log10(train_MSE))
        decrease_rate *= 10 ** digits
        print(F'Accumulated loss bewteen two consecutive {check_gap} epoches :%.2e' % (decrease_rate))
        if decrease_rate < 1e-2:    
         optimizer.param_groups[0]['lr'] /= 2
         check_gap *= 2
         lr_list.append(optimizer.param_groups[0]['lr'])
         lr_change_epoches.append(int(epoch))
      decrease_rate = 0
      old_loss = this_loss
  
  if variational:
    test_loss, test_loss_other, real_test_MSE, test_KL = validate(autoencoder, variational, optimizer, criterion, other_metric, test_loader, parallel_mode)
  else:
    test_loss, test_loss_other = validate(autoencoder, variational, optimizer, criterion, other_metric, test_loader, parallel_mode)

  if criterion_type == 'MSE':
    test_MSE_re = test_loss_other.cpu().numpy()
    test_MSE = test_loss.cpu().detach().numpy()
  elif criterion_type == 'relative_MSE':
      test_MSE = test_loss_other.cpu().numpy()
      test_MSE_re = test_loss.cpu().detach().numpy()

  total_time_end = time.time()
  
  if variational:
    print('test MSE Error: %e' % test_MSE, '| test MSE: %e' % real_test_MSE, '| test KL: %e' % test_KL, '| relative MSE Error: %e' % test_MSE_re, '\n Total time used for training: %.2f hour.' % ((total_time_end - total_time_start)/3600)) 
  else:
    print('test MSE Error: %e' % test_MSE, '| relative MSE Error: %e' % test_MSE_re, '\n Total time used for training: %.2f hour.' % ((total_time_end - total_time_start)/3600)) 

  MSELoss = np.vstack((np.array(train_MSEs), np.array(valid_MSEs))).T
  reMSELoss = np.vstack((np.array(re_train_MSEs), np.array(re_valid_MSEs))).T

  if isinstance(autoencoder, DataParallel) or isinstance(autoencoder, DDP):
     NN = autoencoder.module.encoder.NN
     sfc_nums = autoencoder.module.encoder.sfc_nums
     latent = autoencoder.module.encoder.dims_latent
     variational = autoencoder.module.encoder.variational
     activate = autoencoder.module.activate
     output_linear = autoencoder.module.decoder.output_linear
  else:
     NN = autoencoder.encoder.NN
     sfc_nums = autoencoder.encoder.sfc_nums
     latent = autoencoder.encoder.dims_latent
     variational = autoencoder.encoder.variational
     activate = autoencoder.activate
     output_linear = autoencoder.decoder.output_linear
  if save_path is not None:

    if varying_lr:
      lr_epoch_lists = np.vstack((np.array(lr_change_epoches), np.array(lr_list))).T
      np.savetxt(save_path +'lr_changes_at_epoch.txt', lr_epoch_lists)
    
    filename = save_path + F'{parallel_mode}_Optimizer_{optimizer_type}_Activation_{activate}_OutputLinear_{output_linear}_Variational_{variational}_Changelr_{varying_lr}_MSELoss_Latent_{latent}_nearest_neighbouring_{NN}_SFC_nums_{sfc_nums}_startlr_{lr}_n_epoches_{n_epochs}.txt'
    refilename = save_path + F'{parallel_mode}_Optimizer_{optimizer_type}_Activation_{activate}_OutputLinear_{output_linear}_Variational_{variational}_Changelr_{varying_lr}_reMSELoss_Latent_{latent}_nearest_neighbouring_{NN}_SFC_nums_{sfc_nums}_startlr_{lr}_n_epoches_{n_epochs}.txt'

    np.savetxt(filename, MSELoss)
    np.savetxt(refilename, reMSELoss)

    print('MESLoss saved to ', filename)
    print('relative MSELoss saved to ', refilename)

    save_path = save_path + F'{parallel_mode}_Optimizer_{optimizer_type}_Activation_{activate}_OutputLinear_{output_linear}_Variational_{variational}_Changelr_{varying_lr}_Latent_{latent}_Nearest_neighbouring_{NN}_SFC_nums_{sfc_nums}_startlr_{lr}_n_epoches_{n_epochs}'
  
    if isinstance(autoencoder, DataParallel) or isinstance(autoencoder, DDP):
      save_model(autoencoder.module, optimizer, check_gap, n_epochs, save_path, dict_only)
    else:
      save_model(autoencoder, optimizer, check_gap, n_epochs, save_path, dict_only)

  return autoencoder

def get_dataloader(rank, train_set, valid_set, test_set, batch_size, world_size = torch.cuda.device_count()):
    train_sampler = distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, sampler=test_sampler)
    return train_loader, valid_loader, test_loader

def train_model_DDP(rank, 
                    autoencoder,
                    train_set, 
                    valid_set,
                    test_set,
                    batch_size,
                    optimizer_type = 'Adam',
                    state_load = None,
                    n_epochs = 100,
                    varying_lr = False, 
                    lr = 1e-4,
                    visualize = True, 
                    seed = 41,
                    save_path = None,
                    dict_only = False):

    print(f"Running DDP on rank {rank}.")
    setup_DDP(rank, torch.cuda.device_count())

    # create model and move it to GPU with id rank
    autoencoder = autoencoder.to(rank)
    autoencoder = DDP(autoencoder, device_ids=[rank])

    train_loader, valid_loader, test_loader = get_dataloader(rank, train_set, valid_set, test_set, batch_size)

    print('pass here now.')

    train_model(autoencoder,
                train_loader, 
                valid_loader,
                test_loader,
                lr = lr,
                n_epochs = n_epochs,
                seed = seed,
                save_path = save_path,
                dict_only = dict_only,
                visualize = visualize,
                parallel_mode = 'DDP',
                optimizer_type = optimizer_type,
                rank = rank)

    cleanup_DDP()

    return autoencoder


  