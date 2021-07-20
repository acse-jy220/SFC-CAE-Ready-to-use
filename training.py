import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torch.utils.data import DataLoader
from livelossplot import PlotLosses
import random 
import numpy as np
from util import *
import util

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

device = 'cuda'  # Set out device to GPU

def relative_MSE(x, y, epsilon = 0):
    '''
    Compute relative MSE
    x: [tensor] prediction
    y: [tensor] true_value
    '''

    assert x.shape == y.shape, 'the input tensors should have the same shape!'
    return ((x - y) ** 2).sum() / (y ** 2).sum()     

def train(autoencoder, optimizer, criterion, other_metric, dataloader):
  autoencoder.train()
  train_loss, train_loss_other, data_length = 0, 0, len(dataloader.dataset)
  count = 0
  for batch in dataloader:
      count += batch.size(0)
      batch = batch.to(device)  # Send batch of images to the GPU
      optimizer.zero_grad()  # Set optimiser grad to 0
      if torch.cuda.device_count() > 1:
         x_hat = autoencoder.module(batch)
      else: x_hat = autoencoder(batch)  # Generate predicted images (x_hat) by running batch of images through autoencoder
      MSE = criterion(batch, x_hat)  # Calculate MSE loss
      with torch.no_grad(): other_MSE = other_metric(batch, x_hat).item() # Calculate (may be) relative loss
      MSE.backward()  # Back-propagate
      if torch.cuda.device_count() > 1: optimizer.module.step()
      else: optimizer.step()  # Step the optimiser
      train_loss += MSE * batch.size(0)
      train_loss_other += other_MSE * batch.size(0)
      del x_hat
      del batch
      del MSE
      del other_MSE
    #   print(count)

  return train_loss / data_length, train_loss_other/ data_length  # Return MSE

def validate(autoencoder, optimizer, criterion, other_metric, dataloader):
    autoencoder.eval()
    validation_loss, valid_loss_other, data_length = 0, 0, len(dataloader.dataset)
    count = 0
    for batch in dataloader:
        with torch.no_grad():
            count += batch.size(0)
            batch = batch.to(device)  # Send batch of images to the GPU
            if torch.cuda.device_count() > 1:
               x_hat = autoencoder.module(batch)
            else: x_hat = autoencoder(batch)  # Generate predicted images (x_hat) by running batch of images through autoencoder
            MSE = criterion(batch, x_hat)  # Calculate MSE loss
            other_MSE = other_metric(batch, x_hat)
            validation_loss += MSE * batch.size(0)
            valid_loss_other += other_MSE * batch.size(0)
            del batch
            del x_hat
            del MSE
            del other_MSE
            # print('valid ', count)

    return validation_loss / data_length, valid_loss_other / data_length   # Return MSE  

# main function for training, returns a trained model as well as the final loss function value and accuracy for the validation set.
def train_model(autoencoder, 
                train_loader, 
                valid_loader, 
                n_epochs = 100, 
                lr = 1e-4, 
                weight_decay = 0, 
                criterion_type = 'MSE', 
                visualize=True, 
                seed = 41,
                save_path = ''):
  set_seed(seed)
  if torch.cuda.device_count() > 1:
     autoencoder = torch.nn.DataParallel(autoencoder)
  autoencoder = autoencoder.to(device)
  optimizer = torch.optim.Adam(autoencoder.parameters(), lr = lr, weight_decay = weight_decay)
  if torch.cuda.device_count() > 1:
     optimizer = torch.nn.DataParallel(optimizer)

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
  
  # do livelossplot if visualize turned-on
  if visualize:
      liveloss = PlotLosses()

  for epoch in range(n_epochs):
    print("epoch %d starting......"%(epoch))
    time_start = time.time()
    train_loss, train_loss_other = train(autoencoder, optimizer, criterion, other_metric, train_loader)
    valid_loss, valid_loss_other = validate(autoencoder, optimizer, criterion, other_metric, valid_loader)

    if criterion_type == 'MSE':
        train_MSE_re = train_loss_other
        valid_MSE_re = valid_loss_other
        train_MSE = train_loss.item()
        valid_MSE = valid_loss.item()
    elif criterion_type == 'relative_MSE':
        train_MSE = train_loss_other
        valid_MSE = valid_loss_other
        train_MSE_re = train_loss.item()
        valid_MSE_re = valid_loss.item()
    
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
    print(type(train_MSE))
    train_MSEs.append(train_MSE)
    valid_MSEs.append(valid_MSE)
    re_train_MSEs.append(train_MSE_re)
    re_valid_MSEs.append(valid_MSE_re)

    print('Epoch: ', epoch, '| train loss: %e' % train_MSE, '| valid loss: %e' % valid_MSE,
          '\n      \t| train loss (relative): %e' % train_MSE_re, '| valid loss (relative): %e' % valid_MSE_re,
          '\nEpoch %d use: %.2f second.' % (epoch, time_end - time_start))

  MSELoss = np.vstack((np.array(train_MSEs), np.array(valid_MSEs))).T
  reMSELoss = np.vstack((np.array(re_train_MSEs), np.array(re_valid_MSEs))).T
  
  filename = save_path + F'MSELoss_nearest_neighbouring_{autoencoder.encoder.NN}_SFC_nums_{autoencoder.encoder.sfc_nums}_lr_{lr}_n_epoches_{n_epochs}.txt'
  refilename = save_path + F'reMSELoss_nearest_neighbouring_{autoencoder.encoder.NN}_SFC_nums_{autoencoder.encoder.sfc_nums}_lr_{lr}_n_epoches_{n_epochs}.txt'

  np.savetxt(filename, MSELoss)
  np.savetxt(refilename, reMSELoss)

  print('MESLoss saved to ', filename)
  print('relative MSELoss saved to ', refilename)

  modelname = save_path + F'Slugflow_nearest_neighbouring_{autoencoder.encoder.NN}_SFC_nums_{autoencoder.encoder.sfc_nums}_lr_{lr}_n_epoches_{n_epochs}.pth'
  modeldictname = save_path + F'Slugflow_nearest_neighbouring_{autoencoder.encoder.NN}_SFC_nums_{autoencoder.encoder.sfc_nums}_lr_{lr}_n_epoches_{n_epochs}_dict.pth'
  
  torch.save(autoencoder, modelname)
  torch.save(autoencoder.state_dict(), modeldictname)
  print('model saved to', modelname)

  return autoencoder
  