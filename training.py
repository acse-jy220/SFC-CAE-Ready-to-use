import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torchvision import transforms  # Transforms from torchvision
from torchvision import datasets  # Datasets from torchvision
from torchvision import transforms  # Transforms from torchvision
from torch.utils.data import DataLoader
from livelossplot import PlotLosses

def relative_MSE(x, y, epsilon = 0):
    '''
    Compute relative MSE
    x: [tensor] prediction
    y: [tensor] true_value
    '''

    assert x.shape == y.shape, 'the input tensors should have the same shape!'
    return nn.MSELoss()(x, y) / (y ** 2).sum()     

def train(autoencoder, optimizer, criterion, dataloader):
  autoencoder.train()
  train_loss, data_length = 0, len(dataloader.dataset)
  for batch in dataloader:
      batch = batch.to(device)  # Send batch of images to the GPU
      optimizer.zero_grad()  # Set optimiser grad to 0
      x_hat = autoencoder(batch)  # Generate predicted images (x_hat) by running batch of images through autoencoder
      MSE = criterion(batch, x_hat)  # Calculate MSE loss
      MSE.backward()  # Back-propagate
      optimizer.step()  # Step the optimiser
      train_loss += MSE * batch.size(0)

  return train_loss / data_length  # Return MSE

def validate(autoencoder, optimizer, criterion, dataloader):
    autoencoder.eval()
    validation_loss, data_length = 0, len(dataloader.dataset)
    for batch in dataloader:
        with torch.no_grad():
            batch = batch.to(device)  # Send batch of images to the GPU
            x_hat = autoencoder(batch)  # Generate predicted images (x_hat) by running batch of images through autoencoder
            MSE = criterion(batch, x_hat)  # Calculate MSE loss
            validation_loss += MSE * batch.size(0)

    return validation_loss / data_length   # Return MSE  

# main function for training, returns a trained model as well as the final loss function value and accuracy for the validation set.
def train_model(autoencoder, batch_size=64, n_epochs = 10, lr = 5e-5, weight_decay = 0, criterion = nn.MSELoss(), visualize=True):
  set_seed(seed)
  autoencoder = autoencoder.to(device)
  optimizer = torch.optim.Adam(autoencoder.parameters(), lr = lr)

  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
  valid_loader = DataLoader(valid_set, batch_size=valid_set.shape[0], shuffle=True, num_workers=0)
  
  # do livelossplot if visualize turned-on
  if visualize:
      liveloss = PlotLosses()

  for epoch in range(n_epochs):
    train_MSE = train(autoencoder, optimizer, criterion, train_loader)
    validation_MSE = validate(autoencoder, optimizer, criterion, valid_loader)
    print("eppoch %d starting......"%(epoch))
    
    # do livelossplot if visualize turned-on 
    if visualize: 
      logs = {}

      logs['' + 'log loss'] = train_MSE.cpu().data.numpy()
      logs['val_' + 'log loss'] = validation_MSE.cpu().data.numpy()

      liveloss.update(logs)
      liveloss.draw()

      print('Epoch: ', epoch, '| train loss: %e' % train_MSE.cpu().data.numpy(), '| valid loss: %e' % validation_MSE.cpu().data.numpy())

      
  return autoencoder
