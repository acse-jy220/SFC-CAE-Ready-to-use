from structured import *
from training import *
from util import *
from sfc_cae import *

parameters = read_parameters()


# if vtu file folder
if parameters['data_type'] == 'vtu':
   vtu_fields = list(parameters['vtu_fields'].split(','))
   for i in range(len(vtu_fields)): 
       vtu_fields[i] = vtu_fields[i].strip()
   full_tensor, coords, cells = read_in_files(parameters['data_dir'], vtu_fields = vtu_fields)
   samples = full_tensor.shape[0]

# if one tensor
if parameters['data_type'] == 'one_tensor':
   full_tensor = torch.load(parameters['data_dir'])
   samples = full_tensor.shape[0]

# load coords and cells 
if parameters['coords_file'] != 'None':
   coords = torch.load(parameters['coords_file']).detach().numpy()
if parameters['cells_file'] != 'None':
   cells = np.load(parameters['cells_file'], allow_pickle= True)

# other parameters
if parameters['structured'] == 'True':
   structured = True
   if parameters['activation'] == 'None':
      activation = nn.ReLU()
elif parameters['structured'] == 'False':
   structured = False
   if parameters['activation'] == 'None':
      activation = nn.Tanh()

if parameters['activation'] == 'ReLU':
   activation = nn.ReLU()
elif parameters['activation'] == 'Tanh':
   activation = nn.Tanh()

self_concat = int(parameters['self_concat'])
sfc_nums = int(parameters['sfc_nums'])
dims_latent = int(parameters['dims_latent'])
components = int(parameters['components'])

if parameters['nearest_neighbouring'] == 'True':
    nearest_neighbouring = True
elif parameters['nearest_neighbouring'] == 'False':
    nearest_neighbouring = False

if parameters['visualize'] == 'True':
    visualize = True
elif parameters['visualize'] == 'False':
    visualize = False

if parameters['output_reconstructed'] == 'True':
   output = True
elif parameters['output_reconstructed'] == 'False':
   output = False

# if tensors folder
if parameters['data_type'] == 'tensors':
   if parameters['got_min_max'] == 'True':
      path_data = find_min_and_max(parameters['data_dir'], True)
   elif parameters['got_min_max'] == 'False':
      path_data = find_min_and_max(parameters['data_dir'], False) 

   if parameters['activation'] == 'ReLU':
      full_tensor =  MyTensorDataset(path_data, components, 0, 1)
   elif parameters['activation'] == 'Tanh':
      full_tensor =  MyTensorDataset(path_data, components, -1, 1)
   samples = len(full_tensor)

# if specifies sfc_file and inv_sfc_file
if parameters['sfc_file'] != 'None':
   space_filling_orderings = list(np.loadtxt(parameters['sfc_file'], delimiter=',').T)
   if parameters['inv_sfc_file'] != 'None':
      inv_space_filling_orderings = list(np.loadtxt(parameters['inv_sfc_file'], delimiter = '.').T)
else:
   space_filling_orderings, invert_space_filling_orderings = get_sfc_curves_from_coords(coords, sfc_nums)

train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1
train_index, valid_index, test_index = index_split(train_ratio, valid_ratio, test_ratio, total_num = samples)

if parameters['data_type'] == 'vtu' or parameters['data_type'] == 'one_tensor':
   train_set = full_tensor[train_index - 1]
   valid_set = full_tensor[valid_index - 1]
   test_set = full_tensor[test_index - 1]
   # standardlisation
   if parameters['activation'] == 'ReLU':
      train_set, train_k, train_b = standardlize_tensor(train_set, lower = 0, upper = 1)
      valid_set, valid_k, valid_b = standardlize_tensor(valid_set, lower = 0, upper = 1)
      test_set, test_k, test_b = standardlize_tensor(test_set, lower = 0, upper = 1)
   elif parameters['activation'] == 'Tanh':
      train_set, train_k, train_b = standardlize_tensor(train_set, lower = -1, upper = 1)
      valid_set, valid_k, valid_b = standardlize_tensor(valid_set, lower = -1, upper = 1)
      test_set, test_k, test_b = standardlize_tensor(test_set, lower = -1, upper = 1)       
elif parameters['data_type'] == 'tensors':
     if parameters['activation'] == 'ReLU':
        train_set = MyTensorDataset(path_data[train_index - 1], components, 0, 1)
        valid_set = MyTensorDataset(path_data[valid_index - 1], components, 0, 1)
        test_set = MyTensorDataset(path_data[test_index - 1], components, 0, 1)
     elif parameters['activation'] == 'Tanh':
        train_set = MyTensorDataset(path_data[train_index - 1], components, -1, 1)
        valid_set = MyTensorDataset(path_data[valid_index - 1], components, -1, 1)
        test_set = MyTensorDataset(path_data[test_index - 1], components, -1, 1) 


train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=len(valid_set), shuffle=True)

# training parameters
batch_size = int(parameters['batch_size'])
lr = float(parameters['lr'])
n_epoches = int(parameters['n_epoches'])
seed = int(parameters['seed'])

input_size = space_filling_orderings[0].shape[0]

autoencoder = SFC_CAE(input_size,
                      dimension,
                      components,
                      structured,
                      self_concat,
                      nearest_neighbouring,
                      dims_latent,
                      space_filling_orderings, 
                      invert_space_filling_orderings)


# autoencoder = train_model(autoencoder, 
#                           train_loader = train_loader,
#                           valid_loader = valid_loader,
#                           n_epochs = n_epoches, 
#                           lr = lr, 
#                           seed = seed,
#                           visualize = visualize)

       










