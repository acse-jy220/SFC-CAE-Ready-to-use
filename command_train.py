from structured import *
from training import *
from util import *
from sfc_cae import *
import sys

if(len(sys.argv) > 1):
   parameters = read_parameters(sys.argv[1])
else:
   parameters = read_parameters()

print(parameters)

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

print(coords)
print(cells)

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
# if parameters['data_type'] == 'tensors':
#    if parameters['got_min_max'] == 'True':
#       path_data = find_min_and_max(parameters['data_dir'], False)
#    elif parameters['got_min_max'] == 'False':
#       path_data = find_min_and_max(parameters['data_dir'], True) 

   # if parameters['activation'] == 'ReLU':
   #    full_tensor =  MyTensorDataset(path_data, components, 0, 1)
   # elif parameters['activation'] == 'Tanh':
   #    full_tensor =  MyTensorDataset(path_data, components, -1, 1)

samples = len(glob.glob(parameters['data_dir']))

print('structured ', structured, '\n', 'activation ', activation, '\n', 'self concat ', self_concat, '\n', 'sfc_nums ', sfc_nums, '\n')
print('dims_latent ', dims_latent, '\n', 'components ', components, '\n', 'nearest_neighbouring ', nearest_neighbouring, '\n')
print('visualize ', visualize, '\n', 'output', output, '\n', 'sample number ', samples, '\n')

# if specifies sfc_file and inv_sfc_file
if parameters['sfc_file'] != 'None':
   print('reading sfc nums......')
   space_filling_orderings = list(torch.load(parameters['sfc_file']).detach().numpy())
   space_filling_orderings = space_filling_orderings[:sfc_nums]
   print(space_filling_orderings)
   if parameters['inv_sfc_file'] != 'None':
      print('reading inverse sfc nums......')
      invert_space_filling_orderings = list(torch.load(parameters['inv_sfc_file']).detach().numpy())
      invert_space_filling_orderings = invert_space_filling_orderings[:sfc_nums]
      print(invert_space_filling_orderings)
else:
   print('generating sfc and inverse nums......')
   space_filling_orderings, invert_space_filling_orderings = get_sfc_curves_from_coords(coords, sfc_nums)
   print(space_filling_orderings)
   print(invert_space_filling_orderings)

train_ratio = 15/17
valid_ratio = 1/17
test_ratio = 1/17
train_index, valid_index, test_index = index_split(train_ratio, valid_ratio, test_ratio, total_num = samples)

train_index = train_index - 1
valid_index = valid_index - 1
test_index = test_index - 1
# # print(train_index, valid_index, test_index)

# split_1 = int(samples * train_ratio)
# split_2 = -1 + int(samples * test_ratio)

# train_set = MyTensorDataset(path_data[:split_1], components, 0, 1)
# valid_set = MyTensorDataset(path_data[split_1:], components, 0, 1)
# # test_set = MyTensorDataset(path_data[split_2:-1], components, 0, 1)
# test_set = []


if parameters['data_type'] == 'vtu' or parameters['data_type'] == 'one_tensor':
   train_set = full_tensor[train_index].float()
   valid_set = full_tensor[valid_index].float()
   test_set = full_tensor[test_index].float()
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
        train_set = MyTensorDataset(get_path_data(parameters['data_dir'], train_index), 0, 1)
        valid_set = MyTensorDataset(get_path_data(parameters['data_dir'], valid_index), 0, 1)
        test_set = MyTensorDataset(get_path_data(parameters['data_dir'], test_index), 0, 1)
     elif parameters['activation'] == 'Tanh':
        train_set = MyTensorDataset(get_path_data(parameters['data_dir'], train_index), -1, 1)
        valid_set = MyTensorDataset(get_path_data(parameters['data_dir'], valid_index), -1, 1)
        test_set = MyTensorDataset(get_path_data(parameters['data_dir'], test_index), -1, 1) 

print('length of train set:', len(train_set), '\n')
print('length of valid set:',len(valid_set), '\n')
print('length of test set:',len(test_set), '\n')

# training parameters
batch_size = int(parameters['batch_size'])
lr = float(parameters['lr'])
n_epoches = int(parameters['n_epoches'])
seed = int(parameters['seed'])
dimension = int(parameters['dimension'])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=len(valid_set), shuffle=True)

input_size = space_filling_orderings[0].shape[0]

autoencoder = SFC_CAE(input_size,
                      dimension,
                      components,
                      structured,
                      self_concat,
                      nearest_neighbouring,
                      dims_latent,
                      space_filling_orderings, 
                      invert_space_filling_orderings,
                      activation = activation)

autoencoder = train_model(autoencoder, 
                          train_loader = train_loader,
                          valid_loader = valid_loader,
                          n_epochs = n_epoches, 
                          lr = lr, 
                          seed = seed,
                          visualize = visualize)

       










