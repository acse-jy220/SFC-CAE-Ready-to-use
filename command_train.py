from sfc_cae import *
import sys

if(len(sys.argv) > 1):
   parameters = read_parameters(sys.argv[1])
else:
   parameters = read_parameters()

print(parameters)

# get vtu_fields
vtu_fields = list(parameters['vtu_fields'].split(','))
for i in range(len(vtu_fields)): 
   vtu_fields[i] = vtu_fields[i].strip()

# if vtu file folder
if parameters['data_type'] == 'vtu':
   full_tensor, coords, cells = read_in_files(parameters['data_dir'], vtu_fields = vtu_fields)
   samples = full_tensor.shape[0]

# if one tensor
if parameters['data_type'] == 'one_tensor':
   full_tensor = torch.load(parameters['data_dir'])
   samples = full_tensor.shape[0]

# load coords and cells 
if parameters['coords_file'] != 'None':
   coords = torch.load(parameters['coords_file']).detach().numpy()
else:
   coords = meshio.read(glob.glob(parameters['data_dir'] + '*')[0]).points

if parameters['cells_file'] != 'None':
   cells = np.load(parameters['cells_file'], allow_pickle= True)
else:
   cells = meshio.read(glob.glob(parameters['data_dir'] + '*')[0]).cells_dict

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

if parameters['changing_lr'] == 'True':
   change_lr = True
elif parameters['changing_lr'] == 'False':
   change_lr = False

if parameters['variational'] == 'True':
   variational = True
elif parameters['variational'] == 'False':
   variational = False

if parameters['visualize'] == 'True':
    visualize = True
elif parameters['visualize'] == 'False':
    visualize = False

if parameters['state_load'] != 'None':
   state_load = parameters['state_load']
else: state_load = None

optimizer = parameters['optimizer']

samples = len(glob.glob(parameters['data_dir'] + '*'))

print('vtu fields: ', vtu_fields, '\n', 'structured ', structured, '\n', 'activation ', activation, '\n', 'self concat ', self_concat, '\n', 'sfc_nums ', sfc_nums, '\n')
print('dims_latent ', dims_latent, '\n', 'components ', components, '\n', 'nearest_neighbouring ', nearest_neighbouring, '\n')
print('visualize ', visualize, '\n', 'sample number ', samples, '\n')

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
        full_set = MyTensorDataset(glob.glob(parameters['data_dir'] + '*'),  0, 1)
        train_set, valid_set, test_set = torch.utils.data.dataset.random_split(full_set, [len(train_index), len(valid_index), len(test_index)])
     elif parameters['activation'] == 'Tanh':
        full_set = MyTensorDataset(glob.glob(parameters['data_dir'] + '*'), -1, 1)
        train_set, valid_set, test_set = torch.utils.data.dataset.random_split(full_set, [len(train_index), len(valid_index), len(test_index)])

print('length of train set:', len(train_set), '\n')
print('length of valid set:',len(valid_set), '\n')
print('length of test set:',len(test_set), '\n')

# training parameters
batch_size = int(parameters['batch_size'])
lr = float(parameters['lr'])
n_epoches = int(parameters['n_epoches'])
seed = int(parameters['seed'])
dimension = int(parameters['dimension'])
if parameters['reconstruct_start_index'] != 'None':
   reconstruct_start_index = int(parameters['reconstruct_start_index'])
else: reconstruct_start_index = None
if parameters['reconstruct_end_index'] != 'None':
   reconstruct_end_index = int(parameters['reconstruct_end_index'])
else: reconstruct_end_index = None

if parameters['save_path'] != 'None':
   save_path = parameters['save_path']
else: save_path = None

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

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
                      activation = activation,
                      variational = variational)

autoencoder = train_model(autoencoder, 
                          train_loader = train_loader,
                          valid_loader = valid_loader,
                          test_loader = test_loader,
                          optimizer = optimizer,
                          state_load = state_load,
                          n_epochs = n_epoches, 
                          varying_lr = change_lr,
                          lr = lr, 
                          seed = seed,
                          visualize = visualize,
                          save_path = save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if parameters['reconstructed_path'] != 'None':
   result_vtu_to_vtu(parameters['vtu_dir'], parameters['reconstructed_path'], vtu_fields, autoencoder, tk = full_set.tk, tb = full_set.tb, variational = variational, start_index = reconstruct_start_index, end_index = reconstruct_end_index, model_device = device, dimension = dimension)



       










