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
elif parameters['activation'] == 'SELU':
   activation = nn.SELU()

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

optimizer_type = parameters['optimizer']

# read in extra arguments
kwargs={}

if 'kernel_size' in parameters.keys():
   kwargs.update({'kernel_size': int(parameters['kernel_size'])})
if 'stride' in parameters.keys():
   kwargs.update({'stride': int(parameters['stride'])})
if 'padding' in parameters.keys():
   kwargs.update({'padding': int(parameters['padding'])})
if 'increase_multi' in parameters.keys():
   kwargs.update({'increase_multi': int(parameters['increase_multi'])})
if 'direct_neigh' in parameters.keys():
   kwargs.update({'direct_neigh': bool(parameters['direct_neigh'])})
if 'share_sp_weights' in parameters.keys():
   kwargs.update({'share_sp_weights': bool(parameters['share_sp_weights'])})
if 'share_conv_weights' in parameters.keys():
   kwargs.update({'share_conv_weights': bool(parameters['share_conv_weights'])})
if 'place_center' in parameters.keys():
   kwargs.update({'place_center': bool(parameters['place_center'])})    
else: kwargs.update({'place_center': False}) # always place the unstructured FEM mesh at the center at the 2nd sfc
if 'num_final_channels' in parameters.keys():
       kwargs.update({'num_final_channels': int(parameters['num_final_channels'])})    
if 'interpolation' in parameters.keys():
       kwargs.update({'interpolation': bool(parameters['interpolation'])}) 
if 'conv_smooth_layer' in parameters.keys():
       kwargs.update({'conv_smooth_layer': bool(parameters['conv_smooth_layer'])}) 


samples = len(glob.glob(parameters['data_dir'] + '*'))

print('vtu fields: ', vtu_fields, '\n', 'structured ', structured, '\n', 'activation ', activation, '\n', 'self concat ', self_concat, '\n', 'sfc_nums ', sfc_nums, '\n')
print('dims_latent ', dims_latent, '\n', 'components ', components, '\n', 'nearest_neighbouring ', nearest_neighbouring, '\n')
print('visualize ', visualize, '\n', 'sample number ', samples, '\n')

# if specifies sfc_file and inv_sfc_file
if parameters['sfc_file'] != 'None':
   print('reading sfc nums......')
   space_filling_orderings = torch.tensor(torch.load(parameters['sfc_file']))
   space_filling_orderings = space_filling_orderings[:sfc_nums]
   print(space_filling_orderings)
   if parameters['inv_sfc_file'] != 'None':
      print('reading inverse sfc nums......')
      invert_space_filling_orderings = torch.tensor(torch.load(parameters['inv_sfc_file']))
      invert_space_filling_orderings = invert_space_filling_orderings[:sfc_nums]
      print(invert_space_filling_orderings)
else:
   print('generating sfc and inverse nums......')
   space_filling_orderings, invert_space_filling_orderings = get_sfc_curves_from_coords(coords, sfc_nums)
   print(space_filling_orderings)
   print(invert_space_filling_orderings)

if parameters['tk_file'] != 'None':
   tk = torch.load(parameters['tk_file'])
if parameters['tb_file'] != 'None':
   tb = torch.load(parameters['tb_file']) 

# parallel mode
if 'parallel_mode' in parameters.keys():
      parallel_mode = parameters['parallel_mode']
else: parallel_mode = 'DP'

# training parameters
batch_size = int(parameters['batch_size'])
lr = float(parameters['lr'])
n_epoches = int(parameters['n_epoches'])
seed = int(parameters['seed'])
dimension = int(parameters['dimension'])
samples = len(glob.glob(parameters['data_dir'] + '*'))
if parameters['reconstruct_start_index'] != 'None':
   start_index = int(parameters['reconstruct_start_index'])
else: start_index = 0
if parameters['reconstruct_end_index'] != 'None':
   end_index = int(parameters['reconstruct_end_index'])
else: end_index = samples
if 'output_linear' in parameters.keys() and parameters['output_linear'] == 'True': output_linear = True
else: output_linear = False

if 'AE_type' in parameters.keys() and parameters['AE_type'] == 'md': md = True
else: md = False

if parameters['mode'] == 'train':
   if parameters['load_index'] == 'True':
      train_index = np.load(parameters['train_index'])
      valid_index = np.load(parameters['valid_index'])
      test_index = np.load(parameters['test_index'])
   else:
      if 'train_index' not in parameters.keys():
         train_ratio = 15/17
         valid_ratio = 1/17
         test_ratio = 1/17 
      else:
         if not parameters['train_index'] == 'None':
            train_ratio = float(parameters['train_index'])
            valid_ratio = float(parameters['valid_index'])
            test_ratio = float(parameters['test_index'])    
          
      if start_index is not None or end_index is not None: samples = end_index - start_index
      train_index, valid_index, test_index = index_split(train_ratio, valid_ratio, test_ratio, total_num = samples)
      train_index = train_index + start_index
      valid_index = valid_index + start_index
      test_index = test_index + start_index
      np.save('train_index', train_index)
      np.save('valid_index', valid_index)
      np.save('test_index', test_index)
   full_path = get_path_data((parameters['data_dir']), file_format='pt')
   train_path = get_path_data((parameters['data_dir']), train_index - 1, file_format='pt')
   valid_path = get_path_data((parameters['data_dir']), valid_index - 1, file_format='pt')
   test_path = get_path_data((parameters['data_dir']), test_index - 1, file_format='pt')

   if parameters['data_type'] == 'vtu' or parameters['data_type'] == 'one_tensor':
      train_set = full_tensor[train_index].float()
      valid_set = full_tensor[valid_index].float()
      test_set = full_tensor[test_index].float()
      # standardlisation
      if parameters['activation'] == 'ReLU':
         train_set, train_k, train_b = standardlize_tensor(train_set, lower = 0, upper = 1)
         valid_set, valid_k, valid_b = standardlize_tensor(valid_set, lower = 0, upper = 1)
         test_set, test_k, test_b = standardlize_tensor(test_set, lower = 0, upper = 1)
      elif parameters['activation'] == 'Tanh' or parameters['activation'] == 'SELU':
         train_set, train_k, train_b = standardlize_tensor(train_set, lower = -1, upper = 1)
         valid_set, valid_k, valid_b = standardlize_tensor(valid_set, lower = -1, upper = 1)
         test_set, test_k, test_b = standardlize_tensor(test_set, lower = -1, upper = 1)       
   elif parameters['data_type'] == 'tensors':
      if parameters['activation'] == 'ReLU':
         if parameters['tk_file'] != 'None' and parameters['tb_file'] != 'None':
            # full_set = MyTensorDataset(glob.glob(parameters['data_dir'] + '*'),  0, 1, tk, tb)
            train_set = MyTensorDataset(train_path,  0, 1, tk, tb, md = md)
            valid_set = MyTensorDataset(valid_path,  0, 1, tk, tb, md = md)
            test_set = MyTensorDataset(test_path,  0, 1, tk, tb, md = md)
         else: 
            full_set = MyTensorDataset(full_path,  0, 1, md = md)
            train_set = MyTensorDataset(train_path,  0, 1, full_set.tk, full_set.tb, md = md)
            valid_set = MyTensorDataset(valid_path,  0, 1, full_set.tk, full_set.tb, md = md)
            test_set = MyTensorDataset(test_path,  0, 1, full_set.tk, full_set.tb, md = md)
            tk = full_set.tk
            tb = full_set.tb
            # train_set, valid_set, test_set = torch.utils.data.dataset.random_split(full_set, [len(train_index), len(valid_index), len(test_index)])
      elif parameters['activation'] == 'Tanh' or parameters['activation'] == 'SELU':
         if parameters['tk_file'] != 'None' and parameters['tb_file'] != 'None':
            train_set = MyTensorDataset(train_path,  -1, 1, tk, tb, md = md)
            valid_set = MyTensorDataset(valid_path,  -1, 1, tk, tb, md = md)
            test_set = MyTensorDataset(test_path,  -1, 1, tk, tb, md = md)
         else: 
            full_set = MyTensorDataset(full_path, -1, 1, md = md)
            train_set = MyTensorDataset(train_path,  -1, 1, full_set.tk, full_set.tb, md = md)
            valid_set = MyTensorDataset(valid_path,  -1, 1, full_set.tk, full_set.tb, md = md)
            test_set = MyTensorDataset(test_path,  -1, 1, full_set.tk, full_set.tb, md = md)
            # train_set, valid_set, test_set = torch.utils.data.dataset.random_split(full_set, [len(train_index), len(valid_index), len(test_index)])
            tk = full_set.tk
            tb = full_set.tb

   print('length of train set:', len(train_set), '\n')
   print('length of valid set:',len(valid_set), '\n')
   print('length of test set:',len(test_set), '\n')
       
   if parallel_mode == 'DP':
     train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
     valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
     test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
   # elif parameters['parallel_mode'] == 'DDP':
   #   train_sampler = distributed.DistributedSampler(train_set, num_replicas=torch.cuda.device_count(), shuffle=True)
   #   valid_sampler = distributed.DistributedSampler(valid_set, num_replicas=torch.cuda.device_count(), shuffle=True)
   #   test_sampler = distributed.DistributedSampler(test_set, num_replicas=torch.cuda.device_count(), shuffle=True)
   #   train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler)
   #   valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, sampler=valid_sampler)
   #   test_loader = DataLoader(dataset=test_set, batch_size=batch_size, sampler=test_sampler)
         
# whether save the model
if parameters['save_path'] != 'None':
   save_path = parameters['save_path']
else: save_path = None

input_size = space_filling_orderings[0].shape[0]

if md:
      if 'second_sfc' in parameters.keys():
          second_sfc = torch.tensor(torch.load(parameters['second_sfc']))
          if len(second_sfc.shape) > 1: second_sfc = second_sfc[0]
          print('second_sfc_shape:', second_sfc.shape)
      else: second_sfc = None
      if 'reduce_strategy' in parameters.keys():
          reduce_strategy = parameters['reduce_strategy']
      else: reduce_strategy = 'truncate'    
      autoencoder = SFC_CAE_md(input_size,
                      dimension,
                      components,
                      structured,
                      self_concat,
                      nearest_neighbouring,
                      dims_latent,
                      space_filling_orderings, 
                      invert_space_filling_orderings,
                      activation = activation,
                      variational = variational,
                      output_linear = output_linear,
                      sfc_mapping_to_structured = second_sfc,
                      reduce_strategy = reduce_strategy,
                      **kwargs)
      
      print('num_neigh_md:', autoencoder.encoder.num_neigh_md)
else:
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
                      variational = variational,
                      output_linear = output_linear)

if parameters['mode'] == 'train':
   if parallel_mode == 'DP': 
      print('\ntraining with DataParallel...\n')   
      autoencoder = train_model(autoencoder, 
                                train_loader = train_loader,
                                valid_loader = valid_loader,
                                test_loader = test_loader,
                                optimizer_type = optimizer_type,
                                state_load = state_load,
                                n_epochs = n_epoches, 
                                varying_lr = change_lr,
                                lr = lr, 
                                seed = seed,
                                visualize = visualize,
                                save_path = save_path,
                                dict_only = True,
                                parallel_mode = parallel_mode)
   elif parallel_mode == 'DDP':
      print('\ntraining with DataDistributedParallel...\n')  
      if __name__ == '__main__':
        torch.multiprocessing.freeze_support()
      #   mp.spawn(setup_DDP, args=(), nprocs=torch.cuda.device_count(), join=True)
      #   train_sampler = distributed.DistributedSampler(train_set, num_replicas=torch.cuda.device_count(), shuffle=True)
      #   valid_sampler = distributed.DistributedSampler(valid_set, num_replicas=torch.cuda.device_count(), shuffle=True)
      #   test_sampler = distributed.DistributedSampler(test_set, num_replicas=torch.cuda.device_count(), shuffle=True)
      #   train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler)
      #   valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, sampler=valid_sampler)
      #   test_loader = DataLoader(dataset=test_set, batch_size=batch_size, sampler=test_sampler)
        mp.spawn(train_model_DDP,
               args=(autoencoder, 
                     train_set,
                     valid_set,
                     test_set,
                     batch_size,
                     optimizer_type,
                     state_load,
                     n_epoches, 
                     change_lr,
                     lr,
                     visualize, 
                     seed,
                     save_path,
                     True),
               nprocs=torch.cuda.device_count(),
               join=True)      
   
else: 
   autoencoder.load_state_dict(torch.load(state_load)['model_state_dict'])
   device = torch.device("cpu")
   print('------------------------------------\n')
   print('Entering Reconstruction mode......')
   print('------------------------------------\n')

if parameters['reconstructed_path'] != 'None':
   result_vtu_to_vtu(parameters['vtu_dir'], parameters['reconstructed_path'], vtu_fields, autoencoder, tk = tk, tb = tb, start_index = start_index, end_index = end_index, model_device = device)



       










