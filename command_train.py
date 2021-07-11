from structured import *
from training import *
from util import *
from sfc_cae import *

parameters = read_parameters()


# if vtu file folder
if parameters['data_type'] == 'vtu':
   vtu_fields = list(parameters['vtu_fields'].split(',')
   for vtu_field in vtu_fields:
       vtu_field = vtu_field.strip()
   full_tensor, coords, cells = read_in_files(parameters['data_dir'], vtu_fields = vtu_fields)

# if one tensor
if parameters['data_type'] == 'one_tensor':
   full_tensor = torch.load(parameters['data_dir'])
if parameters['coords_file'] != 'None':
   coords = np.loadtxt(parameters['coords_file'])
if parameters['cells_file'] != 'None':
   cells = np.loadtxt(parameters['cells_file'])

# other parameters
if parameters['structured'] == 'True':
   structured = True
   if parameters['activation'] == 'None':
      activation = nn.ReLU()
elif parameters['structured'] == 'False':
   structured = False
   if parameters['activation'] == 'None':
      activation = nn.Tanh()

 
      

# if tensors folder
if parameters['data_type'] == 'tensors':
   path_data = find_min_and_max(parameters['data_dir'])





