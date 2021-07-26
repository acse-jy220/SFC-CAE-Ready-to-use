import sys
from structured import *
from training import *
from util import *
from sfc_cae import *

if(len(sys.argv) > 1):
       parameters = read_parameters(sys.argv[1])
else:
   parameters = read_parameters()

data_path = '/rds/general/user/jy220/home/slugflow/'

tk = torch.from_numpy(np.array([0.7581, 0.0418, 0.1840, 0.1464]))
tb = torch.from_numpy(np.array([-0.3816, -0.3618, -0.0308,  0.1446]))

vtu_fields = list(parameters['vtu_fields'].split(','))
for i in range(len(vtu_fields)): 
    vtu_fields[i] = vtu_fields[i].strip()

autoencoder = torch.load('/rds/general/user/jy220/home/results/Slugflow_nearest_neighbouring_True_SFC_nums_3_lr_0.0001_n_epoches_100.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder = autoencoder.to(device)

result_vtu_to_vtu(data_path, vtu_fields, autoencoder, tk, tb, start_index = 1100, end_index= 1200, model_device = device)