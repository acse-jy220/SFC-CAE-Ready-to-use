from simple_hilbert import *
from advection_block_analytical import *
import space_filling_decomp_new as sfc
import numpy as np  # Numpy
import scipy.sparse.linalg as spl
import scipy.linalg as sl
import scipy.sparse as sp
from util import *


def loadsimulation(data_dir, simulaion_steps, simulaion_num, reshape = False):
    for i in range(simulaion_steps + 1):
        iter_data = np.loadtxt(F'{data_dir}_%d/step_%d.txt'% (simulaion_num, i))
        if reshape: 
            size = np.sqrt(iter_data.shape[0]).astype('int')
            iter_data = iter_data.reshape((size, size))
        if i != 0: tensor = torch.cat((tensor, torch.unsqueeze(torch.from_numpy(iter_data), 0)), 0)
        else: 
           tensor = torch.unsqueeze(torch.from_numpy(iter_data), 0)
      
    return tensor

def load_tensor(simulation_indexes):
    total = len(simulation_indexes)
    cnt_progress = 0
    bar=progressbar.ProgressBar(maxval=total)
    tensor = loadsimulation(simulaion_steps, simulation_indexes[0])
    cnt_progress+=1
    bar.update(cnt_progress)    
    for i in range(1, total):
        tensor = torch.cat((tensor, loadsimulation(simulaion_steps, simulation_indexes[i])))
        cnt_progress+=1
        bar.update(cnt_progress)          
    bar.finish()
    return tensor

def index_split(train_ratio, valid_ratio, test_ratio, total_num = 500):
    if train_ratio + valid_ratio + test_ratio != 1:
        raise ValueError("The sum of three input ratios should be 1!")
    total_index = np.arange(1, total_num + 1)
    rng = np.random.default_rng()
    total_index = rng.permutation(total_index)
    knot_1 = int(total_num * train_ratio)
    knot_2 = int(total_num * valid_ratio) + knot_1
    train_index, valid_index, test_index = np.split(total_index, [knot_1, knot_2])
    return train_index, valid_index, test_index

def sparse_square_grid(N):
    n = N ** 2
    
    offsets = [-N, -1, 0, 1, N]
    diags = []
    # coefficient in front of u_{i-N}:
    diags.append(np.ones(n-N))
    # coefficient in front of u_{i-1}:
    diags.append(np.ones(n-1))
    # main diagonal, zero for centre difference in space
    diags.append(np.ones(n))
    # coefficient in front of u_{i+1}:
    diags.append(diags[1])
    # coefficient in front of u_{i+N}:
    diags.append(diags[0])
    
    K = sp.diags(diags, offsets, format='csr')
    
    # loop over left-most column in grid (except first row)
    for i in range(N, n, N):
        K[i, i-1] = 0
        K[i-1, i] = 0
    K.eliminate_zeros()
    
    return K.indptr + 1, K.indices + 1, K.getnnz()

def get_hilbert_curves(size, num):
    Hilbert_index = hilbert_space_filling_curve(size)
    invert_Hilbert_index = np.argsort(Hilbert_index)
    if num == 1: return [Hilbert_index], [invert_Hilbert_index]
    elif num == 2:
        Hilbert_index_2 = Hilbert_index.reshape(size, size).T.flatten()
        invert_Hilbert_index_2 = np.argsort(Hilbert_index_2)
        return [Hilbert_index, Hilbert_index_2], [invert_Hilbert_index, invert_Hilbert_index_2]

def get_MFT_RNN_curves_structured(size, num):
    findm, colm, ncolm  = sparse_square_grid(size)
    curve_lists = []
    inv_lists = []
    ncurve = num
    graph_trim = -10  # has always been set at -10
    starting_node = 0 # =0 do not specifiy a starting node, otherwise, specify the starting node
    whichd, space_filling_curve_numbering = sfc.ncurve_python_subdomain_space_filling_curve(colm, findm, starting_node, graph_trim, ncurve, size**2, ncolm)
    for i in range(space_filling_curve_numbering.shape[-1]):
        curve_lists.append(np.argsort(space_filling_curve_numbering[:,i]))
        inv_lists.append(np.argsort(np.argsort(space_filling_curve_numbering[:,i])))

    return curve_lists, inv_lists
