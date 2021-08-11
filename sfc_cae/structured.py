from sfc_cae.simple_hilbert import *
from sfc_cae.advection_block_analytical import *
import space_filling_decomp_new as sfc
import numpy as np  # Numpy
import scipy.sparse.linalg as spl
import scipy.linalg as sl
import scipy.sparse as sp
from sfc_cae.utils import *


def loadsimulation(data_dir, simulaion_steps, simulaion_num, reshape = False):
    '''
    Load a single simulation generated in sfc_cae.advection_block_analytical.py
    ---
    data_dir: [string] the abosolute address of the root dir of the simulation folders
    simulation_steps: [int] the steps of each simulation, defined in the run_simulation_advection() class
    simulaion_num: [int] index of the simulation to load
    reshape: [boolean] whether reshape this 1d-array to 2d, default is False.
    ---
    Returns:
    
    tensor: [torch.floatTensor] the tensor of the loaded simulation, of shape tuple(shape of each simulation tensor)   

    '''
    for i in range(simulaion_steps + 1):
        iter_data = np.loadtxt(F'{data_dir}_%d/step_%d.txt'% (simulaion_num, i))
        if reshape: 
            size = np.sqrt(iter_data.shape[0]).astype('int')
            iter_data = iter_data.reshape((size, size))
        if i != 0: tensor = torch.cat((tensor, torch.unsqueeze(torch.from_numpy(iter_data), 0)), 0)
        else: 
           tensor = torch.unsqueeze(torch.from_numpy(iter_data), 0)
      
    return tensor

def load_tensor(data_dir, simulation_indexes):
    '''
    Load simulation tensors by the simulation generated in sfc_cae.advection_block_analytical.py
    ---
    data_dir: [string] the abosolute address of the root dir of the simulation folders
    simulation_indexes: [1d-array] the indices of the simulation number to be loaded.
    ---
    Returns:

    tensor: [torch.floatTensor] the whole tensors of the loaded simulations, of shape[len(simulation_indexes), tuple(shape of each simulation tensor)]
    '''

    total = len(simulation_indexes)
    cnt_progress = 0
    bar=progressbar.ProgressBar(maxval=total)
    tensor = loadsimulation(data_dir, simulaion_steps, simulation_indexes[0])
    cnt_progress+=1
    bar.update(cnt_progress)    
    for i in range(1, total):
        tensor = torch.cat((tensor, loadsimulation(data_dir, simulaion_steps, simulation_indexes[i])))
        cnt_progress+=1
        bar.update(cnt_progress)          
    bar.finish()
    return tensor

def index_split(train_ratio, valid_ratio, test_ratio, total_num = 500):
    '''
    Get random spilting indexes, for train, valid and test set.
    ---
    train_ratio: [float] The ratio of the number of train set.
    valid_ratio: [float] The ratio of the number of valid set.
    test_ratio: [float] The ratio of the number of test set.
    total_num: [int] The total number of the dataset (snapshots)
    ---
    Returns:

    train_index: [1d-array] The indices of the train set, shape of [total_num * train_ratio]
    valid_index: [1d-array] The indices of the valid set, shape of [total_num * valid_ratio]
    test_index: [1d-array] The indices of the test set, shape of [total_num * test_ratio]
    '''

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
    '''
    Get the Fortran CSRformat of the connectivity matrix of a square grid.
    ---
    N: [int] the size of the square grid.
    ---
    Returns:

    findm: [1d-array] The Intptr of the CSRMatrix, start index is 1
    colm: [1d-array] The Column Indices of the CSRMatrix, start index is 1
    ncolm: [int] The number of non-zeros in this sparse Matrix, equal to findm[-1]
    '''

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
    '''
    Get the hilbert_curves on a structured square grid of size [size]^ 2.
    ---
    size: [int] the size of the square grid.
    num: [int] the number of space-filling curves want to generate.
    ---
    Returns:

    curve_lists: [list of 1d-arrays] the list of space-filling curve orderings, of shape (number of curves, number of Nodes).
    inv_lists: [list of 1d-arrays] the list of inverse space-filling curve orderings, of shape (number of curves, number of Nodes).
    '''

    Hilbert_index = hilbert_space_filling_curve(size)
    invert_Hilbert_index = np.argsort(Hilbert_index)
    if num == 1: return [Hilbert_index], [invert_Hilbert_index]
    elif num == 2:
        Hilbert_index_2 = Hilbert_index.reshape(size, size).T.flatten()
        invert_Hilbert_index_2 = np.argsort(Hilbert_index_2)
        return [Hilbert_index, Hilbert_index_2], [invert_Hilbert_index, invert_Hilbert_index_2]

def get_MFT_RNN_curves_structured(size, num):
    '''
    Get the MFT_RNN_curves on a structured square grid of size [size]^ 2.
    ---
    size: [int] the size of the square grid.
    num: [int] the number of space-filling curves want to generate.
    ---
    Returns:

    curve_lists: [list of 1d-arrays] the list of space-filling curve orderings, of shape (number of curves, number of Nodes).
    inv_lists: [list of 1d-arrays] the list of inverse space-filling curve orderings, of shape (number of curves, number of Nodes).
    '''

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
   
def plot_trace_structured_2D(sfc_ordering, levels = 16):
    '''
    Plot the trace of index ordering for a structured square grid.
    ---
    sfc_ordering: [1d-array] the space-filling orderings on a square grid.
    levels: [int] the colorbar level for the index ordering, default is 16.
    ---
    Returns:

    Nonetype: The plot of the index ordering on a structured square grid.
    '''
    num = int(np.sqrt(len(sfc_ordering)))
    x_coords = sfc_ordering // num 
    y_coords = sfc_ordering % num 
    fig, ax = plt.subplots(figsize=(15, 15))
    cuts = np.linspace(0, len(sfc_ordering), levels + 1).astype(np.int32)
    for i in range(levels):
        ax.plot(x_coords[cuts[i]:cuts[i+1]], y_coords[cuts[i]:cuts[i+1]], '-')
    plt.axis('off')
    plt.show() 

def plot_contour_structured_2D(sfc_ordering, levels = 256, cmap = None):
    '''
    Generate the contour plot of the index ordering for a structured square grid.
    ---
    sfc_ordering: [1d-array] the space-filling orderings on a square grid.
    levels: [int] the colorbar level for the index ordering, default is 256.
    cmap: [cmap] the colormap for the contour plot, default is None (i.e. viridis)
    ---
    Returns:

    Nonetype: The contour plot of the index ordering on a structured square grid.    
    '''
    num = int(np.sqrt(len(sfc_ordering)))
    fig, ax = plt.subplots(figsize=(15, 15))
    xx, yy = np.meshgrid(np.arange(0, num), np.arange(0, num))
    cset = plt.contourf(xx, yy, np.argsort(sfc_ordering).reshape(size, size), levels=levels, cmap=cmap)
    ax.axis('off')
    plt.show()

def csr_to_edges(findm, colm, direct = False):
    '''
    Convert the Intptr and Indices of Fortran (start with 1) compressed row array 
    to a 2d-array of Edges, remove self-loop.
    ---
    findm: [1d-array] the Intptr of the CSRMatrix, start index is 1
    colm: [1d-array] The Column Indices of the CSRMatrix, start index is 1
    direct: [boolean] Whether the graph is directed or undirected, default is undirected.
    ---
    Returns:

    edge_list: [2d-array] The list of edges, shape of (number of Nodes, 2).
    '''

    findm = findm - 1
    colm = colm - 1
    csr_1 = sp.csr_matrix((np.ones(findm[-1]), colm, findm))
    coo_1 = csr_1.tocoo()
    adjency_indice = (coo_1.row != coo_1.col) 
    edge_list = (np.vstack((coo_1.row, coo_1.col)).T)[adjency_indice]
    if not direct: edge_list = np.sort(edge_list, axis = -1)
    return np.unique(edge_list, axis = 0)

def filled_edges_for_sfcs(edge_list, sfc_orderings):
    '''
    Obtain the number of edges filled by the sfc_list of an abstract graph.
    ---
    edge_list: [2d-array] list of edges, shape of (Number of Nodes, 2)
    sfc_orderings: [list of 1d-arrays] list of space-filling orderings, shape of (Number of sfcs, Number of Nodes)
    ---
    Returns:

    Some print indices the (edges_filled) / (total edges at the graph) by (number) space-filling curves.
    '''
    cnt = 0

    for sfc_num in sfc_orderings:
        vertices_1 = sfc_num[:-1]
        vertices_2 = sfc_num[1:]
        edges_sfc = np.unique(np.sort(np.vstack((vertices_1, vertices_2)).T, axis = -1), axis = 0)
        if cnt == 0: exist_edges = edges_sfc
        exist_edges = np.unique(np.vstack((exist_edges, edges_sfc)), axis = 0)
        common_edges = np.array([x for x in set(tuple(x) for x in exist_edges) & set(tuple(x) for x in edge_list)])
        cnt += 1
        print('filled adjacencies by the %d sfcs : %d / %d' % (cnt, len(common_edges), len(edge_list)))
