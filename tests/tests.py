"""
This module contains the tests of the sfc_cae module.

Author: Jin Yu
Github handle: acse-jy220
"""

import pytest
from sfc_cae import *
from pytest import fixture # Use pytest fixtures to avoid reloading variables/objects

#################################################### test for simple_hilbert.py ######################################################

class TestHilbert(object):
    '''
    A simple test for the Hilbert curve generation on 4 * 4, 8 * 8, 16 * 16 grids.
    '''
    @pytest.mark.parametrize('hilbert_4, hilbert_8, hilbert_16', 
    [
        ([ 0,  1,  5,  4,  8, 12, 13,  9, 10, 14, 15, 11,  7,  6,  2,  3],
         [ 0 , 8 , 9 , 1 , 2 , 3 , 11 , 10 ,18 ,19 ,27 ,26 ,25, 17 ,16 ,24 ,32, 33, 41, 40 ,48 ,56 ,57, 49,
 50, 58, 59, 51, 43, 42, 34, 35 ,36, 37, 45, 44, 52, 60 ,61 ,53, 54, 62, 63 ,55, 47, 46 ,38, 39,
 31, 23, 22, 30 ,29, 28, 20, 21, 13, 12,  4 , 5 , 6 ,14, 15,  7], 
         [  0,   1,  17,  16,  32,  48,  49,  33,  34,  50,  51,  35,  19,
        18,   2,   3,   4,  20,  21,   5,   6,   7,  23,  22,  38,  39,
        55,  54,  53,  37,  36,  52,  68,  84,  85,  69,  70,  71,  87,
        86, 102, 103, 119, 118, 117, 101, 100, 116, 115, 114,  98,  99,
        83,  67,  66,  82,  81,  65,  64,  80,  96,  97, 113, 112, 128,
       144, 145, 129, 130, 131, 147, 146, 162, 163, 179, 178, 177, 161,
       160, 176, 192, 193, 209, 208, 224, 240, 241, 225, 226, 242, 243,
       227, 211, 210, 194, 195, 196, 197, 213, 212, 228, 244, 245, 229,
       230, 246, 247, 231, 215, 214, 198, 199, 183, 167, 166, 182, 181,
       180, 164, 165, 149, 148, 132, 133, 134, 150, 151, 135, 136, 152,
       153, 137, 138, 139, 155, 154, 170, 171, 187, 186, 185, 169, 168,
       184, 200, 201, 217, 216, 232, 248, 249, 233, 234, 250, 251, 235,
       219, 218, 202, 203, 204, 205, 221, 220, 236, 252, 253, 237, 238,
       254, 255, 239, 223, 222, 206, 207, 191, 175, 174, 190, 189, 188,
       172, 173, 157, 156, 140, 141, 142, 158, 159, 143, 127, 126, 110,
       111,  95,  79,  78,  94,  93,  77,  76,  92, 108, 109, 125, 124,
       123, 107, 106, 122, 121, 120, 104, 105,  89,  88,  72,  73,  74,
        90,  91,  75,  59,  43,  42,  58,  57,  56,  40,  41,  25,  24,
         8,   9,  10,  26,  27,  11,  12,  13,  29,  28,  44,  60,  61,
        45,  46,  62,  63,  47,  31,  30,  14,  15])
    ])
    def test_hilbert_order_generating(self, hilbert_4, hilbert_8, hilbert_16):
      h4 = hilbert_space_filling_curve(4)
      h8 = hilbert_space_filling_curve(8)
      h16 = hilbert_space_filling_curve(16)

      assert (h4 == hilbert_4).all()
      assert (h8 == hilbert_8).all()
      assert (h16 == hilbert_16).all()

#################################################### test for advection_block_analytical.py ######################################################

@fixture(scope='module')
def advection_block():
    """
    generate a Block simulation
    """
    # generate a Block simulation
    advection_block = run_simulation_advection()
    advection_block()

    return advection_block

@fixture(scope='module')
def advection_gaussian():
    """
    generate a gaussian simulation
    """
    # generate a Gaussian simulation
    advection_gaussian = run_simulation_advection(init_func = gaussian_wave)
    advection_gaussian()

    return advection_gaussian


class TestAdvection(object):
    '''
    Test for advection block/gaussian
    '''

    def test_advection_block(self, advection_block):
       assert (advection_block.full_stage.shape == (41, 128, 128)) # shape test
       assert (advection_block.full_stage[advection_block.full_stage != 0] == 1).all() # the analytical solution only contains 0 and 1 for a block problem.

       end_x0 = advection_block.Lx - advection_block.x0 # end_point in x_coords
       end_y0 = advection_block.Ly - advection_block.y0 # end_point in y_coords

       # test the simulation does not exceed the boundary of the grid.
       assert(end_x0 < advection_block.Lx + advection_block.d/2 and end_x0 > -advection_block.d/2 and end_y0 < advection_block.Ly + advection_block.d/2 and end_y0 > -advection_block.d/2)

    
    def test_advection_gaussian(self, advection_gaussian):
       assert (advection_gaussian.full_stage.shape == (41, 128, 128)) # shape test
       assert (advection_gaussian.full_stage.max() < 1 and advection_gaussian.full_stage.max() > 0) # min_max test

       end_x0 = advection_gaussian.Lx - advection_gaussian.x0 # end_point in x_coords
       end_y0 = advection_gaussian.Ly - advection_gaussian.y0 # end_point in y_coords

       # test the simulation does not exceed the boundary of the grid.
       assert(end_x0 < advection_gaussian.Lx + advection_gaussian.d/2 and end_x0 > -advection_gaussian.d/2 and end_y0 < advection_gaussian.Ly + advection_gaussian.d/2 and end_y0 > -advection_gaussian.d/2)


#################################################### test for structured.py ######################################################

@fixture(scope='module')
def sparse_square_4():
    """
    Fortran CSR matrix for 4 * 4 grid
    """
    findm, colm, ncolm = sparse_square_grid(4)

    return findm, colm, ncolm

@fixture(scope='module')
def sparse_square_8():
    """
    Fortran CSR matrix for 8 * 8 grid
    """
    findm, colm, ncolm = sparse_square_grid(8)

    return findm, colm, ncolm

@fixture(scope='module')
def sparse_square_16():
    """
    Fortran CSR matrix for 16 * 16 grid
    """
    findm, colm, ncolm = sparse_square_grid(16)

    return findm, colm, ncolm

@fixture(scope='module')
def sparse_square_128():
    """
    Fortran CSR matrix for 16 * 16 grid
    """
    findm, colm, ncolm = sparse_square_grid(128)

    return findm, colm, ncolm

class TestStructured(object):
    '''
    Test for functions in structured.py
    '''
    @pytest.mark.parametrize('csr_4_r, csr_4_c, csr_8_r, csr_8_c, csr_16_r, csr_16_n', 
    [   
        ([ 1,  4,  8, 12, 15, 19, 24, 29, 33, 37, 42, 47, 51, 54, 58, 62, 65],
         [ 1,  2,  5,  1,  2,  3,  6,  2,  3,  4,  7,  3,  4,  8,  1,  5,  6,
        9,  2,  5,  6,  7, 10,  3,  6,  7,  8, 11,  4,  7,  8, 12,  5,  9,
       10, 13,  6,  9, 10, 11, 14,  7, 10, 11, 12, 15,  8, 11, 12, 16,  9,
       13, 14, 10, 13, 14, 15, 11, 14, 15, 16, 12, 15, 16],
       [  1,   4,   8,  12,  16,  20,  24,  28,  31,  35,  40,  45,  50,
        55,  60,  65,  69,  73,  78,  83,  88,  93,  98, 103, 107, 111,
       116, 121, 126, 131, 136, 141, 145, 149, 154, 159, 164, 169, 174,
       179, 183, 187, 192, 197, 202, 207, 212, 217, 221, 225, 230, 235,
       240, 245, 250, 255, 259, 262, 266, 270, 274, 278, 282, 286, 289], 
       [ 1,  2,  9,  1,  2,  3, 10,  2,  3,  4, 11,  3,  4,  5, 12,  4,  5,
        6, 13,  5,  6,  7, 14,  6,  7,  8, 15,  7,  8, 16,  1,  9, 10, 17,
        2,  9, 10, 11, 18,  3, 10, 11, 12, 19,  4, 11, 12, 13, 20,  5, 12,
       13, 14, 21,  6, 13, 14, 15, 22,  7, 14, 15, 16, 23,  8, 15, 16, 24,
        9, 17, 18, 25, 10, 17, 18, 19, 26, 11, 18, 19, 20, 27, 12, 19, 20,
       21, 28, 13, 20, 21, 22, 29, 14, 21, 22, 23, 30, 15, 22, 23, 24, 31,
       16, 23, 24, 32, 17, 25, 26, 33, 18, 25, 26, 27, 34, 19, 26, 27, 28,
       35, 20, 27, 28, 29, 36, 21, 28, 29, 30, 37, 22, 29, 30, 31, 38, 23,
       30, 31, 32, 39, 24, 31, 32, 40, 25, 33, 34, 41, 26, 33, 34, 35, 42,
       27, 34, 35, 36, 43, 28, 35, 36, 37, 44, 29, 36, 37, 38, 45, 30, 37,
       38, 39, 46, 31, 38, 39, 40, 47, 32, 39, 40, 48, 33, 41, 42, 49, 34,
       41, 42, 43, 50, 35, 42, 43, 44, 51, 36, 43, 44, 45, 52, 37, 44, 45,
       46, 53, 38, 45, 46, 47, 54, 39, 46, 47, 48, 55, 40, 47, 48, 56, 41,
       49, 50, 57, 42, 49, 50, 51, 58, 43, 50, 51, 52, 59, 44, 51, 52, 53,
       60, 45, 52, 53, 54, 61, 46, 53, 54, 55, 62, 47, 54, 55, 56, 63, 48,
       55, 56, 64, 49, 57, 58, 50, 57, 58, 59, 51, 58, 59, 60, 52, 59, 60,
       61, 53, 60, 61, 62, 54, 61, 62, 63, 55, 62, 63, 64, 56, 63, 64],
       [   1,    4,    8,   12,   16,   20,   24,   28,   32,   36,   40,
         44,   48,   52,   56,   60,   63,   67,   72,   77,   82,   87,
         92,   97,  102,  107,  112,  117,  122,  127,  132,  137,  141,
        145,  150,  155,  160,  165,  170,  175,  180,  185,  190,  195,
        200,  205,  210,  215,  219,  223,  228,  233,  238,  243,  248,
        253,  258,  263,  268,  273,  278,  283,  288,  293,  297,  301,
        306,  311,  316,  321,  326,  331,  336,  341,  346,  351,  356,
        361,  366,  371,  375,  379,  384,  389,  394,  399,  404,  409,
        414,  419,  424,  429,  434,  439,  444,  449,  453,  457,  462,
        467,  472,  477,  482,  487,  492,  497,  502,  507,  512,  517,
        522,  527,  531,  535,  540,  545,  550,  555,  560,  565,  570,
        575,  580,  585,  590,  595,  600,  605,  609,  613,  618,  623,
        628,  633,  638,  643,  648,  653,  658,  663,  668,  673,  678,
        683,  687,  691,  696,  701,  706,  711,  716,  721,  726,  731,
        736,  741,  746,  751,  756,  761,  765,  769,  774,  779,  784,
        789,  794,  799,  804,  809,  814,  819,  824,  829,  834,  839,
        843,  847,  852,  857,  862,  867,  872,  877,  882,  887,  892,
        897,  902,  907,  912,  917,  921,  925,  930,  935,  940,  945,
        950,  955,  960,  965,  970,  975,  980,  985,  990,  995,  999,
       1003, 1008, 1013, 1018, 1023, 1028, 1033, 1038, 1043, 1048, 1053,
       1058, 1063, 1068, 1073, 1077, 1081, 1086, 1091, 1096, 1101, 1106,
       1111, 1116, 1121, 1126, 1131, 1136, 1141, 1146, 1151, 1155, 1158,
       1162, 1166, 1170, 1174, 1178, 1182, 1186, 1190, 1194, 1198, 1202,
       1206, 1210, 1214, 1217], 1216)
    ])

    def test_CSR_genration_square_grid(self, sparse_square_4, sparse_square_8, sparse_square_16, csr_4_r, csr_4_c, csr_8_r, csr_8_c, csr_16_r, csr_16_n):

        # check all those indices are exactly the same.
        assert (sparse_square_4[0] == csr_4_r).all()
        assert (sparse_square_4[1] == csr_4_c).all()
        assert (sparse_square_8[0] == csr_8_r).all()
        assert (sparse_square_8[1] == csr_8_c).all()
        assert (sparse_square_16[0] == csr_16_r).all()
        assert sparse_square_16[2] == csr_16_n
    
    @pytest.mark.parametrize('csr_4_edges, csr_8_edges, csr_16_edges', 
    [
        ([[ 0,  0,  1,  1,  2,  2,  3,  4,  4,  5,  5,  6,  6,  7,  8,  8,
         9,  9, 10, 10, 11, 12, 13, 14],
       [ 1,  4,  2,  5,  3,  6,  7,  5,  8,  6,  9,  7, 10, 11,  9, 12,
        10, 13, 11, 14, 15, 13, 14, 15]],
        [[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  8,
         8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16,
        17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25,
        25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33,
        34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 40, 40, 41, 41, 42,
        42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 48, 48, 49, 49, 50, 50,
        51, 51, 52, 52, 53, 53, 54, 54, 55, 56, 57, 58, 59, 60, 61, 62],
       [ 1,  8,  2,  9,  3, 10,  4, 11,  5, 12,  6, 13,  7, 14, 15,  9,
        16, 10, 17, 11, 18, 12, 19, 13, 20, 14, 21, 15, 22, 23, 17, 24,
        18, 25, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30, 31, 25, 32, 26,
        33, 27, 34, 28, 35, 29, 36, 30, 37, 31, 38, 39, 33, 40, 34, 41,
        35, 42, 36, 43, 37, 44, 38, 45, 39, 46, 47, 41, 48, 42, 49, 43,
        50, 44, 51, 45, 52, 46, 53, 47, 54, 55, 49, 56, 50, 57, 51, 58,
        52, 59, 53, 60, 54, 61, 55, 62, 63, 57, 58, 59, 60, 61, 62, 63]], 
        np.array([[  0,   0,   1,   1,   2,   2,   3,   3,   4,   4,   5,   5,   6,
          6,   7,   7,   8,   8,   9,   9,  10,  10,  11,  11,  12,  12,
         13,  13,  14,  14,  15,  16,  16,  17,  17,  18,  18,  19,  19,
         20,  20,  21,  21,  22,  22,  23,  23,  24,  24,  25,  25,  26,
         26,  27,  27,  28,  28,  29,  29,  30,  30,  31,  32,  32,  33,
         33,  34,  34,  35,  35,  36,  36,  37,  37,  38,  38,  39,  39,
         40,  40,  41,  41,  42,  42,  43,  43,  44,  44,  45,  45,  46,
         46,  47,  48,  48,  49,  49,  50,  50,  51,  51,  52,  52,  53,
         53,  54,  54,  55,  55,  56,  56,  57,  57,  58,  58,  59,  59,
         60,  60,  61,  61,  62,  62,  63,  64,  64,  65,  65,  66,  66,
         67,  67,  68,  68,  69,  69,  70,  70,  71,  71,  72,  72,  73,
         73,  74,  74,  75,  75,  76,  76,  77,  77,  78,  78,  79,  80,
         80,  81,  81,  82,  82,  83,  83,  84,  84,  85,  85,  86,  86,
         87,  87,  88,  88,  89,  89,  90,  90,  91,  91,  92,  92,  93,
         93,  94,  94,  95,  96,  96,  97,  97,  98,  98,  99,  99, 100,
        100, 101, 101, 102, 102, 103, 103, 104, 104, 105, 105, 106, 106,
        107, 107, 108, 108, 109, 109, 110, 110, 111, 112, 112, 113, 113,
        114, 114, 115, 115, 116, 116, 117, 117, 118, 118, 119, 119, 120,
        120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126,
        127, 128, 128, 129, 129, 130, 130, 131, 131, 132, 132, 133, 133,
        134, 134, 135, 135, 136, 136, 137, 137, 138, 138, 139, 139, 140,
        140, 141, 141, 142, 142, 143, 144, 144, 145, 145, 146, 146, 147,
        147, 148, 148, 149, 149, 150, 150, 151, 151, 152, 152, 153, 153,
        154, 154, 155, 155, 156, 156, 157, 157, 158, 158, 159, 160, 160,
        161, 161, 162, 162, 163, 163, 164, 164, 165, 165, 166, 166, 167,
        167, 168, 168, 169, 169, 170, 170, 171, 171, 172, 172, 173, 173,
        174, 174, 175, 176, 176, 177, 177, 178, 178, 179, 179, 180, 180,
        181, 181, 182, 182, 183, 183, 184, 184, 185, 185, 186, 186, 187,
        187, 188, 188, 189, 189, 190, 190, 191, 192, 192, 193, 193, 194,
        194, 195, 195, 196, 196, 197, 197, 198, 198, 199, 199, 200, 200,
        201, 201, 202, 202, 203, 203, 204, 204, 205, 205, 206, 206, 207,
        208, 208, 209, 209, 210, 210, 211, 211, 212, 212, 213, 213, 214,
        214, 215, 215, 216, 216, 217, 217, 218, 218, 219, 219, 220, 220,
        221, 221, 222, 222, 223, 224, 224, 225, 225, 226, 226, 227, 227,
        228, 228, 229, 229, 230, 230, 231, 231, 232, 232, 233, 233, 234,
        234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 240, 241, 242,
        243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254],
       [1, 16, 2, 17, 3, 18, 4, 19, 5, 20, 6, 21, 7, 22, 8, 23, 9, 24, 
       10, 25, 11, 26, 12, 27, 13, 28, 14, 29, 15, 30, 31, 17, 32, 18, 
       33, 19, 34, 20, 35, 21, 36, 22, 37, 23, 38, 24, 39, 25, 40, 26, 
       41, 27, 42, 28, 43, 29, 44, 30, 45, 31, 46, 47, 33, 48, 34, 49, 
       35, 50, 36, 51, 37, 52, 38, 53, 39, 54, 40, 55, 41, 56, 42, 57,
       43, 58, 44, 59, 45, 60, 46, 61, 47, 62, 63, 49, 64, 50, 65, 51, 
       66, 52, 67, 53, 68, 54, 69, 55, 70, 56, 71, 57, 72, 58, 73, 59,
        74, 60, 75, 61, 76, 62, 77, 63, 78, 79, 65, 80, 66, 81, 67, 82,
         68, 83, 69, 84, 70, 85, 71, 86, 72, 87, 73, 88, 74, 89, 75, 90, 
         76, 91, 77, 92, 78, 93, 79, 94, 95, 81, 96, 82, 97, 83, 98, 84, 
         99, 85, 100, 86, 101, 87, 102, 88, 103, 89, 104, 90, 105, 91, 106, 
         92, 107, 93, 108, 94, 109, 95, 110, 111, 97, 112, 98, 113, 99, 114, 
         100, 115, 101, 116, 102, 117, 103, 118, 104, 119, 105, 120, 106, 121, 
         107, 122, 108, 123, 109, 124, 110, 125, 111, 126, 127, 113, 128, 114, 
         129, 115, 130, 116, 131, 117, 132, 118, 133, 119, 134, 120, 135, 121, 
         136, 122, 137, 123, 138, 124, 139, 125, 140, 126, 141, 127, 142, 143, 
         129, 144, 130, 145, 131, 146, 132, 147, 133, 148, 134, 149, 135, 150, 
         136, 151, 137, 152, 138, 153, 139, 154, 140, 155, 141, 156, 142, 157, 
         143, 158, 159, 145, 160, 146, 161, 147, 162, 148, 163, 149, 164, 150, 
         165, 151, 166, 152, 167, 153, 168, 154, 169, 155, 170, 156, 171, 157, 
         172, 158, 173, 159, 174, 175, 161, 176, 162, 177, 163, 178, 164, 179, 
         165, 180, 166, 181, 167, 182, 168, 183, 169, 184, 170, 185, 171, 186,
         172, 187, 173, 188, 174, 189, 175, 190, 191, 177, 192, 178, 193, 179, 
         194, 180, 195, 181, 196, 182, 197, 183, 198, 184, 199, 185, 200, 186, 
         201, 187, 202, 188, 203, 189, 204, 190, 205, 191, 206, 207, 193, 208, 
         194, 209, 195, 210, 196, 211, 197, 212, 198, 213, 199, 214, 200, 215, 
         201, 216, 202, 217, 203, 218, 204, 219, 205, 220, 206, 221, 207, 222, 
         223, 209, 224, 210, 225, 211, 226, 212, 227, 213, 228, 214, 229, 215, 
         230, 216, 231, 217, 232, 218, 233, 219, 234, 220, 235, 221, 236, 222,
          237, 223, 238, 239, 225, 240, 226, 241, 227, 242, 228, 243, 229, 244, 
          230, 245, 231, 246, 232, 247, 233, 248, 234, 249, 235, 250, 236, 251, 
          237, 252, 238, 253, 239, 254, 255, 241, 242, 243, 244, 245, 246, 247, 
          248, 249, 250, 251, 252, 253, 254, 255]]))
    ])

    def test_csr_to_edges(self, sparse_square_4, sparse_square_8, sparse_square_16, csr_4_edges, csr_8_edges, csr_16_edges):
        
        # generate edge list from CSR
        edge_4 = csr_to_edges(sparse_square_4[0], sparse_square_4[1]).T
        edge_8 = csr_to_edges(sparse_square_8[0], sparse_square_8[1]).T
        edge_16 = csr_to_edges(sparse_square_16[0], sparse_square_16[1]).T

        assert((edge_4 == np.array(csr_4_edges)).all())
        assert((edge_8 == np.array(csr_8_edges)).all())
        assert((edge_16 == np.array(csr_16_edges)).all())


#################################################### test for sfc_cae.py/ utils.py ######################################################
dict_path = 'decompressing_examples/'

@fixture(scope='module')
def autoenocoder_FPC_CG():
    """
    Intialize SFC_CAE for the advection FPC-CG case
    """
    # first download dataset for the following advanced tests
    if sys.platform == 'linux' or sys.platform == 'linux2': os.system('bash get_FPC_data_CG.sh')
    
    # parameters for intialising
    input_size = 3571
    dimension = 2
    components = 2
    self_concat = 2
    structured = False
    nearest_neighbouring = False
    dims_latent = 16
    activation = nn.Tanh()
    variational = False
    space_filling_orderings = torch.load(dict_path + 'fpc_cg_sfc_2.pt')
    invert_space_filling_orderings = torch.load(dict_path + 'fpc_cg_invsfc_2.pt')

    autoencoder = SFC_CAE(input_size,
                          dimension,
                          components,
                          structured,
                          self_concat,
                          nearest_neighbouring,
                          dims_latent,
                          space_filling_orderings, 
                          invert_space_filling_orderings,
                          activation,
                          variational = variational)

    return autoencoder

@fixture(scope='module')
def autoenocoder_FPC_CG_VCAE():
    """
    Intialize SFC_VCAE for the advection FPC-CG case
    """
    input_size = 3571
    dimension = 2
    components = 2
    self_concat = 2
    structured = False
    nearest_neighbouring = False
    dims_latent = 16
    activation = nn.Tanh()
    variational = True
    space_filling_orderings = torch.load(dict_path + 'fpc_cg_sfc_2.pt')
    invert_space_filling_orderings = torch.load(dict_path + 'fpc_cg_invsfc_2.pt')

    autoencoder = SFC_CAE(input_size,
                          dimension,
                          components,
                          structured,
                          self_concat,
                          nearest_neighbouring,
                          dims_latent,
                          space_filling_orderings, 
                          invert_space_filling_orderings,
                          activation,
                          variational = variational)

    return autoencoder

@fixture(scope='module')
def model_dict_CAE_FPC_CG():
    """
    Model dict of a trained 2-SFC-CAE on FPC-CG
    """
    SFC_CAE_pathname = 'Variational_False_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000_dict.pth' 
    # local_path = 'C:\\Users\ROG\Desktop\FPC_CG_tsne/'
    # local_path += SFC_CAE_pathname

    model_dict = torch.load(SFC_CAE_pathname, map_location=torch.device('cpu'))['model_state_dict']

    return model_dict

@fixture(scope='module')
def model_dict_VCAE_FPC_CG():
    """
    Model dict of a trained 2-SFC-VCAE on FPC-CG
    """
    SFC_VCAE_pathname = 'Variational_True_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000_dict.pth'
    # local_path = 'C:\\Users\ROG\Desktop\FPC_CG_tsne/'
    # local_path += SFC_VCAE_pathname

    model_dict = torch.load(SFC_VCAE_pathname, map_location=torch.device('cpu'))['model_state_dict']

    return model_dict

@fixture(scope='module')
def FPC_CG_tensor():
    """
    Load FPC_CG data, 2000 snapshots.
    """
    # data_path = 'C:\\Users\ROG\Desktop\Results\FPC_new/FPC_Re3900_CG_new/'
    data_path = 'FPC_Re3900_CG_new/'
    vtu_fields = ['Velocity']
    full_tensor, coords, cells = read_in_files(data_path, vtu_fields = vtu_fields)

    return full_tensor


class Test_SFC_CAE_initialization_and_evaluation(object):
    '''
    Test for functions in sfc_cae.py / utils.py
    '''
    @pytest.mark.parametrize('FPC_CG_channels, FPC_CG_nH, FPC_CG_FC', 
    [(
        [4, 8, 16, 16], [3571, 893, 224, 57], [1824, 456, 114, 16]
    )])

    def test_network_layers_FPC_CG(self, autoenocoder_FPC_CG, FPC_CG_channels, FPC_CG_nH, FPC_CG_FC, FPC_CG_tensor):
        ''' 
        test the autoencoder has the correct structure.
        '''
        assert autoenocoder_FPC_CG.encoder.channels == FPC_CG_channels
        assert autoenocoder_FPC_CG.encoder.conv_size == FPC_CG_nH
        assert autoenocoder_FPC_CG.encoder.size_fc == FPC_CG_FC

    def test_load_state_dict_SFC_CAE(self, autoenocoder_FPC_CG, model_dict_CAE_FPC_CG):
        '''
        Test the ability to correctly load model_dict for SFC_CAE.
        '''
        try:
          autoenocoder_FPC_CG.load_state_dict(model_dict_CAE_FPC_CG)
        except AssertionError as error_msg:
          print(error_msg)

    def test_load_state_dict_SFC_VCAE(self, autoenocoder_FPC_CG_VCAE, model_dict_VCAE_FPC_CG):
        '''
        Test the ability to correctly load model_dict for SFC_VCAE.
        '''
        try:
          autoenocoder_FPC_CG_VCAE.load_state_dict(model_dict_VCAE_FPC_CG)
        except AssertionError as error_msg:
          print(error_msg)

    def test_load_data_from_vtu(self, FPC_CG_tensor):
        '''
        Test the FPC_CG data has been corrected loaded from vtu files.
        '''
        assert FPC_CG_tensor.shape == (2000, 3571, 2)

    def test_standardlization(self, FPC_CG_tensor):
        ''' 
        test standardlization function, the tensor has been scaled to [-1, 1] for all components.
        '''
        full_set, tk, tb = standardlize_tensor(FPC_CG_tensor, lower = -1, upper = 1)

        for i in range(full_set.shape[-1]):
           assert (full_set[..., i].max().detach().numpy() == 1)
           assert (full_set[..., i].min().detach().numpy() == -1)

    def test_accuracy_SFC_CAE(self, autoenocoder_FPC_CG, FPC_CG_tensor):
        '''
        test the accuracy of the loaded SFC_CAE, as well as the latent has the correct shape
        '''

        # test latent variables have correct shape.
        latent_tensor = autoenocoder_FPC_CG.encoder(FPC_CG_tensor)
        assert latent_tensor.shape == (2000, 16)
          

        # test model accuracy.
        full_reconstructed = autoenocoder_FPC_CG.decoder(latent_tensor)
        assert np.allclose(nn.MSELoss()(FPC_CG_tensor, full_reconstructed).item(), 7e-5, atol= 1e-5)

    def test_accuracy_SFC_VCAE(self, autoenocoder_FPC_CG_VCAE, FPC_CG_tensor):
        '''
        test the accuracy of the loaded SFC_VCAE, as well as the latent has the correct shape
        '''

        # test latent variables have correct shape.
        latent_tensor, KL = autoenocoder_FPC_CG_VCAE.encoder(FPC_CG_tensor)
        assert latent_tensor.shape == (2000, 16)
          

        # test model accuracy and KL.
        full_reconstructed = autoenocoder_FPC_CG_VCAE.decoder(latent_tensor)
        assert np.allclose(nn.MSELoss()(FPC_CG_tensor, full_reconstructed).item(), 0.00045, atol = 5e-5)
        assert np.allclose(KL.item(), 0.0016152136959, atol = 1e-5)








# autoenocoder_FPC_CG_VCAE




