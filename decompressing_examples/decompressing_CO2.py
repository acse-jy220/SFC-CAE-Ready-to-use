from sfc_cae import *

path = torch.load('CO2_Variational_False_Changelr_False_Latent_4_Nearest_neighbouring_True_SFC_nums_3_startlr_0.001_n_epoches_2000_dict.pth', map_location = torch.device('cpu'))

model_dict = path['model_state_dict']

space_filling_orderings = torch.load('CO2_sfc.pt')
invert_space_filling_orderings = torch.load('CO2_invsfc.pt')

input_size = space_filling_orderings.shape[1]
dimension = 3
components = 4
structured = False
self_concat = 2
nearest_neighbouring = True
dims_latent = 4
activation = nn.Tanh()
variational = False

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


autoencoder.load_state_dict(model_dict)

latent_path = 'CO2_latent_4/'

latent_variables = read_in_compressed_tensors(latent_path)

reconstructed_path = 'reconstrcuted_CO2_letent_4'
vtu_fields = ['CO2_ppm', 'Velocity']
field_spliter = [0, 1, 4]
tamplate_vtu = meshio.read('tamplete_CO2.vtu')

tk = torch.load('CO2_tk.pt')
tb = torch.load('CO2_tb.pt')

decompress_to_vtu(latent_variables, tamplate_vtu, reconstructed_path, vtu_fields, field_spliter, autoencoder, tk, tb, start_index = None, end_index = None, model_device = torch.device('cpu'))