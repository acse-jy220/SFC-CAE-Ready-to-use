## run 'bash get_FPC_data_CG' first to download the FPC_CG data as well as the sfc curves.

from sfc_cae import *
from sklearn import manifold, datasets, cluster # install it if not installed

# load tensor
vtu_fields = ['Velocity']
data_path = 'FPC_Re3900_CG_new/'
full_tensor, coords, cells = read_in_files(data_path, vtu_fields = vtu_fields)
full_set, tk, tb = standardlize_tensor(full_tensor, lower = -1, upper = 1)


# initialise t-SNE
tsne = manifold.TSNE(n_components=2, init='pca', random_state=3, perplexity=10, n_iter_without_progress= 1000)

# Generation of Gaussian noise 
# Gaussian = torch.distributions.Normal(0, 1) 
# Gaussian_noise = Gaussian.sample((1, latent_tensor.shape[-1]))
# Gaussian_noise = torch.cat([Gaussian_noise] * latent_tensor.shape[0], 0)

# define Gaussian noise, we use a fixed example here
Gaussian_noise = torch.tensor([-0.6622,  2.2681,  1.3811, -0.2305, -0.5013,  0.5346,  0.1032,  1.7119,
         1.3086, -0.6927, -1.2516, -0.4273,  0.1774, -1.0925,  1.5616,  0.3671]).unsqueeze(0)
Gaussian_noise = torch.cat([Gaussian_noise] * full_set.shape[0], 0)


################################################  t-SNE for SFC-CAE  ################################################################
pathname = 'Variational_False_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000'

model_dict = pathname + '_dict.pth'
model_dict = torch.load(model_dict, map_location = torch.device('cpu'))['model_state_dict']

input_size = 3571
dimension = 2
components = 2
self_concat = 2
structured = False
nearest_neighbouring = False
dims_latent = 16
activation = nn.Tanh()
variational = False
space_filling_orderings = torch.load('fpc_cg_sfc_2.pt')
invert_space_filling_orderings = torch.load('fpc_cg_invsfc_2.pt')


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


autoencoder.load_state_dict(model_dict)

latent_tensor = autoencoder.encoder(full_set)

fake_latent = latent_tensor  + Gaussian_noise

# snapshot 134
X_tsne = tsne.fit_transform(np.vstack((latent_tensor.detach().numpy(), fake_latent[134].detach().numpy())))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # standardisation
plt.figure(figsize=(8, 8))
plt.scatter(X_norm[:, 0], X_norm[:, 1], c = 'xkcd:sky blue', label = 'Real Latent Variables')
plt.scatter(X_norm[-1, 0], X_norm[-1, 1], c = 'red', label = 'Fabricated Latent Variable')
leg = plt.legend(loc = 'upper right')
plt.savefig('t-SNE-AE-CG-16-latent.png', dpi = 200)
print('t-SNE plot for SFC-CAE saved.')

################################################  t-SNE for SFC-VCAE  ################################################################
variational = True

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

pathname = 'Variational_True_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000'

model_dict = pathname + '_dict.pth'
model_dict = torch.load(model_dict, map_location = torch.device('cpu'))['model_state_dict']

autoencoder.load_state_dict(model_dict)

latent_tensor, KL = autoencoder.encoder(full_set)

fake_latent = latent_tensor  + Gaussian_noise

# snapshot 134
X_tsne = tsne.fit_transform(np.vstack((latent_tensor.detach().numpy(), fake_latent[134].detach().numpy())))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # standardisation
plt.figure(figsize=(8, 8))
plt.scatter(X_norm[:, 0], X_norm[:, 1], c = 'xkcd:sky blue', label = 'Real Latent Variables')
plt.scatter(X_norm[-1, 0], X_norm[-1, 1], c = 'red', label = 'Fabricated Latent Variable')
leg = plt.legend(loc = 'upper right')
plt.savefig('t-SNE-VAE-CG-16-latent.png', dpi = 200)
print('t-SNE plot for SFC-VCAE saved.')
