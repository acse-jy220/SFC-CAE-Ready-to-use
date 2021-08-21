mkdir FPC_Re3900_CG_new
wget https://www.dropbox.com/sh/aid0yv2685nln51/AADmtVChECW_B85M8O2cDmR0a
unzip AADmtVChECW_B85M8O2cDmR0a -d './FPC_Re3900_CG_new/'
rm -rf AADmtVChECW_B85M8O2cDmR0a

# get two pretrained model dict, one is 2-SFC-CAE, another is 2-SFC-VCAE, both compressed down to 16 latent variables.
wget https://www.dropbox.com/s/x4earqswako7bf5/Variational_False_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000_dict.pth
wget https://www.dropbox.com/s/m8jceafgej6xzo7/Variational_True_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000_dict.pth

# get (inv) sfcs for the above two autoencoders.
wget https://www.dropbox.com/s/dywgqdrizitc9w8/fpc_cg_sfc_2.pt
wget https://www.dropbox.com/s/a1batae5f3lb7dy/fpc_cg_invsfc_2.pt