# get the FPC_CG dataset, in '.vtu' format
gdown https://drive.google.com/uc?id=1BpItXH0Rvwf2NvTBLTIZwr7-LmSyLd9M
unzip FPC_Re3900_CG_new.zip
# get two pretrained model dict, one is 2-SFC-CAE, another is 2-SFC-VCAE, both compressed down to 16 latent variables.
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1SmMWemkN2ykR3Hwa_IoU2ka49P69_G5j' -O 'Variational_False_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000_dict.pth'
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Ki3i5wLsdVgx3YupQlLQIZf1A34y5RgN' -O 'Variational_True_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000_dict.pth'
