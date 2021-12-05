mkdir FPC_Re3900_CG_new
wget https://www.dropbox.com/sh/aid0yv2685nln51/AADmtVChECW_B85M8O2cDmR0a
unzip AADmtVChECW_B85M8O2cDmR0a -d './FPC_Re3900_CG_new/'
rm -rf AADmtVChECW_B85M8O2cDmR0a

# get two pretrained model dict, one is 2-SFC-CAE, another is 2-SFC-VCAE, both compressed down to 16 latent variables.
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1SmMWemkN2ykR3Hwa_IoU2ka49P69_G5j' -O 'Variational_False_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000_dict.pth'
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Ki3i5wLsdVgx3YupQlLQIZf1A34y5RgN' -O 'Variational_True_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000_dict.pth'