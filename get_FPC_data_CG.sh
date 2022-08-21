# get the FPC_CG dataset, in Zip format
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BpItXH0Rvwf2NvTBLTIZwr7-LmSyLd9M' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*
/\1\n/p')&id=1BpItXH0Rvwf2NvTBLTIZwr7-LmSyLd9M" -O FPC_Re3900_CG_new.zip && rm -rf /tmp/cookies.txt
unzip FPC_Re3900_CG_new.zip
# get two pretrained model dict, one is 2-SFC-CAE, another is 2-SFC-VCAE, both compressed down to 16 latent variables.
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1SmMWemkN2ykR3Hwa_IoU2ka49P69_G5j' -O 'Variational_False_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000_dict.pth'
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Ki3i5wLsdVgx3YupQlLQIZf1A34y5RgN' -O 'Variational_True_Changelr_False_Latent_16_Nearest_neighbouring_False_SFC_nums_2_startlr_0.0001_n_epoches_2000_dict.pth'
