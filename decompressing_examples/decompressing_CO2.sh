# get sfcs/inv_sfcs as well as a templete vtu file.
wget https://www.dropbox.com/s/oww3ph1z8b85pms/tamplete_CO2.vtu
wget https://www.dropbox.com/s/32qieivxyon7554/CO2_sfc.pt
wget https://www.dropbox.com/s/2z2rqak5vu65eib/CO2_invsfc.pt
# get model_dict (108 MB)
wget https://www.dropbox.com/s/8290f2amf7bgbu5/CO2_Variational_False_Changelr_False_Latent_4_Nearest_neighbouring_True_SFC_nums_3_startlr_0.001_n_epoches_2000_dict.pth
# reconstructing
python3 decompressing_CO2.py