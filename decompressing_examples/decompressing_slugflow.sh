# get sfcs/inv_sfcs as well as a templete vtu file.
wget https://www.dropbox.com/s/zxqkvm8b7a0tz45/tamplete_slugflow.vtu
wget https://www.dropbox.com/s/52s2efjw9dnxaja/sfcs.pt
wget https://www.dropbox.com/s/828zfj7nwg6os7c/inv_sfcs.pt
# get model_dict (903 MB)
wget https://www.dropbox.com/s/3xwcwvo43nadbgo/Slugflow_Variational_False_Changelr_False_Latent_64_Nearest_neighbouring_True_SFC_nums_3_startlr_0.001_n_epoches_1500_dict.pth
# reconstructing
python3 decompressing_slugflow.py