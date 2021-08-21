wget https://www.dropbox.com/s/yzh99n64mxpj9db/FPC_Re3900_DG_old.zip
wget https://www.dropbox.com/s/b0kciap7pgq9cp5/FPC_Re3900_DG_new.zip
unzip FPC_Re3900_DG_old.zip -d './'
rm -rf FPC_Re3900_DG_old.zip
unzip FPC_Re3900_DG_new.zip -d './'
rm -rf FPC_Re3900_DG_new.zip
rm -rf './FPC_Re3900_DG_new/copy_over_N_files.py'