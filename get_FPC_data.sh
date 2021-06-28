DATADIR='./' #location where data gets downloaded to
# get data
mkdir -p $DATADIR && cd $DATADIR
# wget https://www.dropbox.com/s/ibpwa5e8xxzyla9/FPC_Re3900_2D_CG_new.zip
# unzip FPC_Re3900_2D_CG_new.zip -d './'
# rm -rf FPC_Re3900_2D_CG_new.zip
wget https://www.dropbox.com/s/yzh99n64mxpj9db/FPC_Re3900_DG_old.zip
unzip FPC_Re3900_DG_old.zip -d './'
rm -rf FPC_Re3900_DG_old.zip
echo "downloaded the Flow Past Cylinder data and putting it in: " $DATADIR