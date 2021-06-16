#!/usr/bin/env bash

# get librarys
wget https://raw.githubusercontent.com/FluidityProject/fluidity/main/python/vtktools.py
wget https://www.dropbox.com/s/w8me9a8e2t6iwij/space_filling_decomp_new.f90
python3 -m numpy.f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new
DATADIR='FPCdata' #location where data gets downloaded to

# get data
mkdir -p $DATADIR && cd $DATADIR
wget https://www.dropbox.com/s/ibpwa5e8xxzyla9/FPC_Re3900_2D_CG_new.zip
unzip FPC_Re3900_2D_CG_new.zip -d './' && rm -rf FPC_Re3900_2D_CG_new.zip
echo "downloaded the Flow Past Cylinder data and putting it in: " $DATADIR