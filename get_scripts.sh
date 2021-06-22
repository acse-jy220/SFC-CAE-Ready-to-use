#!/usr/bin/env bash

# get librarys
wget https://raw.githubusercontent.com/FluidityProject/fluidity/main/python/vtktools.py
wget https://www.dropbox.com/s/w8me9a8e2t6iwij/space_filling_decomp_new.f90
python3 -m numpy.f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new