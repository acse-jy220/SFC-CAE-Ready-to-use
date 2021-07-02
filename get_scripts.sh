#!/usr/bin/env bash

# get vtktools and compile fortran library
wget https://raw.githubusercontent.com/FluidityProject/fluidity/main/python/vtktools.py
python3 -m numpy.f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new