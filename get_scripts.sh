#!/usr/bin/env bash

# get vtktools and compile fortran library
wget https://raw.githubusercontent.com/FluidityProject/fluidity/main/python/vtktools.py
# on linux, mac and windows
f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new --compiler=mingw32