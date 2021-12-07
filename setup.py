import os
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

reqs = []
for ir in required:
    if ir[0:3] == 'git':
        name = ir.split('/')[-1]
        reqs += ['%s @ %s@master' % (name, ir)]
    else:
        reqs += [ir]

if sys.platform == 'win32' or sys.platform == 'cygwin' or sys.platform == 'msys':
   # on windows
   compile_commands = ['f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new --compiler=mingw32']
   compile_commands.append('f2py -c x_conv_fixed_length.f90 -m sfc_interpolate --compiler=mingw32')
elif sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
   # on linux
   compile_commands = ['python3 -m numpy.f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new']
   compile_commands.append('python3 -m numpy.f2py -c x_conv_fixed_length.f90 -m sfc_interpolate')  

setup(name='SFC-CAE',
      description="A self-adjusting Space-filling curve (variational) convolutional autoencoder for compressing data on unstructured mesh.",
      url='https://github.com/acse-jy220/SFC-CAE-Ready-to-use',
      author="Imperial College London",
      author_email='jin.yu20@imperial.ac.uk',
      install_requires=reqs,
      test_suite='tests',
      version='0.2.8',
      packages=['sfc_cae'])

# compile fortran
for compile_command in compile_commands: os.system(compile_command)