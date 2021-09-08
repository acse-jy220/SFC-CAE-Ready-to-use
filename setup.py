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
   compile_command = 'f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new --compiler=mingw32'
elif sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
   # on linux
   compile_command = 'python3 -m numpy.f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new'   

setup(name='SFC-CAE',
      description="A self-adjusting Space-filling curve (variational) convolutional autoencoder for compressing data on unstructured mesh.",
      url='https://github.com/acse-jy220/SFC-CAE-Ready-to-use',
      author="Imperial College London",
      author_email='jin.yu20@imperial.ac.uk',
      install_requires=reqs,
      test_suite='tests',
      version='0.1.0',
      packages=['sfc_cae'])

# compile fortran
os.system(compile_command)