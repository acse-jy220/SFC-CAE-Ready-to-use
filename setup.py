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
else:
   # on linux and mac
   compile_command = 'python3 -m numpy.f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new'

setup(name='SFC-CAE',
      description="A self-adjusting Space-filling curve (variational) convolutional autoencoder for compressing data on unstructured mesh.",
      url='https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jy220',
      author="Imperial College London",
      author_email='jin.yu20@imperial.ac.uk',
      install_requires=reqs,
      test_suite='tests',
      packages=['sfc_cae'])

os.system(compile_command)