from distutils.command.build import build
import sys
from distutils.ccompiler import new_compiler
from numpy.distutils.core import setup, Extension
from numpy.distutils.command.build_ext import build_ext

with open('requirements.txt') as f:
    required = f.read().splitlines()

reqs = []
for ir in required:
    if ir[0:3] == 'git':
        name = ir.split('/')[-1]
        reqs += ['%s @ %s@master' % (name, ir)]
    else:
        reqs += [ir]

# define fortran extension
if sys.platform == 'win32' or sys.platform == 'cygwin' or sys.platform == 'msys':
   # on windows
   fortran_compiler_type = 'mingw32'
elif sys.platform == 'darwin':
   # on macOS
   fortran_compiler_type = 'bcpp'
elif sys.platform == 'linux' or sys.platform == 'linux2':
   # on unix
   fortran_compiler_type = None
sfc_lib = Extension('space_filling_decomp_new', sources=['space_filling_decomp_new.f90'])
interpolate_lib = Extension('sfc_interpolate', sources=['x_conv_fixed_length.f90'])

# build_ext subclass, define custom compiler
class build_ext_subclass(build_ext):
    def initialize_options(self):
        build_ext.initialize_options(self)
        self.compiler = fortran_compiler_type

setup(name='SFC-CAE',
      description="A self-adjusting Space-filling curve (variational) convolutional autoencoder for compressing data on unstructured mesh.",
      url='https://github.com/acse-jy220/SFC-CAE-Ready-to-use',
      author="Imperial College London",
      author_email='jin.yu20@imperial.ac.uk',
      cmdclass={'build_ext': build_ext_subclass},
      install_requires=reqs,
      ext_modules=[sfc_lib, interpolate_lib],
      test_suite='tests',
      version='v1.1',
      packages=['sfc_cae'])