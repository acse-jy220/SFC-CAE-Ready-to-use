import sys
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
elif sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
   # on unix
   fortran_compiler_type = None
sfc_lib = Extension('space_filling_decomp_new', sources=['space_filling_decomp_new.f90'])
interpolate_lib = Extension('sfc_interpolate', sources=['x_conv_fixed_length.f90'])

# build_ext subclass, define custom compiler
class build_ext_subclass(build_ext):
    def build_extensions(self):
        print(self.compiler.compiler_type)
        self.compiler.compiler_type = fortran_compiler_type
        build_ext.build_extensions(self)

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