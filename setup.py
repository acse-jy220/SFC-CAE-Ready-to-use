import os
from setuptools import setup, find_packages
from distutils.core import Command

with open('requirements.txt') as f:
    required = f.read().splitlines()

reqs = []
for ir in required:
    if ir[0:3] == 'git':
        name = ir.split('/')[-1]
        reqs += ['%s @ %s@master' % (name, ir)]
    else:
        reqs += [ir]

os.system('bash get_scripts.sh')

setup(name='SFC-CAE',
      version="Version: 0.01",
      description="A Space-filling curve autoencoder for compressing data on unstructured mesh.",
      long_description="""need filling""",
      url='wait for the url',
      author="Imperial College London",
      author_email='jin.yu20@imperial.ac.uk',
      license='MIT',
      install_requires=reqs,
      test_suite='tests')