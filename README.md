# SFC-CAE-Ready-to-use

## A self-adjusting Space-filling curve autoencoder
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/LICENSE)
[![Testing workflow](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/actions/workflows/test.yml/badge.svg)](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/.github/workflows/test.yml)


<details open="open">
  <summary>**Table of Contents**</summary>
  <ol>
    <li>
      <a href="#Project\ Description">Project Description</a>
    </li>
    <li>
      <a href="#Getting\ started">Getting Started</a>
      <ul>
        <li><a href="#Prerequisites\ &\ Dependencies">Prerequisites & Dependencies</a></li>
        <li><a href="#Installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#Examples">Colab\ Notebook Examples</a>
      <ul>
        <li><a href="#Advection\ of\ a\ block">Advection of a block  (128 * 128 Structured Grid)</a></li>
        <li><a href="#Flow Past Cylinder-DG">Flow Past Cylinder - DG Mesh (2000 snapshots, 20550 Nodes, 2/3 components)</a></li>
        <li><a href="#Flow Past Cylinder-CG Mesh">Flow Past Cylinder - CG Mesh (2000 snapshots, 3571 Nodes, 2 components) </a></li>
        <li><a href="#CO2"> CO2 in the room - CG Mesh (455 snapshots, 148906 Nodes, 4 components)</a></li>
        <li><a href="#Slugflow"> Slugflow - DG mesh (1706 snapshots, 3571 Nodes, 4 components)</a></li>
      </ul>   
    </li>
    <li><a href="#License">License</a></li>
    <li><a href="#Testing">Testing</a></li>
    <li><a href="#Contact">Contact</a></li>
    <li><a href="#Acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

## Project Description

This project contains a self-adjusting Space-filling curve Convolutional Autoencoder (SFC-CAE), of which the methodlogy is based on the work of previous year https://arxiv.org/abs/2011.14820, this new tool automatically generates a SFC-CAE network for unadapted mesh examples, a simple variational autoencoder is also included.

## Getting started
### Prerequisites & Dependencies

* Python 3.8.5
* numpy >= 1.19.5
* scipy >= 1.4.1
* matplotlib ~= 3.2.2
* vtk >= 9.0
* livelossplot ~= 0.5.4
* meshio[all]
* cmocean ~= 2.0
* torch >= 1.8.0
* dash ~= 1.21.0
* pytest >= 3.6.4
* progressbar2 ~= 3.38.0
* (Optional) GPU/multi GPUs with CUDA

### Installation
1. Clone the repository:
```sh
$ git clone https://github.com/acse-jy220/SFC-CAE-Ready-to-use
```
2. cd to the repo:
```sh 
$ cd SFC-CAE-Ready-to-use
```
3. Install the module:
(1) For `pip` install, just use 
```sh
$ pip install -e .
```
It will compile the fortran library automatically, no matter you are on Windows or Linux.

(2) If you like to create a `conda` environment, just enter
```sh
$ conda env create -f environment.yml
```
but you need to compile the fortran code by yourself in this way. 
<br>
On linux, type
```sh
$ python3 -m numpy.f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new
```
<br>
On windows, install 

[MinGW](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.2.0/threads-posix/seh/)
(I use version 7.2.0) and compile fortran use

```sh
$ f2py -c space_filling_decomp_new.f90 -m space_filling_decomp_new --compiler=mingw32
```

4. For convenience, you could just simply import all functions in my module:
```python
from sfc_cae import *
```
and call the functions you want! Please have a look at the instruction notebooks.

## Examples
### Advection of a block
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-jy220/SFC-CAE-Ready-to-use/blob/main/Colab\ Notebooks/Instruction_SFC_CAE.ipynb)

### Flow Past Cylinder-DG
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-jy220/SFC-CAE-Ready-to-use/blob/main/Colab\ Notebooks/Instruction_SFC_CAE.ipynb)

### Flow Past Cylinder-CG
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-jy220/SFC-CAE-Ready-to-use/blob/main/Colab\ Notebooks/Instruction_SFC_CAE.ipynb)

### CO2
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-jy220/SFC-CAE-Ready-to-use/blob/main/Colab\ Notebooks/Instruction_SFC_CAE.ipynb)

### Slugflow
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-jy220/SFC-CAE-Ready-to-use/blob/main/Colab\ Notebooks/Instruction_SFC_CAE.ipynb)

## License

Distributed under the [Apache 2.0](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/LICENSE) License.

## Testing 
Some basic tests for the module are avaliable in [tests/](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/tests/) folder, a [github workflow](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/.github/workflows/test.yml) is built to run those tests.

## Contact
* Jin Yu jin.yu20@imperial.ac.uk or yu19832059@gmail.com

## Acknowledgements
Great thanks to my supervisors:
* Dr. Claire Heaney
* Prof. Christopher Pain 