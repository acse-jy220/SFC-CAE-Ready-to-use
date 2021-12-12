# SFC-CAE-Ready-to-use

## A self-adjusting Space-filling curve autoencoder
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/LICENSE)
[![Testing workflow](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/actions/workflows/test.yml/badge.svg)](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/.github/workflows/test.yml)

<br />
<p align="center">
  <a href="https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/pics/structure_SFC_CAE.svg">
    <img src="pics/structure_SFC_CAE.svg">
    <figcaption> Achitechture of a Space-filling curve Convolutional Autoencoder </figcaption>
  </a>
</p>

<p align="center">
  <a href="https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/pics/structure_SFC_VCAE.svg">
    <img src="pics/structure_SFC_VCAE.svg">
    <figcaption> Achitechture of a Space-filling curve Variational Convolutional Autoencoder </figcaption>
  </a>
</p>


<details open="open">
  <summary>__Table of Contents__</summary>
  <ol>
    <li>
      <a href="#project-description">Project Description</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#Dependencies">Prerequisites & Dependencies</a></li>
         <li><a href="#Contribution-of-Codes">Clearance of contribution of codes</a></li>
        <li><a href="#Installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#Template-Notebooks">Colab Notebooks</a>
      <ul>
        <li><a href="#advecting-block">Advection of a Block/Gaussian  (128 * 128 Structured Grid)</a></li>
        <li><a href="#FPC-DG">Flow Past Cylinder - DG Mesh (2000 snapshots, 20550 Nodes, 2/3 components)</a></li>
        <li><a href="#FPC-CG">Flow Past Cylinder - CG Mesh (2000 snapshots, 3571 Nodes, 2 components) </a></li>
        <li><a href="#CO2"> CO2 in the room - CG Mesh (455 snapshots, 148906 Nodes, 4 components)</a></li>
        <li><a href="#Slugflow"> Slugflow - DG mesh (1706 snapshots, 1342756 Nodes, 4 components)</a></li>
      </ul>   
    </li>
    <li><a href="#tSNE-Plots">t-SNE plots</a></li>
    <li><a href="#Decompressing-Examples">Decompressing Examples</a></li>
    <li><a href="#Training-on-HPC">Training on HPC</a></li>
    <li><a href="#License">License</a></li>
    <li><a href="#Testing">Testing</a></li>
    <li><a href="#Contact">Contact</a></li>
    <li><a href="#Acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

## Project Description

This project contains a self-adjusting Space-filling curve Convolutional Autoencoder (SFC-CAE), of which the methodlogy is based on the work of previous year 
[![DOI:2011.14820](https://img.shields.io/badge/math.CO-arXiv%2011.14820-B31B1B.svg)](https://arxiv.org/abs/2011.14820), this new tool automatically generates a SFC-CAE network for unadapted mesh examples, a simple variational autoencoder is also included.

## Getting started
### Dependencies

* Python ~= 3.8.5
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

### Contribution of Codes
External Libraries:
* **space_filling_decomp_new.f90**
<br>

A domian decompositon method for unstructured mesh, developed by Prof. Christopher Pain, for detail please see [Paper](https://doi.org/10.1002/(SICI)1097-0207(19990220)44:5<593::AID-NME516>3.0.CO;2-0).

* **vtktools.py**
<br>

The Python wrappers for vtu file I/O, from [FluidityProject](https://github.com/FluidityProject/fluidity/blob/main/python/vtktools.py)

Other codes in this repository are implemented by myself.
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
<br>

(2) Create a `conda` environment via
```sh
$ conda env create -f environment.yml
```
but you need to compile the fortran code by yourself in this way. 
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
and call the functions you want! 

5. Initializing the autoencoder by passing the following arguments:
```python
autoencoder = SFC_CAE(input_size,
                      dimension,
                      components,
                      structured,
                      self_concat,
                      nearest_neighbouring,
                      dims_latent,
                      space_filling_orderings, 
                      invert_space_filling_orderings,
                      activation,
                      variational = variational)
```
The meaning of each parameters are:
* input\_size: [int] the number of Nodes in each snapshot.
* dimension: [int] the dimension of the problem, 2 for 2D and 3 for 3D.
* components: [int] the number of components we are compressing.
* structured: [bool] whether the mesh is structured or not.
* self\_concat: [int] a channel copying operation, of which the input\_channel of the 1D Conv Layers would be components * self\_concat.
* nearest\_neighbouring: [bool] whether the sparse layers are added to the ANN or not.
* dims\_latent: [int] the dimension of the latent variable
* space\_filling\_orderings: [list of 1D-arrays or 2D-array] the space-filling curves, of shape [number of curves, number of Nodes]
* activation: [torch.nn.functional] the activation function, ReLU() and Tanh() are usually used.
* variational: [bool] whether this is a variational autoencoder or not.

For advance training options, please have a look at the [instruction notebooks](#Template-Notebooks)

## Template Notebooks
### Advecting Block
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-jy220/SFC-CAE-Ready-to-use/blob/main/Colab_Notebooks/Instruction_SFC_CAE_Advecting.ipynb)

<p align="center">
  <p float="middle">
     <p float="middle">
     <img src="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/original_block.gif" width="500">
     <p float="middle">
     <a href="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/original_block.gif"><strong>Analytical Block Advection</strong></a>
     <p float="middle">
     <img src="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/reconstructed_block.gif" width="500">
     <p float="middle">
     <a href="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/reconstructed_block.gif"><strong>Reconstructed by 2-SFC-CAE-NN, 16 latent</strong></a>
  </p>
</p>

### FPC-DG
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-jy220/SFC-CAE-Ready-to-use/blob/main/Colab_Notebooks/Instruction_SFC_CAE_FPC_DG.ipynb)

<p align="center">
  <p float="middle">
     <img src="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/original_FPC.gif">
     <a href="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/original_FPC.gif"><strong>Original Velocity Magnitude</strong></a>
     <img src="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/reconstructed_FPC.gif">
     <a href="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/reconstructed_FPC.gif"><strong>Reconstructed by 2-SFC-CAE-NN, 8 latent</strong></a>
  </p>
</p>

### FPC-CG
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-jy220/SFC-CAE-Ready-to-use/blob/main/Colab_Notebooks/Instruction_SFC_CAE_FPC_CG.ipynb)

### CO2
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-jy220/SFC-CAE-Ready-to-use/blob/main/Colab_Notebooks/Instruction_SFC_CAE_CO2.ipynb)

<p align="center">
  <p float="left">
     <img src="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/original_ppm.gif" width="1000">
     <a href="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/original_ppm.gif"><strong>Original CO2 PPM</strong></a>
     <img src="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/CO2_latent_4_PPM.gif" width="1000">
     <a href="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/CO2_latent_4_PPM.gif"><strong>Reconstructed by 3-SFC-VCAE-NN, 4 latent variables</strong></a>
  </p>
</p>


### Slugflow
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-jy220/SFC-CAE-Ready-to-use/blob/main/Colab_Notebooks/Instruction_SFC_CAE_Slugflow.ipynb)

<p align="center">
  <p float="middle">
     <p float="middle">
     <img src="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/original_slugflow.gif">
     <p float="middle">
     <a href="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/original_slugflow.gif"><strong>Original Volume Fraction of the Slugflow</strong></a>
     <p float="middle">
     <img src="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/reconstructed_slugflow.gif">
     <p float="middle">
     <a href="https://github.com/acse-jy220/SFC-CAE-gifs/blob/main/reconstructed_slugflow.gif"><strong>Reconstructed by 3-SFC-CAE-NN, 64 latent</strong></a>
  </p>
</p>

## tSNE plots
The creation of t-SNE plots in the Thesis are offered,

After you get FPC-CG data as well as sfcs by
```sh
$ bash get_FPC_data_CG.sh 
```

run  
```sh
$ python3 tSNE.py
```
at the root of this directory.
<p align="center">
  <p float="middle">
     <img src="pics/t-SNE-AE-CG-16-latent.png" width="400">
     <a href="pics/t-SNE-AE-CG-16-latent.png"><strong>t-SNE for SFC-CAE</strong></a>
     <img src="pics/t-SNE-VAE-CG-16-latent.png" width="400">
     <a href="pics/t-SNE-VAE-CG-16-latent.png"><strong>t-SNE for SFC-VCAE</strong></a>
  </p>
</p>

## Decompressing Examples
I have attached the compressing variables for the CO2 and Slugflow data in [decompressing_examples/](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/decompressing_examples/), scripts for downloading pretrained models/ decompressing `vtu` files are introduced in that folder.

## Training on HPC
I wroted a (not very smart) simple script to do training using command line, simply do:
```python
$ python3 command_train.py
```
will do training based on the configuration file `parameters.ini`, all parameters goes there for training on the College HPC.
You could also write a custom configuration file, say `my_config.ini`, and training w.r.t that, by passing argument:
```python
$ python3 command_train.py my_config.ini
```

## License

Distributed under the [Apache 2.0](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/LICENSE) License.

## Testing 
Some basic tests for the module are avaliable in [tests/tests.py](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/tests/tests.py) , you could execute them locally by 
```sh
$ python3 -m pytest tests/tests.py --doctest-modules -v
```
at the root of the repository, by running it, you will automatically download the **FPC_CG** data and two pretrained model (one SFC-CAE, one SFC-VCAE) for that problem and the MSELoss() / KL_div will be evaluated. A [github workflow](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/.github/workflows/test.yml) is also built to run those tests on github.

## Contact
* Jin Yu jin.yu20@imperial.ac.uk or yu19832059@gmail.com

## Acknowledgements
Great thanks to my supervisors:
* Dr. Claire Heaney  [[mail](mailto:c.heaney@imperial.ac.uk)]
* Prof. Christopher Pain  [[mail](mailto:c.pain@imperial.ac.uk)]
