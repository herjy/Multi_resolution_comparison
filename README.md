# Multi_resolution_comparisons
A repository of notebooks that show the comparison between the resampling schemes as implemented in galsim and scarlet. This repo can be used to reproduce the figures of Joseph et al. 2020.
 
This repository also contains simulations for multi-resolution scenes (requires galsim). 
The folders in this repo are organised as follows:

* `mr_tools` contains the mr_tools package which holds functions that create multi-resolution galsim simulations. Install it by running 
```
python setup.py install 
```
* `Real_images` contains a set of notebook and scenes exctracted from the COSMOS fields HST and HSC images. The notebooks run scarlet multi resolution on these scenes and shows comparisons with runs of scarlet on independent scenes.
* `Reconstruction_test` contains two notebooks used for qualitative comparison of the interpolation schemes in galsim and scarlet (Galsim_comparison.ipynb)  and a quantitative comparison of the same reconstructions on a subsample of 1000 images from the COSMOS r<23.5 catalog.
* `Simulated_multiresolution` Contains a notebook to test scarlet's multi-resolution framework on simulated blends generated with galsim
* `Timing_test` Contains a notebook used to time the execution of galsim's and scarlet's resampling schemes (Timing_comparison.ipynb). The notebook Time fit.ipynb is used to fit a straight line to the time ratio between scarlet and galsim timings as a function of the number of low resolution samples.


Requires 
* Galsim
* scarlet
* astropy
* Numpy
* Matplotlib
* pickle
