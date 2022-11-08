
This is a Python implementation of a data-driven approach to mathematically model physical systems whose governing partial differential equations are unknown, by learning their associated Green's function. The package is implemented in python and uses Jupyter notebooks to demonstrate the examples. I personally recommend using anaconda to install the package dependencies in a virtual environment. An environment with all the dependencies is available at https://anaconda.org/praveenharsh01/egf. If you want to install the packages individually, the stack includes:

- FENICS: https://fenicsproject.org/download/
- mshr: https://anaconda.org/conda-forge/mshr
- scipy: https://anaconda.org/anaconda/scipy
- numpy: https://anaconda.org/anaconda/numpy
- matplotlib: https://anaconda.org/conda-forge/matplotlib
- pandas: https://anaconda.org/anaconda/pandas

The code also samples randomly from a Gaussian Processs with a Squared-Exponential Kernel. That bit is implemented in MATLAB (https://it.cornell.edu/software-licensing/install-matlab) using chebfun (https://www.chebfun.org/download/).