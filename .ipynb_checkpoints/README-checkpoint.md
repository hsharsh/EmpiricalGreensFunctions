
This is a Python implementation of a data-driven approach to mathematically model physical systems whose governing partial differential equations are unknown, by learning their associated Green's function. The package is implemented in python and uses Jupyter notebooks to demonstrate the examples. I personally recommend using conda to install the package dependencies in a virtual environment (because some packages require older versions of python and there can be dependency issues).

```bash
conda create -n env -c conda-forge fenics mshr matplotlib scipy numpy pandas jupyterlab ipympl 
conda activate env
```
The code also samples randomly from a Gaussian Processs with a Squared-Exponential Kernel. That bit is implemented in MATLAB https://www.mathworks.com/help/install/install-products.html) using chebfun (https://www.chebfun.org/download/).

If you want to install the packages individually, the stack includes:

- FENICS: https://fenicsproject.org/download/
- mshr: https://anaconda.org/conda-forge/mshr
- scipy: https://anaconda.org/anaconda/scipy
- numpy: https://anaconda.org/anaconda/numpy
- matplotlib: https://anaconda.org/conda-forge/matplotlib
- pandas: https://anaconda.org/anaconda/pandas