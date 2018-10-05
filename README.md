# Fast Relaxed Vector Fitting Implementation in C++

The `vectfit` function approximates a response ![fs](https://latex.codecogs.com/gif.latex?\mathbf{f}(s))
(generally a vector) with a rational function:

![vf](https://latex.codecogs.com/gif.latex?\mathbf{f}(s)&space;\approx&space;\sum_{j=1}^{N}&space;\frac{\mathbf{r}_j}{s-p_j}&space;&plus;&space;\sum_{n=0}^{Nc}&space;\mathbf{c}_n&space;s^{n})

where ![pj](https://latex.codecogs.com/gif.latex?p_j) and
![rj](https://latex.codecogs.com/gif.latex?r_j) are poles and residues in
the complex plane and ![cn](https://latex.codecogs.com/gif.latex?c_n) are
the polynomial coefficients.
When ![fs](https://latex.codecogs.com/gif.latex?\mathbf{f}(s)) is a vector, all
elements become fitted with a common pole set.

The identification is done using the pole relocating method known as Vector
Fitting [1] with relaxed non-triviality constraint for faster convergence
and smaller fitting errors [2], and utilization of matrix structure for fast
solution of the pole identifion step [3].

- [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
    domain responses by Vector Fitting", IEEE Trans. Power Delivery,
    vol. 14, no. 3, pp. 1052-1061, July 1999.
- [2] B. Gustavsen, "Improving the pole relocating properties of vector
    fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
    July 2006.
- [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
    "Macromodeling of Multiport Systems Using a Fast Implementation of
    the Vector Fitting Method", IEEE Microwave and Wireless Components
    Letters, vol. 18, no. 6, pp. 383-385, June 2008.

All credit goes to Bjorn Gustavsen for his MATLAB implementation.
(http://www.sintef.no/Projectweb/VECTFIT/)


## Implementation

The `vectfit` functions are implemented in C++, using [`xtensor`](https://github.com/QuantStack/xtensor)
for multi-dimensional arrays and [`xtensor-blas`](https://github.com/QuantStack/xtensor-blas)
for linear-algebra operations.
The C++ functions are wrapped into a Python extension module through
[`xtensor-python`](https://github.com/QuantStack/xtensor-python) and
[`pybind11`](https://github.com/pybind/pybind11).

``` python
import vectfit as m
import numpy as np

s = np.array([3., 3.5, 4., 4.5, 5., 5.5, 6.])
f = np.array([[4.98753117e-02, 3.09734513e-01, 1.18811881e+00,
               6.53846154e+00, 2.20000000e+02, 1.03846154e+01,
               3.16831683e+00],
             [-4.98753117e-01,-2.21238938e-01, 9.90099010e-01,
               9.61538462e+00, 4.00000000e+02, 2.11538462e+01,
               6.93069307e+00]])
weight = 1.0/f
init_poles = [4.5 + 0.045j, 4.5 - 0.045j]
poles, residues, cf, fit, rms = m.vectfit(f, s, init_poles, weight)
```

## Installation

### Prerequisites

- C++ compiler such as g++

  `sudo apt install g++`

- Necessary libraries including [`xtensor`](https://github.com/QuantStack/xtensor),
  [`xtensor-blas`](https://github.com/QuantStack/xtensor-blas),
  [`xtensor-python`](https://github.com/QuantStack/xtensor-python), and
  [`pybind11`](https://github.com/pybind/pybind11)

  It is convenient to install all the libraries through conda package manager:

  `conda install -c conda-forge xtensor-blas xtensor-python`

### Build and install

**On Unix (Linux, OS X)**

 - clone this repository
 - `pip install ./vectfit`

**On Windows (Requires Visual Studio 2015)**

 - For Python 3.5:
     - clone this repository
     - `pip install ./vectfit`
 - For earlier versions of Python, including Python 2.7:

   xtensor requires a C++14 compliant compiler (i.e. Visual Studio 2015 on
   Windows). Running a regular `pip install` command will detect the version
   of the compiler used to build Python and attempt to build the extension
   with it. We must force the use of Visual Studio 2015.

     - clone this repository
     - `"%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x64`
     - `set DISTUTILS_USE_SDK=1`
     - `set MSSdk=1`
     - `pip install ./vectfit`

   Note that this requires the user building `vectfit` to have registry edition
   rights on the machine, to be able to run the `vcvarsall.bat` script.


## Windows runtime requirements

On Windows, the Visual C++ 2015 redistributable packages are a runtime
requirement for this project. It can be found [here](https://www.microsoft.com/en-us/download/details.aspx?id=48145).

If you use the Anaconda python distribution, you may require the Visual Studio
runtime as a platform-dependent runtime requirement for you package:

```yaml
requirements:
  build:
    - python
    - setuptools
    - pybind11

  run:
   - python
   - vs2015_runtime  # [win]
```

## Building the documentation

Documentation for the example project is generated using Sphinx. Sphinx has the
ability to automatically inspect the signatures and documentation strings in
the extension module to generate beautiful documentation in a variety formats.
The following command generates HTML-based reference documentation; for other
formats please refer to the Sphinx manual:

 - `cd vectfit/docs`
 - `make html`

## Running the tests

Running the tests requires `pytest`.

```bash
py.test .
```
