# CWTPy

**CWTPy** is a fast, cross-platform Python library for computing the Continuous Wavelet Transform (CWT) using a C++ backend, FFTW, and optional OpenMP parallelization. It builds a native extension module named **`cwt_module`**.

## Features

- **Morlet wavelet** with L2 normalization
- **User-specified** frequency or scale range
- **OpenMP** support for parallel processing
- **Cross-platform** (Linux, macOS, Windows)

## Installation

1. **Install FFTW** on your system (e.g. `sudo apt-get install libfftw3-dev` on Ubuntu or `brew install fftw` on macOS).
2. **Install CWTPy** from PyPI (once published) or from source:
   ```bash
   pip install CWTPy
