import os
import sys
from setuptools import setup, Extension, find_packages
import pybind11

# ---------------------------
# 1) Determine OpenMP Flags
# ---------------------------
extra_compile_args = ["-O3", "-std=c++11"]
extra_link_args = []
if sys.platform.startswith("linux") or sys.platform.startswith("win32"):
    extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")
elif sys.platform == "darwin":
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args += ["-lomp"]





# ---------------------------
# 2) Determine FFTW Paths
# ---------------------------
# Default paths: Homebrew on macOS, /usr/local on Linux.
fftw_inc_default = "/opt/homebrew/include" if sys.platform == "darwin" else "/usr/local/include"
fftw_lib_default = "/opt/homebrew/lib"     if sys.platform == "darwin" else "/usr/local/lib"

fftw_inc = os.environ.get("FFTW_INC", fftw_inc_default)
fftw_lib = os.environ.get("FFTW_LIB", fftw_lib_default)

# ---------------------------
# 3) Define the Extension with ABI3 Support
# ---------------------------
module = Extension(
    "CWTPy.cwt_module",
    sources=["CWTPy/cwt_module.cpp"],
    include_dirs=[pybind11.get_include(), fftw_inc],
    libraries=["fftw3", "fftw3_threads"],
    library_dirs=[fftw_lib],
    extra_compile_args=extra_compile_args,  # Note: removed -DPy_LIMITED_API=0x03060000
    extra_link_args=extra_link_args,
    # Removed: py_limited_api=True
)

setup(
    name="CWTPy",
    version="0.1.3",  # Update the version here
    description="A fast continuous wavelet transform (CWT) implementation using C++/FFTW/pybind11.",
    author="Nikos Sioulas",
    author_email="nsioulas@berkeley.edu",
    url="https://github.com/yourusername/CWTPy",
    packages=find_packages(),
    ext_modules=[module],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

