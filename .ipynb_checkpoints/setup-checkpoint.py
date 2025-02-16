from setuptools import setup, Extension, find_packages
import sys
import pybind11

# Determine OpenMP flags based on platform
extra_compile_args = ["-O3", "-std=c++11"]
extra_link_args = []
if sys.platform.startswith("linux") or sys.platform.startswith("win32"):
    extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")
elif sys.platform == "darwin":
    # macOS may require llvm openmp; adjust as needed.
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args += ["-lomp"]

module = Extension(
    "CWTPy.cwt_module",
    sources=["CWTPy/cwt_module.cpp"],
    include_dirs=[pybind11.get_include()],
    libraries=["fftw3"],
    library_dirs=["/opt/homebrew/lib"],  # or use appropriate defaults; allow override via environment
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="CWTPy",
    version="0.1.0",
    description="A fast continuous wavelet transform (CWT) implementation using C++/FFTW/pybind11.",
    author="Nikos Sioulas",
    author_email="nsioulas@berkeley.edu",
    url="https://github.com/yourusername/CWTPy",  # update with your repo URL
    packages=find_packages(),
    ext_modules=[module],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
