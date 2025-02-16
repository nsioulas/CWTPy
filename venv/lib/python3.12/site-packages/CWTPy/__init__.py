"""
CWTPy - A fast continuous wavelet transform package.
"""

__version__ = "0.1.0"

# Import everything from the C++ extension module so users can do:
#    from CWTPy import cwt_morlet_full, ...
from .cwt_module import *
