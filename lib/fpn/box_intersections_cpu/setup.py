from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    name="bbox_cython",
    ext_modules=cythonize("bbox.pyx"),
    include_dirs=[numpy.get_include()],
)
