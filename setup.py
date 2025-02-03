#from distutils.core import setup
from setuptools import setup
from setuptools.extension import Extension
#from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy
import platform as plt
import sys
import pathlib

os.system('rm WFA_fullReg.*.so src/REG.cpp')
p = pathlib.Path(sys.executable)
root_dir = str(pathlib.Path(*p.parts[0:-2]))


if(plt.system() == 'Darwin'):
    root_dir = '/Users/jade0464/miniforge3/' # using this one if macports are installed
    CC = 'clang'
    CXX= 'clang++'
    link_opts = ["-stdlib=libc++","-bundle","-undefined","dynamic_lookup", "-fopenmp","-lgomp"]
else:
    root_dir = '/usr/'
    CC = 'gcc'
    CXX= 'g++'
    link_opts = ["-shared", "-fopenmp"]

os.environ["CC"] = CC
os.environ["CXX"] = CXX


# Optimization flags. With M-processor Macs remove the -march=native!

comp_flags=['-O3', '-flto','-g0','-fstrict-aliasing','-mcpu=native','-mtune=native',\
            '-std=c++20','-fPIC','-fopenmp', '-I./src', "-DNPY_NO_DEPRECATED_API", '-DNDEBUG', \
            '-pedantic', '-Wall']


extension = Extension("WFA_fullReg",
                      sources=["src/REG.pyx", "src/fullReg.cpp"], 
                      include_dirs=["./","./src/",numpy.get_include(), root_dir+'/include/eigen3/', root_dir+"/include/"],
                      language="c++",
                      extra_compile_args=comp_flags,
                      extra_link_args=comp_flags+link_opts,
                      library_dirs=['./',"/usr/lib/"],
                      libraries=[])

extension.cython_directives = {'language_level': "3"}

setup(
    name = 'WFA_fullReg',
    version = '1.0',
    author = 'J. de la Cruz Rodriguez & Jorrit Leenaarts (ISP-SU, 2023)',
    ext_modules=[extension],
    cmdclass = {'build_ext': build_ext}
)

