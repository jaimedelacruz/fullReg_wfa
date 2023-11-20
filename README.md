# WFA_fullReg
A fully regularized (in space and time) weak-field approximation implementation.

This module has a C++ backend that takes care of the number crunching in parallel,
while keeping a convenient python interface. 


## Compilation of the C++ module
This module makes use of the Eigen-3 (3.3.7) and FFTW-3 
libraries, which should be in your path. The compilation has been tested
in Linux and Mac systems (with MacPorts).

To compile it simply use:
```
python3 setup.py build_ext --inplace
```

If everything compiles well, you should see a new file called WFA_fullReg.???.so
that should be copied along with wfa_fullReg.py to your PYTHONPATH folder or
to the folder where you want to execute these routines.

NOTE: We need a modern version of Eigen3. If your system does not have it,
simply download the latest stable version and untar it in the pyMilne directory.
Just rename the eigen-3.3.7 folder to eigen3 and the code should compile.

## Usage
We refer to the commented example.py file that is included with the distribution.
We have also prepared an example with a real SST/CRISP dataset that can be found in the example_CRISP/ folder. Simply run invert_crisp.py. That example is also extensively commented.
We also have included an example that makes use of the spatially-regularized Levenberg-Marquardt (invert_crisp_spatially_regularized.py).

## Citing
These routines were developed and used as part of the study by [de la Cruz Rodriguez & Leenaarts in prep.](bla). If you find these routines useful for your research, I would appreciate it the most if that publication is cited in your paper.

## Acknowledgements
This project has been funded by the European Union (ERC, MAGHEAT, 101088184). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
