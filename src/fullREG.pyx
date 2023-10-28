"""
CYTHON interface for C++ tools.
Author: J. de la Cruz Rodriguez (ISP-SU, 2023)
"""
cimport numpy as np
from numpy cimport ndarray as ar
from numpy import zeros, abs, sqrt, arctan2, where, pi, float32, float64, ndarray
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string


__author__="J. de la Cruz Rodriguez"
__status__="Developing"

# *****************************************************************************************

cdef extern from "fullReg.hpp" namespace "reg":

     # ----------------------------------------------------------

     cdef void timeRegularization(long ny, \
			          long nx, \
			          long nt, \
			          const double* const rhs, \
                                  const double* const lhs, \
                                  double alpha_t, \
                                  double alpha_s, \
                                  double beta, \
                                  double* const res, \
                                  int nthreads);
     
     # ----------------------------------------------------------

     cdef void centDer(int  N, const double* const x,
	               const double* const y, double* const yp);

     # ----------------------------------------------------------

     cdef void centDerMany(long  nPix, int  N, const double* const x,
		   const double* const y, double* const yp, int nthreads);
     
     # ----------------------------------------------------------

# *****************************************************************************************

def getDerivativeOne(ar[double,ndim=1] x, ar[double,ndim=1] y):

    cdef int N = x.size
    cdef ar[double,ndim=1] yp = zeros((N), dtype='float64')

    centDer(N, <double*>x.data, <double*>y.data, <double*>yp.data)

    return yp

# *****************************************************************************************

def getDerivativeMany(ar[double,ndim=1] x, ar[double,ndim=3] y, int nthreads=4):

    cdef long ny = y.shape[0]
    cdef long nx = y.shape[1]
    cdef long nPix = ny*nx
    cdef int N = y.shape[2]
    
    cdef ar[double,ndim=3] yp = zeros((ny,nx,N), dtype='float64')

    centDerMany(nPix, N, <double*>x.data, <double*>y.data, <double*>yp.data, <int>nthreads)

    return yp

# *****************************************************************************************
    
def spatial_constraints_spat_time(long ny, long nx, long nt, double alpha_t, double alpha_s,\
                                  double beta, ar[double,ndim=1] lhs, ar[double,ndim=1] rhs, \
                                  int nthreads = 4):

    
    cdef ar[double,ndim=3] res = zeros((nt,ny,nx), dtype=float64)
        
    timeRegularization(ny, nx, nt, <double*>rhs.data, <double*>lhs.data, \
                       alpha_t, alpha_s, beta, <double*>res.data, nthreads)

    
    return res
    
    
# *****************************************************************************************
