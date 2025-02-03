/* ---
   Fully regularized WFA routines in space and time

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)
   References:
               de la Cruz Rodriguez & Leenaarts (in prep.)
	       Morosin, de la Cruz Rodriguez, et al. (2020)
   --- */
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
//#include <unsupported/Eigen/IterativeSolvers>

#include <cstdio>
#include <cmath>

#include "fullReg.hpp"

using namespace std;

using Mat = Eigen::SparseMatrix<double, Eigen::RowMajor, long>;
using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using VecMap = Eigen::Map<Vec>;
using cVecMap = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>>;

// ********************************************************* //

  template<class T>
  inline T signFortran(T const &val)
  {return ((val < static_cast<T>(0)) ? static_cast<T>(-1) : static_cast<T>(1));}

// ********************************************************* //

double harmonic_derivative_Steffen_one(double const xu, double const x0, double const xd,
				       double const yu, double const y0, double const yd)
  {
    // ---
    // High order harmonic derivatives
    // Ref: Steffen (1990), A&A..239..443S
    //
    // Arguments:
    //   Assuming three consecutive points of a function yu, y0, yd and the intervals between them odx, and dx:
    //      odx: x0 - xu
    //       dx: xd - x0
    //       yu: Upwind point
    //       y0: Central point
    //       yd: downwind point
    //
    // ---

    using T = double;
    
    T const dx = (xd - x0);
    T const odx = (x0 - xu);
    T const S0 = (yd - y0) / dx;
    T const Su = (y0 - yu) / odx;
    T const P0 = std::abs((Su*dx + S0*odx) / (odx+dx)) * 0.5;
    
    return (signFortran(S0) + signFortran(Su)) * std::min<T>(std::abs(Su),std::min<T>(std::abs(S0), P0));
  }
  
// ********************************************************* //

void reg::centDer(int const N, const double* const __restrict__ x,
		  const double* const __restrict__ y, double* const __restrict__ yp)
{
  int const N1 = N-1;
  
  yp[0] = (y[1]-y[0]) / (x[1]-x[0]);
  yp[N-1] = (y[N1]-y[N-2]) / (x[N1]-x[N-2]);

  
  for(int ii=1; ii<N1; ++ii){
    yp[ii] = harmonic_derivative_Steffen_one(x[ii-1], x[ii], x[ii+1], y[ii-1], y[ii], y[ii+1]);
  }
  
}

// ********************************************************* //

void reg::centDerMany(long const nPix,
		      int const N,
		      const double* const __restrict__ x,
		      const double* const __restrict__ y,
		      double* const __restrict__ yp,
		      int const nthreads)
{
  
  omp_set_num_threads(nthreads);
  
#pragma omp parallel default(shared) num_threads(nthreads)
  {
#pragma omp for
    for(long ipix=0; ipix<nPix; ++ipix){
      
      reg::centDer(N, x, &y[N*ipix], &yp[N*ipix]);
      
    } // ipix
  } // parallel block 
}

// ********************************************************* //

Mat BuildLinearSytem(long const ny, long const nx, long const nt,
		     const double* const __restrict__ lhs,
		     double const alpha_t, double const alpha_s,
		     double const beta, int const verbose)
{

  long const nDat = ny*nx*nt;
  Mat A(nDat,nDat);
  
  
  // --- we will first count the number of elements per row --- //
  
  Eigen::VectorXi n_elements(nDat); n_elements.setConstant(1); // Diagonal term already counted 

  if(verbose){
    fprintf(stderr,"[info] problem dimensions -> ny=%ld, nx%ld, nt=%ld\n", ny, nx, nt);
    fprintf(stderr,"[info] counting non-zero elements of sparse matrix with dimensions %ld x %ld ... ", nDat,nDat);
  }

  
  // --- Count non-zero terms --- //

  long const ny1 = ny-1;
  long const nx1 = nx-1;
  long const nt1 = nt-1;

  for(long tt=0; tt<nt; ++tt){
    for(long yy=0; yy<ny; ++yy){      
      for(long xx=0; xx<nx; ++xx){
	if(tt > 0) n_elements[((tt-1)*ny+yy)*nx+xx]     += 1;
	if(yy > 0) n_elements[(tt*ny+(yy-1))*nx+xx]     += 1;
	if(xx > 0) n_elements[(tt*ny+yy)*nx+(xx-1)]     += 1;
	
	if(xx < nx1) n_elements[(tt*ny+yy)*nx+(xx+1)]   += 1;
	if(yy < ny1) n_elements[(tt*ny+(yy+1))*nx+xx]   += 1;
	if(tt < nt1) n_elements[((tt+1)*ny+yy)*nx+xx]   += 1;

      }
    } // xx
  } // yy


  // --- preallocate non-zero elements --- //
  
  A.reserve(n_elements);

  

  // --- populate matrix --- //

  if(verbose)
    fprintf(stderr,"filling non-zero terms in matrix ... ");

  
  
  // --- Fill matrix --- //
  
  for(long tt=0; tt<nt; ++tt){
    for(long yy=0; yy<ny; ++yy){
      for(long xx=0; xx<nx; ++xx){
	
      
      // --- all time step --- //

	long const y = (tt*ny+yy)*nx+xx;
	
	int ntadded = 0;
	int nsadded = 0;

	if(tt > 0) A.insert(y,((tt-1)*ny+yy)*nx+xx) = -alpha_t, ++ntadded; // (yy,xx,tt-1)
	if(yy > 0) A.insert(y,(tt*ny+(yy-1))*nx+xx) = -alpha_s, ++nsadded; // (yy-1,xx,tt)
	if(xx > 0) A.insert(y,y-1)                  = -alpha_s, ++nsadded; // (yy,xx-1,tt) 

	A.insert(y,y)                               =  lhs[y] + beta;      // (yy,xx,tt)

	if(xx<nx1) A.insert(y,y+1)                  = -alpha_s, ++nsadded; // (yy,xx+1,tt)
	if(yy<ny1) A.insert(y,(tt*ny+(yy+1))*nx+xx) = -alpha_s, ++nsadded; // (yy+1,xx,tt)
	if(tt<nt1) A.insert(y,((tt+1)*ny+yy)*nx+xx) = -alpha_t, ++ntadded; // (yy,xx,tt+1)

	A.coeffRef(y,y) += nsadded*alpha_s + ntadded*alpha_t;
      }
    } // xx
  } // yy

  if(verbose)
    fprintf(stderr,"done\n");  
  
  return A;
}

// ********************************************************* //

void getInitialSolution(long const ny, long const nx, long const nt, double const beta_in,
			const double* const lhs, const double* const rhs, double* const res)
{
  constexpr static long const nSum=1; // +/- from the central pixel
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> tmp(ny,nx);

  constexpr const double psf[3][3] = {{0.0625,0.125,0.0625}, {0.125,0.25,0.125},{0.0625,0.125,0.0625}};

  double const beta = std::max(beta_in, 5.0E-15);
  
  
  for(long tt=0; tt<nt;++tt){

    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> LHS(&lhs[tt*ny*nx],ny,nx);
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> RHS(&rhs[tt*ny*nx],ny,nx);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> img(&res[tt*ny*nx],ny,nx);

    
    // --- Use as initial result the unconstrained solution, using beta --- //
    
    for(long yy=0;yy<ny; ++yy)
      for(long xx=0;xx<nx; ++xx){
	tmp(yy,xx) = RHS(yy,xx) / (LHS(yy,xx) + beta);
      }


    
    // --- now use a 3x3 kernel to smooth prediction --- //

    for(long yy=0;yy<ny; ++yy){

      long const j0 = std::max<long>(0,   yy-nSum);
      long const j1 = std::min<long>(ny-1,yy+nSum);
      long const nj = j1-j0+1;
      
      for(long xx=0;xx<nx; ++xx){
	
	long const i0 = std::max<long>(0,   xx-nSum);
	long const i1 = std::min<long>(nx-1,xx+nSum);
	long const ni = i1-i0+1;
	
	double sum = 0.0;
	
	for(long jj=j0; jj<=j1; ++jj)
	  for(long ii=i0; ii<=i1; ++ii){
	    sum += tmp(jj,ii)*psf[jj-yy+1][ii-xx+1];
	  }
	img(yy,xx) = sum;
	
      } // xx
    } // yy    
  } // tt
  
}

// ********************************************************* //

void reg::timeRegularization(long const ny, 
			     long const nx, 
			     long const nt, 
			     const double* const rhs,
			     const double* const lhs,
			     double const alpha_t,
			     double const alpha_s,
			     double const beta,
			     double* const res,
			     int const nthreads,
			     int const verbose)
{

  // --- Init number of threads --- //
  
  Eigen::setNbThreads(nthreads);
  omp_set_num_threads(nthreads);


  
  // --- Build the linear system --- //
  
  Mat A = BuildLinearSytem(ny, nx, nt, lhs, alpha_t, alpha_s, beta, verbose);

  

  // --- Get a simple initial solution to minimize iterations --- //
  
  getInitialSolution(ny,nx,nt,beta,lhs,rhs,res);
  

  
  // --- Solve linear system with BiCGSTAB using an educated initial guess --- //

  long const nDat = nt*ny*nx;
  cVecMap B(rhs,nDat);
  VecMap Res(res,nDat);

  if(verbose)
    fprintf(stderr,"[info] inverting linear system ... ");

  Eigen::BiCGSTAB<Mat> solver(A);  
  Res = solver.solveWithGuess(B, Res);
  
  
  if(verbose)
    fprintf(stderr,"done\n");

}

// ********************************************************* //
