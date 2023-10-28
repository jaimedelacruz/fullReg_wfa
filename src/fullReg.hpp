#ifndef TIMEREGHPP
#define TIMEREGHPP

namespace reg{
  
  // ********************************************************* //

  void centDer(int const N, const double* const x,
	       const double* const y, double* const yp);
  
  // ********************************************************* //

  void centDerMany(long const nPix, int const N, const double* const x,
		   const double* const y, double* const yp, int const nthreads);
  
  // ********************************************************* //
  
  void timeRegularization(long const ny, 
			  long const nx, 
			  long const nt, 
			  const double* const rhs,
			  const double* const lhs,
			  double const alpha_t,
			  double const alpha_s,
			  double const beta,
			  double* const res,
			  int const nthreads);
  
  // ********************************************************* //

}

#endif
