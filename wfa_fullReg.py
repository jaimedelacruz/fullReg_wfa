"""
Fully regularized WFA approximation implementation
Coded by J. de la Cruz Rodriguez (ISP-SU, 2023)

References:
de la Cruz Rodriguez & Leenaarts (in prep.)
Morosin et al. (2020)
"""
import WFA_fullReg as TR
import numpy as np

# *************************************************************************

class line:
    """
    Class line is used to store the atomic data of spectral lines. We use this
    class as input for the WFA routines below.
    Usage: 
        lin = line(8542)
    """
    def __init__(self, cw=8542):

        self.larm = 4.668645048281451e-13
        
        if(cw == 8542):
            self.j1 = 2.5; self.j2 = 1.5; self.g1 = 1.2; self.g2 = 1.33; self.cw = 8542.091
        elif(cw == 6301):
            self.j1 = 2.0; self.j2 = 2.0; self.g1 = 1.84; self.g2 = 1.50; self.cw = 6301.4995
        elif(cw == 6302):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.49; self.g2 = 0.0; self.cw = 6302.4931
        elif(cw == 8468):
            self.j1 = 1.0; self.j2 = 1.0; self.g1 = 2.50; self.g2 = 2.49; self.cw = 8468.4059
        elif(cw == 6173):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.50; self.g2 = 0.0; self.cw = 6173.3340
        elif(cw == 5173):
            self.j1 = 1.0; self.j2 = 1.0; self.g1 = 1.50; self.g2 = 2.0; self.cw = 5172.6843
        elif(cw == 5896):
            self.j1 = 0.5; self.j2 = 0.5; self.g1 = 2.00; self.g2 = 2.0/3.0; self.cw = 5895.9242
        elif(cw == 6563):
            self.cw = 6562.8518
            self.geff = 1.048 # Casini & Landi Degl'Innocenti (1994)
            self.Gg = 0.0 # No Q & U data?
            return
        else:
            print("line::init: ERROR, line not implemented")
            self.j1 = 0.0; self.j2 = 0.0; self.g1 = 0.0; self.g2 = 0.0; self.cw = 0.0
            return

        j1 = self.j1; j2 = self.j2; g1 = self.g1; g2 = self.g2
        
        d = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        self.geff = 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d;
        ss = j1 * (j1 + 1.0) + j2 * (j2 + 1.0);
        dd = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        gd = g1 - g2;
        self.Gg = (self.geff * self.geff) - (0.0125  * gd * gd * (16.0 * ss - 7.0 * dd * dd - 4.0));

        print("line::init: cw={0}, geff={1}, Gg={2}".format(self.cw, self.geff, self.Gg))

# *************************************************************************

def getBlos(wav, obs, sig, line, alpha_time, alpha_spat, beta = 0.0, \
            nthreads=4, Bnorm=300., mask = None, verbose = True):
    """
    Computes the Bparallel from the Stokes data.
    
    Input: 
         wav: 1D array with the wavelength offset from line center [Angstroms].
         obs: 4D [ny,nx,nStokes,nLambda] or 5D [nt,ny,nx,nStokes,nLambda] cube with the data.
         sig: 2D array [nStokes,nLambda] with the estimate of the noise for each spectral point [float64].
         lin: line object containing the information of the spectral line (see above).
  alpha_time: regularization weight in the temporal direction.
  alpha_spat: regularization weight in the spatial direction.
        beta: low-norm regularization weight (prefers solution with a lower B amplitude).
    nthreads: number of threads (int)
       Bnorm: typical magnetic field strength to scale the regularization terms relative to Chi2.
        mask: indexes of the wavelengths that we want to use in the calculation (optional).
     verbose: printout information (or not). Default True.

    Coded by J. de la Cruz Rodriguez (ISP-SU, 2023).
    
    """
    
    # get dimensions
    nt, ny, nx, ns, nw = obs.shape
    
    
    # Init tmp storage
    lhs = np.zeros((nt, ny, nx), dtype='float64', order='c')
    rhs = np.zeros((nt, ny, nx), dtype='float64', order='c')

    
    # Init constants
    c = -line.larm * line.cw**2 * line.geff; cc = c*c


    if(mask is None):
        mask = np.arange(nw)

    inw = len(mask)
    
    for tt in range(nt):
        StokesI =  np.ascontiguousarray(obs[tt,:,:,0,:].squeeze(), dtype='float64')
        der = TR.getDerivativeMany(np.float64(wav),StokesI, \
                                   nthreads = nthreads)

        
        for ii in mask:

            isig2 = inw*sig[3,ii]**2
            lhs[tt,:,:] += cc*der[:,:,ii]**2 / isig2
            rhs[tt,:,:] += c *der[:,:,ii]*obs[tt,:,:,3,ii] / isig2

    Blos = TR.spatial_constraints_spat_time(ny, nx, nt, alpha_time/((nt-1)*Bnorm**2), alpha_spat/(2*Bnorm**2), \
                                            beta/(2*Bnorm**2), lhs.flatten(), rhs.flatten(), int(nthreads), \
                                            verbose=int(verbose))
    
    return Blos
    
# *************************************************************************

def getBhorAzi(wav, obs, sig, lin, alpha_time, alpha_spat, beta = 0.0, \
               nthreads=4, Bnorm=300., mask = None, vdop = 0.06, verbose=True):
    """
    Computes the Btrans and the azimuth from the Stokes data
    
    Input: 
         wav: 1D array with the wavelength offset from line center [Angstroms].
         obs: 4D [ny,nx,nStokes,nLambda] or 5D [nt,ny,nx,nStokes,nLambda] cube with the data.
         sig: 2D array [nStokes,nLambda] with the estimate of the noise for each spectral point [float64].
         lin: line object containing the information of the spectral line (see above).
  alpha_time: regularization weight in the temporal direction.
  alpha_spat: regularization weight in the spatial direction.
        beta: low-norm regularization weight (prefers solution with a lower B amplitude).
    nthreads: number of threads (int)
       Bnorm: typical magnetic field strength to scale the regularization terms relative to Chi2.
        mask: indexes of the wavelengths that we want to use in the calculation (optional).
        vdop: typical Doppler width in Angstroms. The line offsets falling inside +/- vdop will be ignored.
     verbose: printout information (or not). Default True.

    Coded by J. de la Cruz Rodriguez (ISP-SU, 2023).
    
    """

    Verbose=1
    if(verbose is not True):
        verbose = 0
    
    # get dimensions
    nt, ny, nx, ns, nw = obs.shape
    
    
    # Init tmp storage
    lhsQ = np.zeros((nt,ny,nx), dtype='float64', order='c')
    rhsQ = np.zeros((nt,ny,nx), dtype='float64', order='c')
    lhsU = np.zeros((nt,ny,nx), dtype='float64', order='c')
    rhsU = np.zeros((nt,ny,nx), dtype='float64', order='c')

    
    # Init constants
    c = 0.75 * (lin.larm * lin.cw**2)**2 * lin.Gg;
    cc = c*c

    if(mask is None):
        mask = np.arange(nw)

    inw = len(mask)
    
    for tt in range(nt):
        StokesI =  np.ascontiguousarray(obs[tt,:,:,0,:].squeeze(), dtype='float64')
        der = TR.getDerivativeMany(np.float64(wav), StokesI, nthreads = nthreads)

        for ii in mask:
            iw = wav[ii]*1

            if(np.abs(iw) < vdop):
                continue
            else:
                scl = 1.0 / iw

                der[:,:,ii] *= scl

                isig2Q = inw*sig[1,ii]**2
                isig2U = inw*sig[2,ii]**2
                
                ider2 = der[:,:,ii]**2
                
                lhsQ[tt,:,:] += cc*ider2 / isig2Q
                lhsU[tt,:,:] += cc*ider2 / isig2U
                
                rhsQ[tt,:,:] += c*obs[tt,:,:,1,ii]*der[:,:,ii] / isig2Q
                rhsU[tt,:,:] += c*obs[tt,:,:,2,ii]*der[:,:,ii] / isig2U


            
    BhorQ_2 = TR.spatial_constraints_spat_time(ny, nx, nt, alpha_time/(4*Bnorm**4), alpha_spat/(4*Bnorm**4), \
                                               beta/(4*Bnorm**4), lhsQ.flatten(), rhsQ.flatten(), int(nthreads),\
                                               verbose=int(verbose))
             
    BhorU_2 = TR.spatial_constraints_spat_time(ny, nx, nt, alpha_time/((nt-1)*Bnorm**4), alpha_spat/(4*Bnorm**4), \
                                               beta/(4*Bnorm**4), lhsU.flatten(), rhsU.flatten(), int(nthreads),\
                                               verbose=int(verbose))


    # calculate the azimuth
    azimuth = 0.5 * np.arctan2(BhorU_2, BhorQ_2)
    azimuth[np.where(azimuth < 0)] += np.pi

    
    # combine contributions to calculate Btrans
    Bhor = np.sqrt(np.sqrt(BhorQ_2**2 + BhorU_2**2))

    return Bhor, azimuth


