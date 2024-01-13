import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from astropy.io import fits
import wfa_fullReg as TR
import time

# *****************************************************

def readfits(filename, dtype, ext=0):
    io = fits.open(filename, 'readonly', ignore_missing_end=True)
    dat = io[ext].data.astype(dtype, copy=False)
    
    return dat

# *****************************************************

def readData():

    # Read fits files, the data is compressed and scaled
    
    root = "CaII_8542_QS_{0}.fits"
    wav = readfits(root.format("wavelength"), np.float64, 0)
    dat = readfits(root.format("data"), np.float64, 1)

    
    # Reshape to a 5D cube
    
    dum, ny, nx = dat.shape
    nw = wav.size
    ns = 4
    nt = dum // (nw*ns)
    dat = dat.reshape((nt,ns,nw,ny,nx))

    
    # Apply Scaling factors (used to compress the data as int16)
    
    dat[:,0,:,:,:] *= 1.0 / 15000.
    dat[:,1::,:,:,:] *= 1.0 / 500000.


    
    # reshape the data so the axes are (nt, ny, nx, nStokes, nWav)
    
    dat = np.ascontiguousarray(dat.transpose((0,3,4,1,2)))
    
    
    return wav, dat

# *****************************************************

if __name__ == "__main__":

    nthreads = 8 # Adjust this number according to your machine capabilities
    
    
    # Load dataset

    wav, dat = readData()
    nt, ny, nx, ns, nw = dat.shape

    
    # init spectral line and noise level
    
    line = TR.line(8542)
    sig = np.zeros((ns,nw)) + 3.e-3


    # Define spatio-temporal regularization
    
    reg_time = 50.0
    reg_space = 50.0
    reg_lownorm = 0.01
    mask = np.arange(9)


    # calculate Blos using regularization

    BlosReg = TR.getBlos(wav, dat, sig, line, reg_time, reg_space, beta=reg_lownorm,\
                         nthreads=nthreads, Bnorm=300., mask = mask, verbose=True)

    
    # now, for comparison, set regularization to zero and recompute
    
    Blos = TR.getBlos(wav, dat, sig, line, reg_time*0, reg_space*0, beta=reg_lownorm*0,\
                      nthreads=nthreads, Bnorm=300., mask = mask, verbose=True)



    # show results

    asp = ny/nx
    plt.close("all")
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(4+0.2,4*asp/2), sharey=True)
    
    ext = np.float64((0,nx,0,ny))*0.058
    ima0 = ax[0].imshow(Blos[0], cmap="gist_gray", vmax=60, vmin=-60, interpolation="nearest", extent=ext)
    ima1 = ax[1].imshow(BlosReg[0], cmap="gist_gray", vmax=60, vmin=-60, interpolation="nearest", extent=ext)

    ax[0].set_ylabel("y [arcsec]")
    ax[0].set_xlabel("x [arcsec]")
    ax[1].set_xlabel("x [arcsec]")
    ax[0].set_title("Unconstrained")
    ax[1].set_title("Regularized")
    
    f.subplots_adjust(hspace=0.03,bottom=0.07, left=0.125, right=0.90, top=0.875)

    
    # colorbar
    
    ax1 = f.add_subplot(921)
    tmp1 = ax[0].get_position()
    tmp2 = ax[1].get_position()
    pos = (tmp1.x0,0.93,tmp2.x0-tmp1.x0+tmp2.width,0.025)
    ax1.set_position(pos)
    ax1.set_yticks([])
    ax1.set_xticks([-60,-30,0,30,60])
    ax1.imshow(np.arange(256).reshape((1,256)), extent=(-60,60,0,1), aspect='auto', cmap='gist_gray')
    ax1.set_title(r"$B_{\parallel}$ [G]")


    
    # Animate
    
    print("press ctrl+c to stop animation ...")

    while(1):
        for tt in range(nt):
            ima0.set_data(Blos[tt])
            ima1.set_data(BlosReg[tt])

            f.canvas.draw()
            f.canvas.flush_events()
            time.sleep(0.1)
