## Compute and plot curvature of Skafta cauldron
## 26 Mar 2019  EHU

import numpy as np
import scipy.misc as scp
import scipy.signal as signal
from scipy import interpolate
from scipy.ndimage import gaussian_filter, distance_transform_edt
import mpl_toolkits.basemap.pyproj as pyproj
from osgeo import gdal
from netCDF4 import Dataset
#import shapefile
#import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm, SymLogNorm
#from scipy import interpolate
#from scipy.ndimage import gaussian_filter
from cauldron_funcs import *

skafta_region_path = 'Documents/6. MIT/Skaftar collapse/data/arcticDEM/'
nc_20121015_path = skafta_region_path + 'subset_nc/SETSM_WV02_20121015_skaftar_east_ll.nc'
nc_20151010_path = skafta_region_path + 'subset_nc/SETSM_WV02_20151010_skaftar_east_ll.nc'

lon_2012, lat_2012, se_2012 = read_ArcticDEM_nc(nc_20121015_path)
SE_2012 = np.ma.masked_where(se_2012==0, se_2012)
lon_2015, lat_2015, se_2015 = read_ArcticDEM_nc(nc_20151010_path)
SE_2015 = np.ma.masked_where(se_2015==0, se_2015)

SE_2012_restricted = SE_2012[1000:3500, 6000:10000] #slicing to restrict to only the area of Eastern Skafta (speeds up computation)
SE_2015_restricted = SE_2015[1000:3500, 6000:10000]
#SE_2012_rest_fild = SE_2012_restricted.filled(fill_value=SE_2012_restricted.mean())
#SE_2015_rest_fild = SE_2015_restricted.filled(fill_value=SE_2015_restricted.mean())
SE_2012_rest_fild = NearestFill(SE_2012_restricted, mask=SE_2012_restricted.mask)
SE_2015_rest_fild = NearestFill(SE_2015_restricted, mask=SE_2015_restricted.mask)
lon_restricted = lon_2015[6000:10000]
lat_restricted = lat_2015[1000:3500]

SE_2012_western = SE_2012[1800:3200, 2250:5000]
SE_2015_western = SE_2015[1800:3200, 2250:5000]
lon_western = lon_2015[2250:5000]
lat_western = lat_2015[1800:3200]

## Compute and plot curvature of cauldron surface
def savgol2d ( z, window_size, order, derivative=None): #based on SciPy Cookbook entry on savgol
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'gradient':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return signal.fftconvolve(Z, -r, mode='valid'), signal.fftconvolve(Z, -c, mode='valid')
    elif derivative== 'curvature':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        gradx, grady = signal.fftconvolve(Z, -r, mode='valid'), signal.fftconvolve(Z, -c, mode='valid')
        gradmag = [[np.linalg.norm((gradx[i,j], grady[i,j])) for j in range(np.shape(gradx)[1])] for i in range(np.shape(gradx)[0])]
        curvx, curvy = signal.fftconvolve(gradmag, -r, mode='valid'), signal.fftconvolve(gradmag, -c, mode='valid') #take second derivative
        curvmag = [[np.linalg.norm((curvx[i,j], curvy[i,j])) for j in range(np.shape(curvx)[1])] for i in range(np.shape(curvx)[0])]
        return curvmag
   
def gaussian_curvature(Z):
    Zy, Zx = np.gradient(Z)                                                     
    Zxy, Zxx = np.gradient(Zx)                                                  
    Zyy, _ = np.gradient(Zy)                                                    
    K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2             
    return K

k_savgol = savgol2d(SE_2015_rest_fild, window_size=7, order=5, derivative='curvature')
k_gaussian = gaussian_curvature(SE_2015_restricted)
##Plot Gaussian curvature
plt.figure()
plt.contourf(lon_western[:], lat_western[:], k_gaussian, norm=SymLogNorm(linthresh=0.03, linscale=0.03, vmin=k_gaussian.min(), vmax=k_gaussian.max()))
plt.show()

##Plot Savgol curvature
plt.figure()
plt.contourf(lon_restricted[3:-3], lat_restricted[3:-3], k_savgol, norm=SymLogNorm(linthresh=0.03, linscale=0.03, vmin=np.asarray(k_savgol).min(), vmax=np.asarray(k_savgol).max()))
plt.show()