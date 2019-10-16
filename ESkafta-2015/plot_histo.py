##  Import crevasse mask and sample stress along it
## EHU 15 Oct 2019 | Based on classes defined by BM

import sys,os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from scipy.interpolate import interp2d,griddata,SmoothBivariateSpline
from plot_slope_stress_EHU import Data #import BM's Data class for consistent I/O

fpath = 'Documents/6. MIT/Skaftar collapse/Crevasse_mask/'
datadic = {}
### inputs - cropped DEMs
datadic['hdr'] = fpath+'diff/skaftar_east/SETSM_WV02_20151010_skaftar_east_medflt.hdr'
datadic['dem'] = fpath+'diff/skaftar_east/SETSM_WV02_20151010_skaftar_east_medflt.bin'
datadic['hdr2'] = fpath+'diff/skaftar_east/SETSM_WV02_20121015_skaftar_east_medflt.hdr'
datadic['dem2'] = fpath+'diff/skaftar_east/SETSM_WV02_20121015_skaftar_east_medflt.bin'
datadic['mask'] = fpath+'diff/skaftar_east/SETSM_WV02_20151010_skafar_east_dem_highpass_mask_smooth.bin'
### inputs - processed by smooth_crevasses.py
datadic['filled_dem'] = fpath+'SETSM_WV02_20151010_skaftar_east_dem_filled.bin'
datadic['filled_diff'] = fpath+'SETSM_WV02_20151010_skaftar_east_dem_filled_diff.bin'
datadic['filled_slope'] = fpath+'SETSM_WV02_20151010_skaftar_east_dem_filled_slope.bin'
datadic['filled_curvature'] = fpath+'SETSM_WV02_20151010_skaftar_east_dem_filled_curvature.bin'

#skafta = {}
#skafta['ul_polstr'] = [1294500.,-2489500.]
#skafta['lr_polstr'] = [1298500.,-2493500.]
skafta = {}

## Read in
data = Data(datadic,skafta)
data.read_data()
data.region_of_interest(ul_row=948, ul_col=2791, lr_row=2851, lr_col=5126)
data.calc_elastic_stress()
data.dem[data.dem < 0] = np.nan #mask nodata areas

## Create a light source for hillshade plotting
ls = LightSource(azdeg=315, altdeg=45)

## Plot elastic stress
def plot_elastic_stress(data,ls):
    # Choose colormap and data range normalization
    cmap = plt.get_cmap('Spectral')

    fig, ax = plt.subplots()

    hatch = np.isnan(data.dem).astype(np.float32)
    hatch[hatch < 0.5] = np.nan

    ax.imshow(ls.hillshade(data.dem,vert_exag=2,dx=data.hdr['spx'],dy=data.hdr['spy']),cmap='gray')
    cf00 = ax.contourf(hatch,hatches=['xxx'],cmap=None,colors='0.4')
    cf0 = ax.contourf(1.e-6*data.filled_stress,cmap=cmap,extend='both',levels=np.linspace(-30,30,30),alpha=0.8)
    cbar = fig.colorbar(cf0,ax=ax,ticks=[-30,-15,0,15,30])
    cbar.ax.set_ylabel('Elastic surface stresses [MPa]',fontsize=12)
    cbar.ax.set_yticklabels(['-30','-15','0','15','30'])
    # cs0 = ax.contour(data.dem,colors='0.4',linewidths=0.5,levels=np.arange(1500,1800,10))
    # ax.clabel(cs0,list(np.arange(1550,1800,50)),inline=1,fontsize=8,fmt='%d')

    ax.set_xlabel('Relative $x$ position [km]',fontsize=12)
    ax.set_ylabel('Relative $y$ position [km]',fontsize=12)

    ax.set_xticklabels(['%1.1f' % x for x in data.hdr['spx']*1.e-3*ax.get_xticks()])
    ax.set_yticklabels(['%1.1f' % (4-x) for x in data.hdr['spy']*1.e-3*ax.get_yticks()])
    plt.show()
    #plt.savefig('figs/stress_shaded.pdf',bbox_inches='tight')

plot_elastic_stress(data, ls)

## Extract and make histogram of stress values in crevassed and non-crevassed areas
intact_stress_sample = 1E-6*np.array(data.filled_stress[data.mask==0]) #express in MPa
frac_stress_sample = 1E-6*np.array(data.filled_stress[data.mask==1])

intact_weights = np.ones_like(intact_stress_sample)/float(len(intact_stress_sample))
frac_weights = np.ones_like(frac_stress_sample)/float(len(frac_stress_sample))
bins = np.linspace(-10, 10, num=40)

plt.figure()
plt.hist((intact_stress_sample, frac_stress_sample), bins=bins, weights=(intact_weights, frac_weights), label=('intact', 'fractured'))
plt.legend(loc='best')
plt.axes().set_xlabel('Surface elastic stress [MPa]', fontsize=18)
plt.axes().set_ylabel('Density in sample', fontsize=18)
plt.axes().tick_params(which='both', labelsize=14)
plt.show()