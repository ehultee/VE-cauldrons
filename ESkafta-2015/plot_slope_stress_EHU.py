#!/usr/bin/env python3

## 16 Oct 2019 | EHU modifying BM's original code

import sys,os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
#from cpt_tools import cpt2python

def main(datadic,skafta):
	data = Data(datadic,skafta)
	data.read_data()
	data.region_of_interest(ul_row=948, ul_col=2791, lr_row=2851, lr_col=5126)
	data.calc_elastic_stress()
	data.dem[data.dem < 0] = np.nan

   # Create a light source
	ls = LightSource(azdeg=315, altdeg=45)
#
   #plot_dem_only(data,ls)
# 	plot_filled_dem(data,ls)
    # plot_slope(data,ls)
    # plot_curvature(data,ls)
    # plot_strain(data,ls)
	plot_elastic_stress(data,ls)
#     plot_strain_energy_density(data,ls)

# def plot_strain_energy_density(data,ls):
#     # Choose colormap and data range normalization
#     cmap = plt.get_cmap('magma_r')
# 
#     fig, ax = plt.subplots()
# 
#     # data.gridx,data.gridy,
#     hatch = np.isnan(data.dem).astype(np.float32)
#     hatch[hatch < 0.5] = np.nan
# 
#     ax.imshow(ls.hillshade(data.dem,vert_exag=2,dx=data.hdr['spx'],dy=data.hdr['spy']),cmap='gray')
#     cf00 = ax.contourf(hatch,hatches=['xxx'],cmap=None,colors='0.4')
#     cf0 = ax.contourf(data.filled_strainenergy,cmap=cmap,extend='both',levels=np.linspace(1.e3,1.e6,30),alpha=0.8)
#     cbar = fig.colorbar(cf0,ax=ax,ticks=[1e3,100e3,250e3,500e3,750e3,1000e3])
#     cbar.ax.set_ylabel(r'Surface strain energy density [kJ/m$^{3}$]',fontsize=12)
#     cbar.ax.set_yticklabels([1,100,250,500,750,1000])
# 
#     ax.set_xlabel('Relative $x$ position [km]',fontsize=12)
#     ax.set_ylabel('Relative $y$ position [km]',fontsize=12)
# 
#     ax.set_xticklabels(['%1.1f' % x for x in data.hdr['spx']*1.e-3*ax.get_xticks()])
#     ax.set_yticklabels(['%1.1f' % (4-x) for x in data.hdr['spy']*1.e-3*ax.get_yticks()])
#     plt.savefig('figs/strain_energy_shaded.pdf',bbox_inches='tight')

def plot_elastic_stress(data, ls, saturation_stress=10):
	"""Plot maximum stress attainable in a purely elastic deformation.
	Optional argument: 
		saturation_stress: default 30 MPa, can be increased or decreased to highlight qualitative features"""
    # Choose colormap and data range normalization
	cmap = plt.get_cmap('Spectral')
	
	fig, ax = plt.subplots()

	hatch = np.isnan(data.dem).astype(np.float32)
	hatch[hatch < 0.5] = np.nan

	ax.imshow(ls.hillshade(data.dem,vert_exag=2,dx=data.hdr['spx'],dy=data.hdr['spy']),cmap='gray')
	cf00 = ax.contourf(hatch,hatches=['xxx'],cmap=None,colors='0.4')
	cf0 = ax.contourf(1.e-6*data.filled_stress,cmap=cmap,extend='both',levels=np.linspace(-1*saturation_stress,saturation_stress,30),alpha=0.8)
	cbar = fig.colorbar(cf0,ax=ax,ticks=[-1*saturation_stress,-0.5*saturation_stress,0,0.5*saturation_stress,saturation_stress])
	cbar.ax.set_ylabel('Elastic surface stresses [MPa]',fontsize=12)
#     cbar.ax.set_yticklabels([format(),format(),format(),format(),format()])
    # cs0 = ax.contour(data.dem,colors='0.4',linewidths=0.5,levels=np.arange(1500,1800,10))
    # ax.clabel(cs0,list(np.arange(1550,1800,50)),inline=1,fontsize=8,fmt='%d')

	ax.set_xlabel('Relative $x$ position [km]',fontsize=12)
	ax.set_ylabel('Relative $y$ position [km]',fontsize=12)

	ax.set_xticklabels(['%1.1f' % x for x in data.hdr['spx']*1.e-3*ax.get_xticks()])
	ax.set_yticklabels(['%1.1f' % (4-x) for x in data.hdr['spy']*1.e-3*ax.get_yticks()])
	plt.savefig('/Users/ehultee/Documents/6. MIT/Skaftar collapse/Crevasse_mask/figs/stress_shaded-{}MPa.pdf'.format(saturation_stress),bbox_inches='tight')

def plot_strain(data,ls):
    # Choose colormap and data range normalization
    cmap = plt.get_cmap('Spectral')

    fig, ax = plt.subplots()

    # data.gridx,data.gridy,
    hatch = np.isnan(data.dem).astype(np.float32)
    hatch[hatch < 0.5] = np.nan

    ax.imshow(ls.hillshade(data.dem,vert_exag=2,dx=data.hdr['spx'],dy=data.hdr['spy']),cmap='gray')
    cf00 = ax.contourf(hatch,hatches=['xxx'],cmap=None,colors='0.4')
    cf0 = ax.contourf(data.filled_strain,cmap=cmap,extend='both',levels=np.linspace(-2.e-2,2.e-2,30),alpha=0.8)
    cbar = fig.colorbar(cf0,ax=ax,ticks=[-2.e-2,-1e-2,0,1e-2,2.e-2])
    cbar.ax.set_ylabel(r'Surface strain [$\times10^{-3}$]',fontsize=12)
    cbar.ax.set_yticklabels(['20','10','0','10','20'])
    # cs0 = ax.contour(data.dem,colors='0.4',linewidths=0.5,levels=np.arange(1500,1800,10))
    # ax.clabel(cs0,list(np.arange(1550,1800,50)),inline=1,fontsize=8,fmt='%d')

    ax.set_xlabel('Relative $x$ position [km]',fontsize=12)
    ax.set_ylabel('Relative $y$ position [km]',fontsize=12)

    ax.set_xticklabels(['%1.1f' % x for x in data.hdr['spx']*1.e-3*ax.get_xticks()])
    ax.set_yticklabels(['%1.1f' % (4-x) for x in data.hdr['spy']*1.e-3*ax.get_yticks()])
    plt.savefig('/Users/ehultee/Documents/6. MIT/Skaftar collapse/Crevasse_mask/figs/strain_shaded.pdf',bbox_inches='tight')

def plot_curvature(data,ls):
    # Choose colormap and data range normalization
    cmap = plt.get_cmap('Spectral')
    norm = Normalize(vmin=0,vmax=1600)

    fig, ax = plt.subplots()

    # data.gridx,data.gridy,
    hatch = np.isnan(data.dem).astype(np.float32)
    hatch[hatch < 0.5] = np.nan

    ax.imshow(ls.hillshade(data.dem,vert_exag=2,dx=data.hdr['spx'],dy=data.hdr['spy']),cmap='gray')
    cf00 = ax.contourf(hatch,hatches=['xxx'],cmap=None,colors='0.4')
    cf0 = ax.contourf(data.filled_curvature,cmap=cmap,extend='both',levels=np.linspace(-1.e-4,1.e-4,30),alpha=0.8)
    cbar = fig.colorbar(cf0,ax=ax,ticks=[-1.e-4,-0.5e-4,0,0.5e-4,1.e-4])
    cbar.ax.set_ylabel('Mean surface curvature [m$^{-1}$]',fontsize=12)
    cbar.ax.set_yticklabels(['-10$^{-4}$','-10$^{-4}$/2','0','10$^{-4}$/2','10$^{-4}$'])
    # cs0 = ax.contour(data.dem,colors='0.4',linewidths=0.5,levels=np.arange(1500,1800,10))
    # ax.clabel(cs0,list(np.arange(1550,1800,50)),inline=1,fontsize=8,fmt='%d')

    ax.set_xlabel('Relative $x$ position [km]',fontsize=12)
    ax.set_ylabel('Relative $y$ position [km]',fontsize=12)

    ax.set_xticklabels(['%1.1f' % x for x in data.hdr['spx']*1.e-3*ax.get_xticks()])
    ax.set_yticklabels(['%1.1f' % (4-x) for x in data.hdr['spy']*1.e-3*ax.get_yticks()])
    plt.savefig('/Users/ehultee/Documents/6. MIT/Skaftar collapse/Crevasse_mask/figs/curvature_shaded.pdf',bbox_inches='tight')

def plot_slope(data,ls):
    # Choose colormap and data range normalization
    cmap = plt.get_cmap('magma_r')
    norm = Normalize(vmin=0,vmax=1600)

    fig, ax = plt.subplots()

    # data.gridx,data.gridy,
    hatch = np.isnan(data.dem).astype(np.float32)
    hatch[hatch < 0.5] = np.nan

    ax.imshow(ls.hillshade(data.dem,vert_exag=2,dx=data.hdr['spx'],dy=data.hdr['spy']),cmap='gray')
    cf00 = ax.contourf(hatch,hatches=['xxx'],cmap=None,colors='0.4')
    cf0 = ax.contourf(data.filled_slope,cmap=cmap,extend='max',levels=np.linspace(0,0.15,30),alpha=0.8)
    cbar = fig.colorbar(cf0,ax=ax,ticks=[0,0.03,0.06,0.09,0.12,0.15])
    cbar.ax.set_ylabel('Surface slope [-]',fontsize=12)
    # cs0 = ax.contour(data.dem,colors='0.4',linewidths=0.5,levels=np.arange(1500,1800,10))
    # ax.clabel(cs0,list(np.arange(1550,1800,50)),inline=1,fontsize=8,fmt='%d')

    ax.set_xlabel('Relative $x$ position [km]',fontsize=12)
    ax.set_ylabel('Relative $y$ position [km]',fontsize=12)

    ax.set_xticklabels(['%1.1f' % x for x in data.hdr['spx']*1.e-3*ax.get_xticks()])
    ax.set_yticklabels(['%1.1f' % (4-x) for x in data.hdr['spy']*1.e-3*ax.get_yticks()])
    plt.savefig('/Users/ehultee/Documents/6. MIT/Skaftar collapse/Crevasse_mask/figs/slope_shaded.pdf',bbox_inches='tight')

def plot_filled_dem(data,ls):
    # Choose colormap and data range normalization
    # cmap = plt.get_cmap('cividis_r')
    # norm = Normalize(1550, 1700)

    # rgb = ls.shade_rgb(cmap(norm(data.dem)), data.filled_dem, blend_mode='overlay', fraction=0.6)

    cmap = plt.get_cmap('viridis_r')

    fig, ax = plt.subplots()

    hatch = np.isnan(data.dem).astype(np.float32)
    hatch[hatch < 0.5] = np.nan

    ax.imshow(ls.hillshade(data.dem,vert_exag=2,dx=data.hdr['spx'],dy=data.hdr['spy']),cmap='gray')
    cf00 = ax.contourf(hatch,hatches=['xxx'],cmap=None,colors='0.4')
    cf0 = ax.contourf(data.filled_dem,cmap=cmap,extend='both',levels=np.linspace(1550,1750,30),alpha=0.8)
    cbar = fig.colorbar(cf0,ax=ax,ticks=[1550,1600,1650,1700,1750])
    cbar.ax.set_ylabel('Surface elevation [m AMSL]',fontsize=12)
    # cs0 = ax.contour(data.dem,colors='0.4',linewidths=0.5,levels=np.arange(1500,1800,10))
    # ax.clabel(cs0,list(np.arange(1550,1800,50)),inline=1,fontsize=8,fmt='%d')

    ax.set_xlabel('Relative $x$ position [km]',fontsize=12)
    ax.set_ylabel('Relative $y$ position [km]',fontsize=12)

    ax.set_xticklabels(['%1.1f' % x for x in data.hdr['spx']*1.e-3*ax.get_xticks()])
    ax.set_yticklabels(['%1.1f' % (4-x) for x in data.hdr['spy']*1.e-3*ax.get_yticks()])
    plt.savefig('/Users/ehultee/Documents/6. MIT/Skaftar collapse/Crevasse_mask/figs/dem_filled_shaded.pdf',bbox_inches='tight')

def plot_dem_only(data,ls):
    # Choose colormap and data range normalization
    cmap = plt.get_cmap('cividis_r')
    norm = Normalize(1550, 1700)

    rgb = ls.shade_rgb(cmap(norm(data.dem)), data.dem, blend_mode='overlay', fraction=0.6)

    fig, ax = plt.subplots()
    ax.imshow(rgb)

    # Use a proxy artist for the colorbar...
    im = ax.imshow(data.dem, cmap=cmap)
    im.remove()
    fig.colorbar(im)

    plt.savefig('figs/dem_only_shaded.pdf',bbox_inches='tight')

class Data():
    def __init__(data,datafiles,skafta):
        data.files = datafiles
        data.skafta = skafta
        data.hdr = data.get_header('hdr')

    def calc_elastic_stress(data):
        data.youngs = 1.e9 ### Youngs modulus in Pa
        data.poissons = 0.3 ### Poissons ratio
        data.thickness_m = 300.

        data.sign_compression = 1.  ### sign convention for compressive stress/strain (-1 for positive tension)
        data.filled_strain = data.sign_compression * 0.5 * data.thickness_m * data.filled_curvature
        data.filled_stress = data.youngs * data.filled_strain / (1.-data.poissons**2)
        data.filled_strainenergy = data.filled_strain*data.filled_stress

    def region_of_interest(data, ul_row=None, ul_col=None, lr_row=None, lr_col=None):
        """Get row and column of DEM to focus on Skafta"""
        if ul_row is None:
            ### get row and column for Skafta
            ul_row = np.int(np.abs(data.hdr['ulx'] - data.skafta['ul_polstr'][0]) / data.hdr['spx'])
            ul_col = np.int(np.abs(data.hdr['uly'] - data.skafta['ul_polstr'][1]) / data.hdr['spy'])
            lr_row = np.int(np.abs(data.hdr['ulx'] - data.skafta['lr_polstr'][0]) / data.hdr['spx'])
            lr_col = np.int(np.abs(data.hdr['uly'] - data.skafta['lr_polstr'][1]) / data.hdr['spy'])
        
        data.dem = data.dem[ul_row:lr_row,ul_col:lr_col]
        data.filled_dem = data.filled_dem[ul_row:lr_row,ul_col:lr_col]
        data.filled_diff = data.filled_diff[ul_row:lr_row,ul_col:lr_col]
        data.filled_slope = data.filled_slope[ul_row:lr_row,ul_col:lr_col]
        data.filled_curvature = data.filled_curvature[ul_row:lr_row,ul_col:lr_col]

        data.cols = data.dem.shape[1]
        data.rows = data.dem.shape[0]
        #data.gridx = data.skafta['ul_polstr'][0] + np.arange(data.cols)*data.hdr['spx']
        #data.gridy = data.skafta['ul_polstr'][1] - np.arange(data.rows)*np.abs(data.hdr['spy'])

    def read_data(data):
        with open(data.files['dem'],'r') as fid:
            data.dem = np.fromfile(fid,dtype=np.float32).reshape(-1,data.hdr['cols'])
            if data.dem.shape[0] != data.hdr['rows']:
                raise ValueError('dem not the right size according to header')
        with open(data.files['filled_dem'],'r') as fid:
            data.filled_dem = np.fromfile(fid,dtype=np.float32).reshape(-1,data.hdr['cols'])
            if data.filled_dem.shape[0] != data.hdr['rows']:
                raise ValueError('mask not the right size according to header')
        with open(data.files['filled_diff'],'r') as fid:
            data.filled_diff = np.fromfile(fid,dtype=np.float32).reshape(-1,data.hdr['cols'])
            if data.filled_diff.shape[0] != data.hdr['rows']:
                raise ValueError('mask not the right size according to header')
        with open(data.files['filled_slope'],'r') as fid:
            data.filled_slope = np.fromfile(fid,dtype=np.float32).reshape(-1,data.hdr['cols'])
            if data.filled_slope.shape[0] != data.hdr['rows']:
                raise ValueError('mask not the right size according to header')
        with open(data.files['filled_curvature'],'r') as fid:
            data.filled_curvature = np.fromfile(fid,dtype=np.float32).reshape(-1,data.hdr['cols'])
            if data.filled_curvature.shape[0] != data.hdr['rows']:
                raise ValueError('mask not the right size according to header')
        with open(data.files['mask'], 'r') as fid:
            data.mask = np.fromfile(fid, dtype=np.float32).reshape(-1,data.hdr['cols'])
            if data.mask.shape[0] != data.hdr['rows']:
                raise ValueError('mask not the right size according to header')


    def get_header(data,hdrstr='hdr'):
        hdr = {}
        with open(data.files[hdrstr],'r') as fid:
            reads = fid.readlines()
        for read in reads:
            if read.startswith('samples'):
                hdr['cols'] = int(read.split('=')[1])
            elif read.startswith('lines'):
                hdr['rows'] = int(read.split('=')[1])
            elif read.startswith('map info') or read.startswith('map_info'):
                r = read.split('=')[1].split(',')
                hdr['ulx'] = np.float(r[3])
                hdr['uly'] = np.float(r[4])
                hdr['spx'] = np.abs(np.float(r[5]))
                hdr['spy'] = np.abs(np.float(r[6]))
                hdr['lrx'] = hdr['ulx'] + (hdr['cols']-1)*hdr['spx']
                hdr['lry'] = hdr['uly'] - (hdr['rows']-1)*np.abs(hdr['spy'])
        return hdr

if __name__=='__main__':
	fpath = '/Users/ehultee/Documents/6. MIT/Skaftar collapse/Crevasse_mask/'
	datadic = {}
	### inputs - cropped DEMs
	datadic['hdr'] = fpath+'diff/skaftar_east/SETSM_WV02_20151010_skaftar_east_medflt.hdr'
	datadic['dem'] = fpath+'diff/skaftar_east/SETSM_WV02_20151010_skaftar_east_medflt.bin'
	datadic['mask'] = fpath+'diff/skaftar_east/SETSM_WV02_20151010_skafar_east_dem_highpass_mask_smooth.bin'
	### inputs - processed by smooth_crevasses.py
	datadic['filled_dem'] = fpath+'SETSM_WV02_20151010_skaftar_east_dem_filled.bin'
	datadic['filled_diff'] = fpath+'SETSM_WV02_20151010_skaftar_east_dem_filled_diff.bin'
	datadic['filled_slope'] = fpath+'SETSM_WV02_20151010_skaftar_east_dem_filled_slope.bin'
	datadic['filled_curvature'] = fpath+'SETSM_WV02_20151010_skaftar_east_dem_filled_curvature.bin'

	skafta = {}
#    skafta['ul_polstr'] = [1294500.,-2489500.]
#    skafta['lr_polstr'] = [1298500.,-2493500.]
#
main(datadic,skafta)
