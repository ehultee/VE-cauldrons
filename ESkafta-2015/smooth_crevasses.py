#!/usr/bin/env python3
'''
Read in ArcticDEM, chop out area of interest, interpolate over crevasses, and stitch into larger area
'''
import sys,os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d,griddata,SmoothBivariateSpline

datadic = {}
### inputs
datadic['hdr'] = '../diff/skaftar_east/SETSM_WV02_20151010_skaftar_east_medflt.hdr'
datadic['dem'] = '../diff/skaftar_east/SETSM_WV02_20151010_skaftar_east_medflt.bin'
datadic['hdr2'] = '../diff/skaftar_east/SETSM_WV02_20121015_skaftar_east_medflt.hdr'
datadic['dem2'] = '../diff/skaftar_east/SETSM_WV02_20121015_skaftar_east_medflt.bin'
datadic['mask'] = '../diff/skaftar_east/SETSM_WV02_20151010_skafar_east_dem_highpass_mask_smooth.bin'
# ### outputs
datadic['output_dem'] = '../SETSM_WV02_20151010_skaftar_east_dem_filled.bin'
datadic['output_diff'] = '../SETSM_WV02_20151010_skaftar_east_dem_filled_diff.bin'
datadic['output_slope'] = '../SETSM_WV02_20151010_skaftar_east_dem_filled_slope.bin'
datadic['output_dem2'] = '../SETSM_WV02_20121015_skaftar_east_dem_filled.bin'
datadic['output_diff2'] = '../SETSM_WV02_20121015_skaftar_east_dem_filled_diff.bin'
datadic['output_slope2'] = '../SETSM_WV02_20121015_skaftar_east_dem_filled_slope.bin'
datadic['output_laplacian'] = '../SETSM_WV02_20151010_skaftar_east_dem_filled_laplacian.bin'
datadic['output_curvature'] = '../SETSM_WV02_20151010_skaftar_east_dem_filled_curvature.bin'
datadic['output_ddx2'] = '../SETSM_WV02_20151010_skaftar_east_dem_filled_ddx2.bin'
datadic['output_ddy2'] = '../SETSM_WV02_20151010_skaftar_east_dem_filled_ddy2.bin'
datadic['output_ddxdy'] = '../SETSM_WV02_20151010_skaftar_east_dem_filled_ddxdy.bin'
datadic['output_skafta_xyz'] = 'SETSM_WV02_20151010_nocrevasse_skafta.xyz'

def main(datafiles):
    skafta = {}
    skafta['ul_polstr'] = [1294500.,-2489500.]
    skafta['lr_polstr'] = [1298500.,-2493500.]
    skafta['mask_val'] = 1

    data = Data(datadic)
    data.read_data()

    ### convert to logical
    if skafta['mask_val'] == 1:
        data.mask_logic = data.mask < 0.5
    else:
        data.mask_logic = data.mask > 0.5
    ### mask null values in data
    null_mask = data.dem > 0.
    data.mask_logic *= null_mask

    ### get row and column for Skafta
    #ul_row = np.int(np.abs(data.hdr['ulx'] - skafta['ul_polstr'][0]) / data.hdr['spx'])
    #ul_col = np.int(np.abs(data.hdr['uly'] - skafta['ul_polstr'][1]) / data.hdr['spy'])
    #lr_row = np.int(np.abs(data.hdr['ulx'] - skafta['lr_polstr'][0]) / data.hdr['spx'])
    #lr_col = np.int(np.abs(data.hdr['uly'] - skafta['lr_polstr'][1]) / data.hdr['spy'])
    ul_row = 948
    ul_col=2791
    lr_row = 2851
    lr_col = 5126

    ### cut out Skafta
    data.dem_skafta = data.dem[ul_row:lr_row,ul_col:lr_col]
    data.mask_logic_skafta = data.mask_logic[ul_row:lr_row,ul_col:lr_col]
    #skafta_shape = data.dem_skafta.shape

    data.skafta_gridx = np.empty_like(data.dem_skafta)
    data.skafta_gridy = np.empty_like(data.dem_skafta)
    for i in range(data.skafta_gridx.shape[0]):
        data.skafta_gridx[i,:] = skafta['ul_polstr'][0] + np.arange(data.dem_skafta.shape[1])*data.hdr['spx']
    for i in range(data.skafta_gridx.shape[1]):
        data.skafta_gridy[:,i] = skafta['ul_polstr'][1] - np.arange(data.dem_skafta.shape[0])*data.hdr['spy']

    data.dem_skafta_mask = data.dem_skafta.flatten()[data.mask_logic_skafta.flatten()]
    data.skafta_gridx_mask = data.skafta_gridx.flatten()[data.mask_logic_skafta.flatten()]
    data.skafta_gridy_mask = data.skafta_gridy.flatten()[data.mask_logic_skafta.flatten()]
    #
    # ## Optional: write xyz file for use with GMT surface
    if False:
        decfac_xyz = 50
        xyzout = np.empty((len(data.dem_skafta_mask[::decfac_xyz]),3),dtype=np.float32)
        xyzout[:,0] = data.skafta_gridx_mask[::decfac_xyz]
        xyzout[:,1] = data.skafta_gridy_mask[::decfac_xyz]
        xyzout[:,2] = data.dem_skafta_mask[::decfac_xyz]
        np.savetxt(datadic['output_skafta_xyz'],xyzout,fmt='%10.8f',delimiter='   ')

    ### first stab: unsat due to artifacts using both linear and cubic...likely due to piecewise nature of gridding
    # data.dem_skafta_filled = griddata((data.skafta_gridx_mask,data.skafta_gridy_mask),data.dem_skafta_mask,
    #             (data.skafta_gridx,data.skafta_gridy),method='cubic')
    # data.dem[ul_row:lr_row,ul_col:lr_col] = data.dem_skafta_filled
    # #data.dem[~null_mask] = -9999.

    ### second stab using global spline
    ### need to decimate

    print('defining spline function')
    decfac = 20
    sporder = 5

    data.dem_skafta_dec = data.dem_skafta[::-decfac,::decfac].flatten()
    data.mask_skafta_dec = data.mask_logic_skafta[::-decfac,::decfac].flatten().astype(np.float32) + np.finfo(np.float64).eps
    data.skafta_gridx_dec = data.skafta_gridx[::-decfac,::decfac].flatten()
    data.skafta_gridy_dec = data.skafta_gridy[::-decfac,::decfac].flatten()

    data.spline_fun = SmoothBivariateSpline(x=data.skafta_gridx_dec,y=data.skafta_gridy_dec,
                        z=data.dem_skafta_dec,w=data.mask_skafta_dec,kx=sporder,ky=sporder,s=750)
    #print(data.spline_fun)

    print('done defining spline function...interpolating')
    data.dem_skafta_filled = data.spline_fun.ev(data.skafta_gridx,data.skafta_gridy)
    data.dem_diff = data.dem_skafta - data.dem_skafta_filled

    data.dem_skafta_filled_ddx = data.spline_fun.ev(data.skafta_gridx,data.skafta_gridy,dx=1) / data.hdr['spx']
    data.dem_skafta_filled_ddy = data.spline_fun.ev(data.skafta_gridx,data.skafta_gridy,dy=1) / data.hdr['spy']
    data.dem_skafta_filled_ddx2 = data.spline_fun.ev(data.skafta_gridx,data.skafta_gridy,dx=2) / data.hdr['spx']**2
    data.dem_skafta_filled_ddy2 = data.spline_fun.ev(data.skafta_gridx,data.skafta_gridy,dy=2) / data.hdr['spy']**2
    data.dem_skafta_filled_ddxdy = data.spline_fun.ev(data.skafta_gridx,data.skafta_gridy,dx=1,dy=1) / (data.hdr['spx']*data.hdr['spy'])

    data.dem_skafta_filled_slope = np.sqrt(data.dem_skafta_filled_ddx**2 + data.dem_skafta_filled_ddy**2)

    data.dem_skafta_filled_laplacian = data.dem_skafta_filled_ddx2 + data.dem_skafta_filled_ddy2
    ### calculate mean curvature...see "Surfaces in 3D space" at https://en.wikipedia.org/wiki/Mean_curvature
    data.dem_skafta_filled_curvature = 0.5*((1. + data.dem_skafta_filled_ddx**2)*data.dem_skafta_filled_ddy2 -
                                        2.*data.dem_skafta_filled_ddx*data.dem_skafta_filled_ddy*data.dem_skafta_filled_ddxdy +
                                        (1. + data.dem_skafta_filled_ddy**2)*data.dem_skafta_filled_ddx2) / (
                                        1. + data.dem_skafta_filled_ddx**2 + data.dem_skafta_filled_ddy**2)**1.5

    data.dem[ul_row:lr_row,ul_col:lr_col] = data.dem_skafta_filled
    data.slope = 0.*data.dem
    data.slope[ul_row:lr_row,ul_col:lr_col] = data.dem_skafta_filled_slope
    data.laplacian = 0.*data.dem
    data.laplacian[ul_row:lr_row,ul_col:lr_col] = data.dem_skafta_filled_laplacian
    data.curvature = 0.*data.dem
    data.curvature[ul_row:lr_row,ul_col:lr_col] = data.dem_skafta_filled_curvature
    data.ddx2 = 0.*data.dem
    data.ddx2[ul_row:lr_row,ul_col:lr_col] = data.dem_skafta_filled_ddx2
    data.ddy2 = 0.*data.dem
    data.ddy2[ul_row:lr_row,ul_col:lr_col] = data.dem_skafta_filled_ddy2
    data.ddxdy = 0.*data.dem
    data.ddxdy[ul_row:lr_row,ul_col:lr_col] = data.dem_skafta_filled_ddxdy


    print('writing')
    with open(datadic['output_dem'],'w') as fid:
        data.dem.flatten().astype(np.float32).tofile(fid)
    with open(datadic['output_slope'],'w') as fid:
        data.slope.flatten().astype(np.float32).tofile(fid)
    with open(datadic['output_laplacian'],'w') as fid:
        data.laplacian.flatten().astype(np.float32).tofile(fid)
    with open(datadic['output_curvature'],'w') as fid:
        data.curvature.flatten().astype(np.float32).tofile(fid)
    with open(datadic['output_ddx2'],'w') as fid:
    	data.ddx2.flatten().astype(np.float32).tofile(fid)
    with open(datadic['output_ddy2'],'w') as fid:
    	data.ddy2.flatten().astype(np.float32).tofile(fid)
    with open(datadic['output_ddxdy'],'w') as fid:
    	data.ddxdy.flatten().astype(np.float32).tofile(fid)

    if True:
        data.dem[:,:] = 0.
        data.dem[ul_row:lr_row,ul_col:lr_col] = data.dem_diff
        with open(datadic['output_diff'],'w') as fid:
            data.dem.flatten().astype(np.float32).tofile(fid)


class Data():
    def __init__(data,datafiles):
        data.files = datafiles
        data.hdr = data.get_header('hdr')
        print(data.files.keys())
        data.hdr_prior = data.get_header('hdr2')

    def read_data(data):
        with open(data.files['dem'],'r') as fid:
            data.dem = np.fromfile(fid,dtype=np.float32).reshape(-1,data.hdr['cols'])
            if data.dem.shape[0] != data.hdr['rows']:
                raise ValueError('dem not the right size according to header')
        with open(data.files['mask'],'r') as fid:
            data.mask = np.fromfile(fid,dtype=np.float32).reshape(-1,data.hdr['cols'])
            if data.mask.shape[0] != data.hdr['rows']:
                raise ValueError('mask not the right size according to header')

        with open(data.files['dem2'],'r') as fid:
            data.dem_prior = np.fromfile(fid,dtype=np.float32).reshape(-1,data.hdr_prior['cols'])
            if data.dem_prior.shape[0] != data.hdr_prior['rows']:
                raise ValueError('dem not the right size according to header')


    def get_header(data,hdrstr):
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
    main(datadic)
