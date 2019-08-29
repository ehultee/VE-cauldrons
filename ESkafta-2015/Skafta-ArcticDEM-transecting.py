# Reading in ArcticDEM, sampling transect across Skafta Cauldron
# 4 Dec 2018 EHU
# Edit 21 Feb 2019 - plot analytical elastic/viscoelastic
# Edit 16 July - move functions to helper module

import numpy as np
import scipy.misc as scp
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from osgeo import gdal
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import math
from cauldron_funcs import *


## Read in ArcticDEM surface
skafta_region_path = 'Documents/6. MIT/Skaftar collapse/data/arcticDEM/'
nc_20121015_path = skafta_region_path + 'subset_nc/SETSM_WV02_20121015_skaftar_east_ll.nc'
nc_20151010_path = skafta_region_path + 'subset_nc/SETSM_WV02_20151010_skaftar_east_ll.nc'

lon_2012, lat_2012, se_2012 = read_ArcticDEM_nc(nc_20121015_path)
SE_2012 = np.ma.masked_where(se_2012==0, se_2012)
lon_2015, lat_2015, se_2015 = read_ArcticDEM_nc(nc_20151010_path)
SE_2015 = np.ma.masked_where(se_2015==0, se_2015)

## Interpolating surface elevation and sampling transect
sefunc_2012 = interpolate.interp2d(lon_2012, lat_2012, SE_2012)
sefunc_2015 = interpolate.interp2d(lon_2015, lat_2015, SE_2015)

#npoints = 1000
#endpoints = [(-17.542113802658239, 64.488141277357315),
# (-17.48586677277758, 64.486397775690023)] #coordinates at either side of the cauldron, selected by inspection with ginput.  
#lonvals = np.linspace(endpoints[0][0], endpoints[1][0], npoints)
#latvals = np.linspace(endpoints[0][1], endpoints[1][1], npoints)
#sevals_2012 = np.asarray([sefunc_2012(lonvals[i], latvals[i]) for i in range(npoints)]).squeeze()
#sevals_2015 = np.asarray([sefunc_2015(lonvals[i], latvals[i]) for i in range(npoints)]).squeeze()

## Prepare transect for plotting, with x-axis of distance along transect in m
def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

#transect_length = haversine(endpoints[0][::-1], endpoints[1][::-1])
#xaxis = np.linspace(0, transect_length, num=npoints)

def sample_transect(endpts, DEM_surface1, DEM_surface2=None, cauldron_name='Eastern_Skafta', npoints=1000, elastic=True, viscoelastic=True, days_simulated = 5, stresses=False):
    """ Function to standardize transecting procedure.  Sets up a cauldron with appropriate radius and computes analytical profiles.
    Arguments:
        endpts = (lat, lon) of the two endpoints of the transect
        DEM_surface1 = a 2D interpolated function of the initial observed surface to plot
        DEM_surface2 = another (optional) 2D interpolated function of the observed surface, possibly from a later time.  Default None will use only 1 surface
    Default settings:
        cauldron_name = name (string) of the Cauldron instance we set up.  Default is 'Eastern_Skafta'.
        npoints = how many points to sample along transect.  Default 1000
        elastic = whether to calculate elastic profile.  Default True
        viscoelastic = whether to calculate viscoelastic profile/stresses.  Default True
        days_simulated = time period over which to simulate viscoelastic collapse. Default 5 (for Eastern Skafta)
        stresses = whether to calculate elastic and VE stresses.  Default False
    Returns dictionary of profiles and stresses, as specified in arguments.
    """
    out_dict = {}
    out_dict['name'] = cauldron_name
    lonvals = np.linspace(endpts[0][0], endpts[1][0], npoints)
    latvals = np.linspace(endpts[0][1], endpts[1][1], npoints)
    surfvals_1 = np.asarray([DEM_surface1(lonvals[i], latvals[i]) for i in range(npoints)]).squeeze() # for Eastern Skafta, use 2012 vals here
    out_dict['initial_surface_obs'] = surfvals_1
    if DEM_surface2 is not None:
        surfvals_2 = np.asarray([DEM_surface2(lonvals[i], latvals[i]) for i in range(npoints)]).squeeze()
        out_dict['second_surface_obs'] = surfvals_2
        
    transect_length = haversine(endpts[0][::-1], endpts[1][::-1])
    out_dict['xaxis'] = np.linspace(0, transect_length, num=npoints) #needed for plotting
    x_cylcoords = np.linspace(-0.5*transect_length, 0.5*transect_length, num=npoints)
    initial_surf_val = np.mean((surfvals_1[0], surfvals_1[-1])) #surface elevation at edges before loading
    initial_surf = interpolate.interp1d(x_cylcoords, surfvals_1, kind='quadratic')
    cldrn = Cauldron(name=cauldron_name, initial_surface = initial_surf, radius = 0.5*transect_length)
    cldrn.set_viscoelastic_bendingmod()
    
    out_dict['Cauldron_instance'] = cldrn #return the Cauldron instance in case further manipulations wanted

    if elastic:
        out_dict['elastic_profile'] = [cldrn.LL_profile(x) for x in x_cylcoords]
    if viscoelastic:
        nseconds = days_simulated*24*60*60 #number of seconds in days_simulated
        times = np.arange(0, nseconds, step=20000)
        out_dict['VE_times'] = times
        out_dict['VE_profiles'] = [[cldrn.viscoelastic_profile(x, t0) for x in x_cylcoords] for t0 in times]
    if stresses:
        if elastic:
            out_dict['elastic_stress'] = [cldrn.elastic_stress(x, config='radial_plate') for x in x_cylcoords]
        if viscoelastic:
            out_dict['max_VE_stress'] = [cldrn.viscoelastic_stress(x, times[4]) for x in x_cylcoords]

    return out_dict

def plot_elastic_transect(in_dict, colormap=cm.get_cmap('winter_r')):
    """Read in quantities and plot elastic profile from a transect dictionary
    """
    xaxis = in_dict['xaxis']
    sevals_1 = in_dict['initial_surface_obs']
    try:
        sevals_2 = in_dict['second_surface_obs']
    except KeyError:
        print 'No secondary surface observations saved on transect {}. Setting identical to first surface for plotting.'.format(in_dict['name'])
        sevals_2 = sevals_1
    elastic_profile = in_dict['elastic_profile']
    transect_length = max(xaxis)
    
    elas_color = colormap(np.linspace(0.1, 0.9, num=len(times)+1))[0]
    
    fig = plt.figure('Elastic profile, {}'.format(in_dict['name']), figsize=(7, 3))
    plt.plot(xaxis, sevals_1, color='k', ls='-.') #, label='15 Oct 2012'
    plt.plot(xaxis, sevals_2, color='k', ls='-', label='Obs.') #, label='10 Oct 2015'
    plt.plot(xaxis, elastic_profile, color=elas_color, lw=2, label='Elastic plate')
    plt.fill_between(xaxis, sevals1, sevals2, color='Gainsboro', hatch='/', edgecolor='DimGray', linewidth=0, alpha=0.7)
    plt.fill_between(xaxis, sevals2, (plt.axes().get_ylim()[0]), color='Azure')
    plt.legend(loc='lower left')
    plt.axes().set_aspect(5)
    plt.axes().set_xlim(0, transect_length)
    plt.axes().set_yticks([1550, 1600, 1650, 1700])
    #plt.axes().set_yticklabels(['1550', '1600', '1650', '1700'], fontsize=14)
    plt.axes().tick_params(which='both', labelsize=14)
    #plt.axes().set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)
    plt.axes().set_xlabel('Along-transect distance [m]', fontsize=16)
    plt.axes().set_ylabel('Surface elevation [m a.s.l.]', fontsize=16)
    #plt.title('Eastern Skafta cauldron transect: observed, ideal elastic, ideal viscoelastic. E={:.1E}'.format(ESkafta.youngmod), fontsize=18)
    plt.show()
    
    return fig #return the figure instance so it can be modified
    

def plot_VE_transect(in_dict, colormap=cm.get_cmap('winter_r'), make_legend=False, ylim_lower=1520):
    """Read in quantities and plot a viscoelastic progression from a transect dictionary
    Arguments:
        in_dict = a dictionary from sample_transect
        colormap = Matplotlib colormap instance, color scheme to use for plotting
    """
    xaxis = in_dict['xaxis']
    transect_length = max(xaxis)
    sevals_1 = in_dict['initial_surface_obs']
    try:
        sevals_2 = in_dict['second_surface_obs']
    except KeyError:
        print 'No secondary surface observations saved on transect {}. Setting identical to first surface for plotting.'.format(in_dict['name'])
        sevals_2 = sevals_1
    try:
        ve_profile_series = in_dict['VE_profiles']
        times = in_dict['VE_times']
    except KeyError:
        print 'No viscoelastic profiles saved. Unable to proceed.'
        return #exit the function
    try:
        elastic_profile = in_dict['elastic_profile']
    except KeyError:
        elastic_profile = ve_profile_series[0] #use first VE profile, from time t=0, as stand-in for pure elastic

    colors = colormap(np.linspace(0.1, 0.9, num=len(times)+1))
    
    fig = plt.figure('Viscoelastic progression, {}'.format(in_dict['name']), figsize=(7, 3))
    plt.plot(xaxis, sevals_1, color='k', ls='-.') #, label='15 Oct 2012'
    plt.plot(xaxis, sevals_2, color='k', ls='-', label='Obs.') #, label='10 Oct 2015'
    #plt.plot(xaxis, elas_profile_array, color='r', ls=':', label='Elastic beam')
    plt.plot(xaxis, elastic_profile, color=colors[0], lw=2, label='Elastic plate')
    for i,ti in enumerate(times[::10]):
        labeltime = int(round(ti/86400)) #time in days
        plt.plot(xaxis, ve_profile_series[i][:], ls='--', color=colors[i+1], lw=2, label='Viscoelastic, t = {} days'.format(labeltime))
    plt.fill_between(xaxis, sevals_1, sevals_2, color='Gainsboro', hatch='/', edgecolor='DimGray', linewidth=0, alpha=0.7)
    plt.fill_between(xaxis, sevals_2, ylim_lower, color='Azure')
    if make_legend:
        plt.legend(loc='lower left')
    else:
        pass
    plt.axes().set_aspect(5)
    plt.axes().set_xlim(0, transect_length)
    plt.axes().set_ylim(ylim_lower, 1700)
    plt.axes().set_yticks([1550, 1600, 1650, 1700])
    plt.axes().set_yticklabels(['', '', '', ''], fontsize=14)
    plt.axes().tick_params(which='both', labelsize=14)
    plt.axes().set_xticklabels([])
    plt.axes().set_xlabel('Along-transect distance [m]', fontsize=16)
    plt.axes().set_ylabel('Surface elevation [m a.s.l.]', fontsize=16)
    #plt.title('Eastern Skafta cauldron transect: observed, ideal elastic, ideal viscoelastic. E={:.1E}, eta={:.1E}'.format(ESkafta.youngmod, ESkafta.dyn_viscos), fontsize=18)
    plt.show()
    
    return fig  


## Test transecting functions
#endpoints_1 = [(-17.542113802658239, 64.488141277357315),
# (-17.48586677277758, 64.486397775690023)] #previous preset
endpoints_1 = [(-17.535314402804026, 64.495192470298178),
 (-17.491964721477643, 64.476306805753708)] #not much crevassing
endpoints_2 = [(-17.530965405648303, 64.478974272497283),
 (-17.49448994563258, 64.495192470298178)] #medium crevassing
endpoints_3 = [(-17.543170655730489, 64.487616864746443),
 (-17.484529339243668, 64.486123083370046)] #more crevassing

transect_dict_1 = sample_transect(endpoints_1, sefunc_2012, sefunc_2015, cauldron_name='Transect 1')
transect_dict_2 = sample_transect(endpoints_2, sefunc_2012, sefunc_2015, cauldron_name='Transect 2')
transect_dict_3 = sample_transect(endpoints_3, sefunc_2012, sefunc_2015, cauldron_name='Transect 3')


#f1 = plot_VE_transect(transect_dict_1)
#f2 = plot_VE_transect(transect_dict_2)
#f3 = plot_VE_transect(transect_dict_3)