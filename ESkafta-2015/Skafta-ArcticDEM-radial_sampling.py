# Reading in ArcticDEM, finding center of cauldron and sampling radii around it
# 11 Mar 2019 EHU

import numpy as np
import scipy.misc as scp
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
import mpl_toolkits.basemap.pyproj as pyproj
from osgeo import gdal
from netCDF4 import Dataset
from sympy.integrals.transforms import inverse_laplace_transform
from sympy import Symbol
from sympy.abc import s, t
#import shapefile
#import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import math
#from matplotlib.colors import LogNorm
from shapely.geometry import *
from cauldron_funcs import *

## Haversine formula to calculate distance in m within cauldron (e.g. for radius)
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

skafta_region_path = 'Documents/6. MIT/Skaftar collapse/data/arcticDEM/'
nc_20121015_path = skafta_region_path + 'subset_nc/SETSM_WV02_20121015_skaftar_east_ll.nc'
nc_20151010_path = skafta_region_path + 'subset_nc/SETSM_WV02_20151010_skaftar_east_ll.nc'

lon_2012, lat_2012, se_2012 = read_ArcticDEM_nc(nc_20121015_path)
SE_2012 = np.ma.masked_where(se_2012==0, se_2012)
lon_2015, lat_2015, se_2015 = read_ArcticDEM_nc(nc_20151010_path)
SE_2015 = np.ma.masked_where(se_2015==0, se_2015)


# Plot cauldron surface elevation for inital selection of peripheral points
#plt.figure()
#plt.contourf(lon_2015, lat_2015, SE_2015, 100)
#plt.show()

### Interpolating surface elevation
sefunc_2012 = interpolate.interp2d(lon_2012, lat_2012, SE_2012)
sefunc_2015 = interpolate.interp2d(lon_2015, lat_2015, SE_2015)

## Finding center and sampling radii
#cauldron_periphery = np.asarray([(-17.543361216627368, 64.495296898783465),
# (-17.54553805337574, 64.492481870049829),
# (-17.54600451839325, 64.490211685587212),
# (-17.545849030054079, 64.487305849475078),
# (-17.544916100019062, 64.484309205984431),
# (-17.54273926327069, 64.482129828900312),
# (-17.539318519808958, 64.47985964443771),
# (-17.53403191627719, 64.477771074732104),
# (-17.528434336067086, 64.47631815667603),
# (-17.52252577917864, 64.475046853376966),
# (-17.515995268933516, 64.474320394348936),
# (-17.509620247027563, 64.474138779591925),
# (-17.503400713460778, 64.474320394348936),
# (-17.498580574946523, 64.474320394348936),
# (-17.492050064701399, 64.475319275512476),
# (-17.487540902865479, 64.477044615704074),
# (-17.484120159403748, 64.47985964443771),
# (-17.482720764351221, 64.482129828900312),
# (-17.481787834316204, 64.484944857633963),
# (-17.481943322655372, 64.488123115881621),
# (-17.482565276012053, 64.491119759372268),
# (-17.48396467106458, 64.493753173348892),
# (-17.485986019473785, 64.495750935675986),
# (-17.488473832900496, 64.497567083246082),
# (-17.491583599683889, 64.498656771788134),
# (-17.496092761519808, 64.499746460330186),
# (-17.500912900034066, 64.500926956250751),
# (-17.506199503565831, 64.501653415278781),
# (-17.5114861070976, 64.502198259549814),
# (-17.517239175646875, 64.502016644792803),
# (-17.52205931416113, 64.501199378386261),
# (-17.526568475997049, 64.50029130460122),
# (-17.53076666115463, 64.499564845573175),
# (-17.535586799668888, 64.498293542274112),
# (-17.538541078113109, 64.497294661110573)]) #latlon coordinates of points around cauldron, selected with ginput
#cauldron_periphery = np.asarray([(-17.542287762544468, 64.488577152774141),
# (-17.542287762544468, 64.482823597272088),
# (-17.537822792131259, 64.478987893604057),
# (-17.531444262969536, 64.475152189936026),
# (-17.523790027975465, 64.473553980074357),
# (-17.516135792981398, 64.471636128240334),
# (-17.505292293406466, 64.470996844295669),
# (-17.49636235258005, 64.472914696129678),
# (-17.489345970502153, 64.475791473880705),
# (-17.484881000088947, 64.477709325714713),
# (-17.481691735508086, 64.484102165161431),
# (-17.481691735508086, 64.488257510801802),
# (-17.485518853005122, 64.493052140386837),
# (-17.494448793831534, 64.497846769971872),
# (-17.50337873465795, 64.499764621805895),
# (-17.512946528400533, 64.500084263778234),
# (-17.52442788089164, 64.498486053916551),
# (-17.531444262969536, 64.496248560110203),
# (-17.535271380466572, 64.49465035024852),
# (-17.539098497963604, 64.492093214469833)]
#) #more southerly selection, tested 9 Apr 2019

print 'Transforming coordinates of cauldron periphery'
wgs84 = pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by ArcticDEM 
hjorsey = pyproj.Proj("+init=EPSG:3056") # UTM zone 28N for Iceland, so that coords will be in km northing/easting
#xt, yt = pyproj.transform(wgs84, hjorsey, cauldron_periphery[:,0], cauldron_periphery[:,1])
#cauldron_periph_utm = np.asarray([(xt[i], yt[i]) for i in range(len(xt))])
#transformed_mp = MultiPoint(cauldron_periph_utm)
#cauldron_center = transformed_mp.centroid #finding UTM coordinates of cauldron center
cauldron_center_latlon = (-17.5158322984487, 64.48773309819093) #ginput selection of deepest point
cauldron_center = Point(pyproj.transform(wgs84, hjorsey, cauldron_center_latlon[0], cauldron_center_latlon[1]))
cauldron_radius = 1540 #m, based on previous calculation with centroid

nradii = 99
res = int(np.floor(nradii/4)) #choose resolution for Shapely buffer based on how many sample points we want
#cauldron_radius = np.mean([cauldron_center.distance(p) for p in transformed_mp])
radial_buffer = cauldron_center.buffer(distance = cauldron_radius, resolution=res) #set of points at distance R from the centroid
radial_pts = np.asarray(list(radial_buffer.exterior.coords))
rpx, rpy = pyproj.transform(hjorsey, wgs84, radial_pts[:,0], radial_pts[:,1])
radial_pts_latlon = np.asarray([(rpx[i], rpy[i]) for i in range(len(rpx))])
#center_latlon = pyproj.transform(hjorsey, wgs84, list(cauldron_center.coords)[0][0], list(cauldron_center.coords)[0][1])
center_latlon = cauldron_center_latlon

### check UTM selection of pts
#plt.figure()
#plt.scatter(radial_pts[:,0], radial_pts[:,1])
#for i in range(len(radial_pts)):
#    plt.axes().annotate(i, radial_pts[i])
#plt.axes().set_aspect(1)
#plt.show()
#
### check latlon pts over ArcticDEM
#plt.figure('Skafta region and selected radial points')
#plt.contourf(lon_2015, lat_2015, SE_2015, 100)
#plt.scatter(radial_pts_latlon[:,0], radial_pts_latlon[:,1])
#plt.plot(center_latlon[0], center_latlon[1], marker='*')
#plt.show()

#endpoints = [(-17.542113802658239, 64.488141277357315),
# (-17.48586677277758, 64.486397775690023)] #coordinates at either side of the cauldron, selected by inspection with ginput.  
#lonvals = np.linspace(endpoints[0][0], endpoints[1][0], npoints)
#latvals = np.linspace(endpoints[0][1], endpoints[1][1], npoints)
#sevals_2012 = np.asarray([sefunc_2012(lonvals[i], latvals[i]) for i in range(npoints)]).squeeze()
#sevals_2015 = np.asarray([sefunc_2015(lonvals[i], latvals[i]) for i in range(npoints)]).squeeze()
#
## sample along radii
nsamples = 500 #number of points to sample from interpolated ArcticDEM.  500 gives resolution ~2.5 m
se_radii_2012 = {}
se_radii_2015 = {}
for j, radial_pt in enumerate(radial_pts_latlon):
    lonvals = np.linspace(center_latlon[0], radial_pt[0], nsamples)
    latvals = np.linspace(center_latlon[1], radial_pt[1], nsamples)
    se_radii_2012[j] = np.asarray([sefunc_2012(lonvals[i], latvals[i]) for i in range(nsamples)]).squeeze()
    se_radii_2015[j] = np.asarray([sefunc_2015(lonvals[i], latvals[i]) for i in range(nsamples)]).squeeze()

radial_curvature_2012 = {}
radial_curvature_2015 = {}
#savgol_window = 7
#savgol_polyorder = 5
#for k in se_radii_2012.keys():
#    radial_curvature_2012[k] = savgol_filter(se_radii_2012[k], savgol_window, savgol_polyorder, deriv=2)
#    radial_curvature_2015[k] = savgol_filter(se_radii_2015[k], savgol_window, savgol_polyorder, deriv=2)
## Do with radial gradient instead
for k in se_radii_2012.keys():
    radial_curvature_2012[k] = np.gradient(np.gradient(se_radii_2012[k]))
    radial_curvature_2015[k] = np.gradient(np.gradient(se_radii_2015[k]))
    

radial_length = haversine(radial_pts_latlon[0][::-1], center_latlon[::-1])
radial_axis = np.linspace(0, radial_length, num=nsamples)

crevasse_locations = []
cl_dict = {} #dictionary to sort crevasse locations by azimuth
lower_SE_limit = SE_2015.min() #lowest surface elevation value we believe, for testing curvature peaks raised by nodata areas
testradius = 3 #how far around suspect point we want to check. determines speed.
crevasse_curvature = 5 #value of curvature at which to treat as a potential crevasse.  Mean curvature is ~0.001, but many non-crevasse peaks around 0.1-1 range.
for k in radial_curvature_2015.keys():
    curv_vals = radial_curvature_2015[k]
    SE_vals = se_radii_2015[k]
    cl_dict[k] = []
    for i in range(len(curv_vals)-testradius):
        test_validity = [SE_vals[i+j]>lower_SE_limit for j in range(-1*testradius, testradius)] #check whether this is an area of no data
        if abs(curv_vals[i])>crevasse_curvature: #candidate crevasse
            if sum(test_validity)==len(test_validity): # if all surrounding values are valid data
                crevasse_locations.append(radial_axis[i])
                cl_dict[k].append(radial_axis[i])

crevasse_locations_12 = [] #compare to 2012 / pre-collapse surface
cl_dict_12 = {} 
for k in radial_curvature_2012.keys():
    curv_12_comp = radial_curvature_2012[k]
    cl_dict_12[k] = []
    for i in range(len(curv_12_comp)):
        if abs(curv_12_comp[i])>crevasse_curvature:    
            crevasse_locations_12.append(radial_axis[i])
            cl_dict_12[k].append(radial_axis[i])
    
#x_cylcoords = np.linspace(-0.5*transect_length, 0.5*transect_length, num=npoints)
#initial_surf_val = np.mean((sevals_2012[0], sevals_2012[-1])) #surface elevation at edges before loading
initial_surf = interpolate.interp1d(radial_axis, se_radii_2012[0], kind='quadratic') #will have matching length as long as num=npoints in x_cylcoords above
ESkafta = Cauldron(name='Eastern_Skafta', initial_surface = lambda x: max(se_radii_2012[0]), radius = radial_length)
#ESkafta.set_viscoelastic_bendingmod()

stress_array = [ESkafta.elastic_stress(x) for x in radial_axis]
#elas_profile_array = [ESkafta.elastic_beam_profile(x) for x in radial_axis]
LL_profile_array = [ESkafta.LL_profile(x) for x in radial_axis]
#nseconds = 5*24*60*60 #number of seconds in the roughly 5-day collapse period
#times = np.arange(0, nseconds, step=86400)
#ve_profile_series = [[ESkafta.viscoelastic_profile(x, t0) for x in radial_axis] for t0 in times]
#
#elas_beam_stress = [ESkafta.elastic_stress(x, config='beam') for x in radial_axis]
elas_plate_stress = [ESkafta.elastic_stress(x, config='radial_plate') for x in radial_axis]
#ve_plate_stress_min = [ESkafta.viscoelastic_stress(x, times[0]) for x in radial_axis]
#ve_plate_stress_max = [ESkafta.viscoelastic_stress(x, times[4]) for x in radial_axis]
#ve_bendingmod_series = [ESkafta.symbolic_ve_D(t0) for t0 in times]


## Make figures

cmap = cm.get_cmap('winter_r')
colors = cmap([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
#colors = cmap(np.linspace(0.1, 0.9, num=len(times)+1))


#for j in (59, 88, 92):
#    plt.figure('Radial profile {}'.format(j))
#    plt.plot(radial_axis, se_radii_2012[j], ls='-.') #, label='15 Oct 2012'
#    plt.plot(radial_axis, se_radii_2015[j], ls='-') #, label='10 Oct 2015'
#    c_locs = cl_dict[j] #crevasses associated with this radial transect
#    for c in c_locs:
#        idx = (np.abs(radial_axis - c)).argmin() #find index of closest radial value to crevasse location c
#        plt.axvline(x=c, color='k', alpha=0.5)
#    #plt.fill_between(radial_axis, sevals_2012, sevals_2015, color='Gainsboro', hatch='/', edgecolor='DimGray', linewidth=0, alpha=0.7)
#    #plt.fill_between(radial_axis, sevals_2015, (plt.axes().get_ylim()[0]), color='Azure')
#    #plt.legend(loc='lower right')
#    plt.axes().set_aspect(1)
#    plt.axes().set_xlim(0, radial_length)
#    plt.axes().set_ylim(1400, 1800)
#    #plt.axes().set_yticks([1550, 1600, 1650, 1700])
#    #plt.axes().set_yticklabels(['1550', '1600', '1650', '1700'], fontsize=14)
#    plt.axes().tick_params(which='both', labelsize=14)
#    #plt.axes().set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)
#    plt.axes().set_xlabel('Radial distance [m]', fontsize=16)
#    plt.axes().set_ylabel('Surface elevation [m a.s.l.]', fontsize=16)
#    plt.title('Eastern Skafta cauldron radial samples', fontsize=18)
#    plt.show()

plt.figure('Radial curvature')
for j in radial_curvature_2015.keys():
    #plt.plot(radial_axis, radial_curvature_2012[j], ls='-.') #, label='15 Oct 2012'
    plt.plot(radial_axis, radial_curvature_2015[j], ls='-') #, label='10 Oct 2015'
#for c in crevasse_locations:
#    plt.axvline(x=c, color='Gainsboro')
#plt.legend(loc='lower right')
plt.axes().set_aspect(1)
plt.axes().set_xlim(0, radial_length)
#plt.axes().set_ylim(1400, 1800)
#plt.axes().set_yticks([1550, 1600, 1650, 1700])
#plt.axes().set_yticklabels(['1550', '1600', '1650', '1700'], fontsize=14)
plt.axes().tick_params(which='both', labelsize=14)
#plt.axes().set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)
plt.axes().set_xlabel('Radial distance [m]', fontsize=16)
plt.axes().set_ylabel('Surface curvature', fontsize=16)
plt.title('Eastern Skafta cauldron radial curvature', fontsize=18)
plt.show()

plt.figure('Elastic only')
plt.plot(radial_axis, se_radii_2012[0], color='k', ls='-.') #, label='15 Oct 2012'
plt.plot(radial_axis, se_radii_2015[0], color='k', ls='-', label='Obs.') #, label='10 Oct 2015'
#plt.plot(radial_axis, elas_profile_array, color='r', ls=':', label='Elastic beam')
plt.plot(radial_axis, LL_profile_array, color=colors[0], lw=2, label='Elastic plate')
plt.fill_between(radial_axis, se_radii_2012[0], se_radii_2015[0], color='Gainsboro', hatch='/', edgecolor='DimGray', linewidth=0, alpha=0.7)
plt.fill_between(radial_axis, se_radii_2015[0], (plt.axes().get_ylim()[0]), color='Azure')
plt.legend(loc='lower left')
plt.axes().set_aspect(5)
plt.axes().set_xlim(0, radial_length)
#plt.axes().set_yticks([1550, 1600, 1650, 1700])
#plt.axes().set_yticklabels(['1550', '1600', '1650', '1700'], fontsize=14)
plt.axes().tick_params(which='both', labelsize=14)
#plt.axes().set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)
plt.axes().set_xlabel('Along-transect distance [km]', fontsize=16)
plt.axes().set_ylabel('Surface elevation [m a.s.l.]', fontsize=16)
plt.title('Eastern Skafta cauldron radial transect: observed, ideal elastic. E={:.1E}'.format(ESkafta.youngmod), fontsize=18)
plt.show()
##plt.savefig('Skafta-transect-aspect_5.png', transparent=True)
#
#plt.figure('Viscoelastic progression')
#plt.plot(radial_axis, se_radii_2012[0], color='k', ls='-.') #, label='15 Oct 2012'
#plt.plot(radial_axis, se_radii_2015[0], color='k', ls='-', label='Obs.') #, label='10 Oct 2015'
##plt.plot(radial_axis, elas_profile_array, color='r', ls=':', label='Elastic beam')
##plt.plot(radial_axis, LL_profile_array, color=colors[0], lw=2, label='Elastic plate')
#for i,ti in enumerate(times):
#    plt.plot(radial_axis, ve_profile_series[i][:], ls='--', color=colors[i+1], lw=2)
#plt.fill_between(radial_axis, se_radii_2012[0], se_radii_2015[0], color='Gainsboro', hatch='/', edgecolor='DimGray', linewidth=0, alpha=0.7)
#plt.fill_between(radial_axis, se_radii_2015[0], (plt.axes().get_ylim()[0]), color='Azure')
#plt.legend(loc='lower left')
##plt.axes().set_aspect(5)
#plt.axes().set_xlim(0, radial_length)
##plt.axes().set_yticks([1550, 1600, 1650, 1700])
##plt.axes().set_yticklabels(['1550', '1600', '1650', '1700'], fontsize=14)
#plt.axes().tick_params(which='both', labelsize=14)
##plt.axes().set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)
##plt.axes().set_xlabel('Radial distance [km]', fontsize=16)
##plt.axes().set_ylabel('Surface elevation [m a.s.l.]', fontsize=16)
#plt.title('Eastern Skafta cauldron transect: observed, ideal elastic, ideal viscoelastic. E={:.1E}'.format(ESkafta.youngmod), fontsize=18)
#plt.show()
#
#
plt.figure('Crevasse stresses')
plt.plot(radial_axis, 1E-6*np.array(elas_plate_stress), color='k', ls='-', lw=2, label='Elastic plate')
#plt.plot(radial_axis, 1E-6*np.array(ve_plate_stress_max), color='k', ls='-.', lw=2, label='Viscoelastic plate, t={} days'.format(np.ceil(max(times)/86400)))
stress_norm = mpl.colors.Normalize(vmin = -1*max(elas_plate_stress), vmax = max(elas_plate_stress))
#for c in crevasse_locations:
for k in cl_dict.keys():
    c_locs = cl_dict[k]
    for c in c_locs:
        idx = (np.abs(radial_axis - c)).argmin() #find index of closest radial value to crevasse location c
        stress_scale = stress_norm(elas_plate_stress[idx]) #norm nearest stress val versus maximum tensile stress
        stress_color = cm.get_cmap('coolwarm')(stress_scale)
        plt.axvline(x=c, color=stress_color, alpha=0.5, lw=2.0)
        #plt.annotate(s=str(k), xy=(c, 0)) #label crevasse line with which azimuth produced it
plt.plot(radial_axis, np.zeros(len(radial_axis)), color='b', ls=':')
plt.legend(loc='upper right')
plt.axes().tick_params(which='both', labelsize=14)
plt.axes().set_xlim(0, radial_length)
plt.axes().set_xlabel('Radial distance [m]', fontsize=16)
plt.axes().set_ylabel('Radial stress [MPa]', fontsize=16)
plt.title('Stress at cauldron surface', fontsize=18)
plt.show()

plt.figure('Crevasse stresses: outliers removed')
plt.plot(radial_axis, 1E-6*np.array(elas_plate_stress), color='k', ls='-', lw=2)
#plt.plot(radial_axis, 1E-6*np.array(ve_plate_stress_max), color='k', ls='-.', lw=2, label='Viscoelastic plate, t={} days'.format(np.ceil(max(times)/86400)))
stress_norm = mpl.colors.Normalize(vmin = -1*max(elas_plate_stress), vmax = max(elas_plate_stress))
#for c in crevasse_locations:
for k in cl_dict.keys():
    c_locs = cl_dict[k]
    if k == 61: #removes a crevasse indicated near 0 stress, when radial profile indicates clearly nonzero stress region
        continue
    for c in c_locs:
        idx = (np.abs(radial_axis - c)).argmin() #find index of closest radial value to crevasse location c
        stress_scale = stress_norm(elas_plate_stress[idx]) #norm nearest stress val versus maximum tensile stress
        stress_color = cm.get_cmap('coolwarm')(stress_scale)
        plt.axvline(x=c, color=stress_color, alpha=0.5, lw=2.0)
        #plt.annotate(s=str(k), xy=(c, 0)) #label crevasse line with which azimuth produced it
plt.plot(radial_axis, np.zeros(len(radial_axis)), color='b', ls=':')
plt.legend(loc='upper right')
plt.axes().tick_params(which='both', labelsize=14)
plt.axes().set_xlim(0, radial_length)
plt.axes().set_xlabel('Radial distance [m]', fontsize=16)
plt.axes().set_ylabel('Radial stress [MPa]', fontsize=16)
plt.title('Stress at cauldron surface', fontsize=18)
plt.show()

# add annotations for zoomed-in first crevasses in each regime
first_tensile_xy = (1058.081049934334, -8.4310432716652333) #annotation location for first crevasse in tensile regime
first_compressive_xy = (664.37888645060593, 14.421886161521616) #annotation location for first crevasse in compressive regime
# plot with annotations
plt.figure('First-in-regime crevasses')
plt.plot(radial_axis, 1E-6*np.array(elas_plate_stress), color='k', ls='-', lw=2)
#plt.plot(radial_axis, 1E-6*np.array(ve_plate_stress_max), color='k', ls='-.', lw=2, label='Viscoelastic plate, t={} days'.format(np.ceil(max(times)/86400)))
stress_norm = mpl.colors.Normalize(vmin = -1*max(elas_plate_stress), vmax = max(elas_plate_stress))
#for c in crevasse_locations:
for k in cl_dict.keys():
    c_locs = cl_dict[k]
    if k == 61: #removes a crevasse indicated near 0 stress, when radial profile indicates clearly nonzero stress region
        continue
    for c in c_locs:
        idx = (np.abs(radial_axis - c)).argmin() #find index of closest radial value to crevasse location c
        stress_scale = stress_norm(elas_plate_stress[idx]) #norm nearest stress val versus maximum tensile stress
        stress_color = cm.get_cmap('coolwarm')(stress_scale)
        plt.axvline(x=c, color=stress_color, alpha=0.5, lw=2.0)
        #plt.annotate(s=str(k), xy=(c, 0)) #label crevasse line with which azimuth produced it
plt.plot(radial_axis, np.zeros(len(radial_axis)), color='b', ls=':')
plt.scatter(first_tensile_xy[0], first_tensile_xy[1], marker='*', s=20, color='b')
plt.scatter(first_compressive_xy[0], first_compressive_xy[1], marker='*', s=20, color='r')
#plt.annotate(s='Compressive stress: 14.4 MPa', xy=first_compressive_xy, fontsize=14)
#plt.annotate(s='Tensile stress: -8.4 MPa', xy=first_tensile_xy, fontsize=14)
plt.legend(loc='upper right')
plt.axes().tick_params(which='both', labelsize=14)
plt.axes().set_xlim(630, 1100)
plt.axes().set_xlabel('Radial distance [m]', fontsize=16)
plt.axes().set_ylabel('Radial stress [MPa]', fontsize=16)
plt.title('Stress at cauldron surface', fontsize=18)
plt.show()
#
### individual plots per azimuth
#for k in range(70, 85):
#    cl = cl_dict[k]
#    plt.figure('Stress comparison, radius {}'.format(k))
#    plt.plot(radial_axis, 1E-6*np.array(elas_plate_stress), color='k', ls='-', lw=2, label='Elastic plate')
#    plt.plot(radial_axis, 1E-6*np.array(ve_plate_stress_max), color='k', ls='-.', lw=2, label='Viscoelastic plate, t={} days'.format(np.ceil(max(times)/86400)))
#    stress_norm = mpl.colors.Normalize(vmin = -1*max(ve_plate_stress_max), vmax = max(ve_plate_stress_max))
#    for c in cl:
#        idx = (np.abs(radial_axis - c)).argmin() #find index of closest radial value to crevasse location c
#        stress_scale = stress_norm(ve_plate_stress_max[idx]) #norm nearest stress val versus maximum tensile stress
#        stress_color = cm.get_cmap('coolwarm')(k)
#        plt.axvline(x=c, color=stress_color, alpha=0.5)
#    plt.plot(radial_axis, np.zeros(len(radial_axis)), color='b', ls=':')
#    plt.legend(loc='upper right')
#    plt.axes().tick_params(which='both', labelsize=14)
#    plt.axes().set_xlim(0, radial_length)
#    plt.axes().set_xlabel('Radial distance [m]', fontsize=16)
#    plt.axes().set_ylabel('Radial stress [MPa]', fontsize=16)
#    plt.title('Stress at cauldron surface', fontsize=18)
#    plt.show()
#
