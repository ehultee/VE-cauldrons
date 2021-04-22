## Utility functions for viscoelastic analysis of cauldron collapse
## 16 July 2019  EHU
import numpy as np
import scipy.misc as scp
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from osgeo import gdal
from netCDF4 import Dataset
from sympy.integrals.transforms import inverse_laplace_transform
from sympy import Symbol
from sympy.abc import s, t
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import math

##------------------------
## Analytical functions
##------------------------
class Ice(object):
    """Holds constants passed to Cauldron, set to default values but adjustable.
    Default values:
        g = 9.8 m/s^2, accel due to gravity
        rho_ice = 920.0 kg m^-3, density of glacier ice
        youngmod = 3E9 Pa, Young's modulus of ice (many estimates, see e.g. 9E9 in Reeh 2003)
        poisson_nu = 0.5, Poisson's ratio for viscoelastic ice
        dyn_viscos = 1E14 Pa s, dynamic viscosity of glacier ice
        #shearmod = 3.46E9 Pa, shear modulus of glacier ice (currently set in terms of Young's modulus, Poisson's ratio)
        #lame_lambda = 5.19E9 Pa, first Lame parameter of glacier ice (currently set in terms of Young's modulus, Poisson's ratio)
    Other attributes:
        t_relax: Maxwell relaxation timescale 
    """
    def __init__(self, g=9.8, rho_ice=920.0, youngmod = 1E9, poisson_nu = 0.3, dyn_viscos = 3E13):
        self.g = g
        self.rho_ice = rho_ice
        self.youngmod = youngmod
        self.poisson_nu = poisson_nu
        self.dyn_viscos = dyn_viscos
        
    @property
    def shearmod(self):
        return self.youngmod / (2*(1+self.poisson_nu))
        
    @property
    def lame_lambda(self):
        return (self.youngmod * self.poisson_nu)/((1+self.poisson_nu)*(1-2*self.poisson_nu))

    @property
    def t_relax(self):
        return self.dyn_viscos/self.shearmod
#         self.lame_lambda = (self.youngmod * self.poisson_nu)/((1+self.poisson_nu)*(1-2*self.poisson_nu))
#         self.t_relax = dyn_viscos/self.shearmod

        
class Cauldron(Ice):
    """Consistent simulation of elastic or viscoelastic cauldron collapse.
    Attributes:
        name: A string with what we call this cauldron for modelling/analysis
        thickness: ice thickness of collapsing portion in m.  Default 300 m (for Skafta)
        radius: radius of cauldron in m.  Default 1500 m (for Skafta) but should be set by observations for best match
        initial_surface: a function of radial coordinate (r=0 at center of cauldron) describing pre-collapse surface elevation
        bending_mod: (elastic) bending modulus in Pa m^3.  Calculated from other inputs.
    Inherits material properties from class Ice.
    """
    def __init__(self, name='Cauldron', thickness=300, radius=1500, initial_surface=lambda x: 1000):
        super(Cauldron,self).__init__() #inherit quantities from Ice
        self.name = name
        self.thickness = thickness
        self.radius = radius
        self.initial_surface = initial_surface
        
    @property
    def bending_mod(self):
        return self.youngmod * self.thickness **3 / (12*(1-self.poisson_nu**2))
    
    def elastic_beam_deform(self, x, loading=None):
        """Calculate displacement due to elastic deformation of an ice beam.  Returns deformation as a function of x, with x=0 at center of cauldron.
        Args:
            
        """
        if loading is None:
            loading = self.rho_ice * self.g * self.thickness #basic uniform loading of unsupported ice
        
        disp = (-1*loading/(24*self.bending_mod)) * (x**4 - 2* self.radius**2 * x**2 + self.radius**4)
        
        return disp
    
    def elastic_beam_profile(self, x):
        return self.initial_surface(x) + self.elastic_beam_deform(x)
    
    def LL_radial_deform(self, r, loading=None):
        """Radially symmetric deformation according to solution presented in Landau & Lifshitz for circular plate"""
    
        if loading is None:
            loading = self.rho_ice * self.g * self.thickness #basic uniform loading of unsupported ice
            
        LL_beta = 3 * loading *(1-self.poisson_nu**2) / (16 * self.thickness**3 * self.youngmod)
        
        LL_disp = (-1*LL_beta) * (self.radius**2 - r**2)**2
        
        return LL_disp
        
    def LL_profile(self, r):
        return self.initial_surface(r) + self.LL_radial_deform(r)
        
    def elastic_stress(self, x_eval, dx = 0.5, z=None, config='radial_plate'):
        """Calculate stress in an elastically deformed ice beam.  Returns stress at point x_eval
        Default args: 
            dx = 0.5 m, step size for finite difference approx to derivative
            z = thickness/2, distance above neutral surface to calculate stress
            config: 'beam' or 'radial_plate' (radially symmetric plate)
        """
        if z is None:
            z = 0.5 * self.thickness ##make the default location location of stress calculation the ice surface, i.e. half the ice thickness above the neutral surface
        
        if config=='beam':    
            disp_func = lambda x: self.elastic_beam_deform(x)    
            elastic_strain = z * scp.derivative(disp_func, x0=x_eval, dx=dx, n=2)
            hookean_stress =  self.youngmod * elastic_strain
            return hookean_stress
        
        if config=='radial_plate':
            disp_func = lambda x: self.LL_radial_deform(x)
            strain_rr = z * scp.derivative(disp_func, x0=x_eval, dx=dx, n=2)
            strain_thth = z * scp.derivative(disp_func, x0=x_eval, dx=dx, n=1) / self.radius
            kl_stress_rr = (self.youngmod/(1-self.poisson_nu**2)) * (strain_rr + self.poisson_nu * strain_thth) #Kirchhoff-Love stress for circular plate
            return kl_stress_rr
    
    def set_viscoelastic_bendingmod(self):
        """Construct time-dependent function viscoelastic (time-dependent) bending modulus by taking inverse Laplace transform of laplace_transformed_D.
        t0: time at which to evaluate"""
        s = Symbol('s') #Laplace variable s, to be used in SymPy computation
        t = Symbol('t', positive=True)
        laml = Symbol('laml', positive=True) #stand-in symbol for lame_lambda
        m = Symbol('m', positive=True) #stand-in symbol for shearmod
        tr = Symbol('tr', positive=True) #stand-in symbol for t_relax
        h = Symbol('h', positive=True) #stand-in symbol for thickness
        
        lambda_bar = laml + (2*m / (3*(1 + tr * s))) #transformed Lame lambda
        mu_bar = (tr * s /(1 + tr * s))*m #transformed Lame mu (shear mod)
        
        #self.lambda_bar = lambda_bar
        #self.mu_bar = mu_bar
        
        youngmod_bar = 2*mu_bar + (mu_bar*lambda_bar / (mu_bar + lambda_bar))
        poisson_bar = lambda_bar / (2*(mu_bar + lambda_bar))
        
        bending_mod_bar = youngmod_bar * h**3 / (12*(1-poisson_bar**2))
                
        symbolic_ve_D = inverse_laplace_transform(bending_mod_bar/s, s, t) #construct viscoelastic D(t) through SymPy inverse Laplace transform
        self.symbolic_ve_D = lambda t0: symbolic_ve_D.subs(((laml, self.lame_lambda), (m, self.shearmod), (tr, self.t_relax), (h, self.thickness), (t, t0)))
        #return symbolic_ve_D.subs(((t, t0), (laml, self.lame_lambda), (m, self.shearmod), (tr, self.t_relax), (h, self.thickness))) #evaluate D(t) at point t0 as expected
    
    
    def viscoelastic_deformation(self, x, t0, loading=None):
        """Collapse of a viscoelastic, radially symmetric plate solved by correspondence with elastic case."""
        if loading is None:
            loading = self.rho_ice * self.g * self.thickness #basic uniform loading of unsupported ice
        if self.symbolic_ve_D is None:
            self.set_viscoelastic_bendingmod()
        
        ve_disp = (-1*loading/(64*self.symbolic_ve_D(t0))) * (self.radius**2 - x**2)**2 #by Laplace transform correspondence with LL_radial_deform
        
        #ve_disp = (-1*loading/(24*self.symbolic_ve_D(t0))) * (x**4 - 2* self.radius**2 * x**2 + self.radius**4)
        
        return ve_disp
    
    def viscoelastic_profile(self, x, t0):
        return self.initial_surface(x) + self.viscoelastic_deformation(x, t0)
    
    def viscoelastic_stress(self, x, t0, loading=None, z=None, config='radial_plate'):
        """Stress in a viscoelastic, radially symmetric plate by correspondence with Kirchhoff-Love elastic case"""
        if loading is None:
            loading = self.rho_ice * self.g * self.thickness #basic uniform loading of unsupported ice
        if z is None:
            z = 0.5 * self.thickness ##make the default location location of stress calculation the ice surface, i.e. half the ice thickness above the neutral surface
        if self.symbolic_ve_D is None:
            self.set_viscoelastic_bendingmod()
        
        ve_strain_rr = (loading*z/(16*self.symbolic_ve_D(t0))) * (self.radius**2 - 3*x**2)
        ve_strain_thth = (loading*z/(16*self.symbolic_ve_D(t0))) * (self.radius**2 - x**2)
        
        ve_stress_rr = (self.youngmod / (1 - self.poisson_nu**2)) *(ve_strain_rr + self.poisson_nu * ve_strain_thth)
        
        return ve_stress_rr

##------------------------
## File I/O and processing
##------------------------

def read_ArcticDEM_tif(filename, return_grid=True, return_proj=False):
    """Extract x, y, v from an ArcticDEM GeoTIFF"""
    ds = gdal.Open(filename)
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize
    
    geotransform = ds.GetGeoTransform()
    xOrigin = geotransform[0]
    xPix = geotransform[1] #pixel width in x-direction
    yOrigin = geotransform[3]
    yPix = geotransform[5] #pixel width in y-direction
    
    lons = xOrigin + np.arange(0, ncols)*xPix
    lats = yOrigin + np.arange(0, nrows)*yPix
    
    x, y = np.meshgrid(lons, lats)
    
    hband = ds.GetRasterBand(1)
    harr = hband.ReadAsArray()
    
    if return_grid and return_proj:
        return x, y, harr, ds.GetProjection()
    elif return_grid:
        return x, y, harr
    else:
        return harr
    

def read_ArcticDEM_nc(filename, return_grid=True):
    fh = Dataset(filename, mode='r')
    lon = fh.variables['lon'][:].copy() #longitude
    lat = fh.variables['lat'][:].copy() #latitude
    se = fh.variables['Band1'][:].copy() #assuming the variable called "GDAL Band Number 1" is actually surface elevation
    
    if return_grid:
        return lon, lat, se
    else:
        return se


def NearestFill(data, mask=None):
    """
    Replace the value of masked 'data' cells (indicated by 'mask') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        mask: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 For a masked input array, use data.mask
                 If None (default), use: mask  = np.isnan(data)

    Output: 
        Return a filled array. 
    """

    if mask is None: mask = np.isnan(data)

    ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return data[tuple(ind)]