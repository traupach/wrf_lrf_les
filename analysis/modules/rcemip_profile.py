# Functions to define profiles for he RCEMIP project, as defined in Wing et al 2018 (doi: 10.5194/gmd-11-793-2018).
# Author: Tim Raupach <t.raupach@unsw.edu.au>

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate
import xarray

# Define default analytic sounding parameters, from Wing et al 2018, Table 2 and text.
z_t_def = 15000    ## Height of the tropopause [m].
q_0_def = 0.01865  ## Surface specific humidity (for SST = 300 K) [kg kg-1].
q_t_def = 1e-14    ## Specific humidity in the upper atmosphere [kg kg-1].
z_q1_def = 4000    ## Constant 1 for specific humidity calculation [m].
z_q2_def = 7500    ## Constant 2 for specific humidity calculation [m].
T_0_def = 300      ## Sea surface temperature (SST) [K].
Gamma_def = 0.0067 ## Virtual temperature lapse rate [K m-1].
p_0_def = 1014.8   ## Surface pressure [hPa].
g_def = 9.79764    ## Mean surface gravity [m s-2].
R_d_def = 287.04   ## Gas constant for dry air [J kg-1 K-1].

def specific_humidity(z, q_0=q_0_def, z_q1=z_q1_def, z_q2=z_q2_def, z_t=z_t_def, q_t=q_t_def, **kwargs):
    """
    Return specific humidity q [kg kg-1] for height as per Wing et al 2018, Equation 2.
    
    Arguments:
    z:        Heights at which to calculate as np.array [m].
    q_0:      Surface specific humidity [kg kg-1].
    z_q1:     Constant altitude 1 [m].
    z_q2:     Constant altitude 2 [m].
    z_t:      Height of the tropopause [m].
    q_t:      Specific humidity in the upper atmosphere [kg kg-1].
    **kwargs: Extra arguments (ignored).
    """
    
    q_z = q_0 * np.exp(-z/z_q1) * np.exp(-(z/z_q2)**2)
    q_z[np.where(z > z_t)] = q_t
    return(q_z)
    
def virtual_temperature(z, T_0=T_0_def, q_0=q_0_def, Gamma=Gamma_def, z_t=z_t_def, **kwargs):
    """
    Return the initial virtual temperature T_z [K] for height as per Wing et al 2018, Equation 3.
    
    Arguments:
    z:     Heights at which to calculate as np.array [m].
    q_0:   Surface specific humidity [kg kg-1].
    T_0:   Sea surface temperature (SST) [K].  
    Gamma: Virtual temperature lapse rate [K m-1].
    z_t:  Height of the tropopause [m].
    **kwargs: Extra arguments (ignored).
    """
    
    T_v0 = T_0 * (1 + 0.608*q_0) # Virtual temperature at the surface [K].
    T_vz = T_v0 - Gamma*z
    T_vz[np.where(z > z_t)] = T_v0 - Gamma*z_t
    return(T_vz)

def temperature(z, **kwargs):
    """
    Return the temperature T [K] for height as per Wing et al 2018, Equation 4.
    
    Arguments: 
    z:        Heights at which to calculate as np.array [m].
    **kwargs: Arguments to virtual_temperature() or specific_humidity().
    """
    
    T_vz = virtual_temperature(z, **kwargs)
    q_z = specific_humidity(z, **kwargs)
    T_z = T_vz / (1 + 0.608*q_z)
    return(T_z)

def pressure(z, q_0=q_0_def, p_0=p_0_def, T_0=T_0_def, Gamma=Gamma_def, z_t=z_t_def, g=g_def, R_d=R_d_def):
    """
    Return the pressure p [hPa] for height as per Wing et al 2018, Equations 5 and 6.
    
    z:     Heights at which to calculate as np.array [m].
    q_0:   Surface specific humidity [kg kg-1].
    p_0:   Surface pressure [hPa].
    T_0:   Sea surface temperature (SST) [K].  
    Gamma: Virtual temperature lapse rate [K m-1].
    z_t:   Height of the tropopause [m].
    g:     Mean surface gravity [m s-2].
    R_d:   Gas constant for dry air [J kg-1 K-1].
    """
    
    T_v0 = T_0 * (1 + 0.608*q_0) # Virtual temperature at the surface [K].
    T_vt = T_v0 - Gamma*z_t      # Constant virtual temperature for upper atmosphere [K].
    p_t = p_0 * (T_vt/T_v0)**(g/(R_d*Gamma))
    p_z = p_0 * ((T_v0 - Gamma*z)/T_v0)**(g/(R_d*Gamma))
    p_z[np.where(z > z_t)] = p_t * np.exp(-((g*(z[np.where(z > z_t)]-z_t))/(R_d*T_vt)))
    return(p_z)

def initial_profile(z, **kwargs):
    """
    Calculate and return an initial profile for given heights.
    
    Arguments:
    z:        Heights at which to calculate as np.array [m].
    **kwargs: Arguments to specific_humidity(), temperature(), or pressure().
    """
     
    profile = xarray.Dataset(
        data_vars={
            'z': z,
            'q': (['z'], specific_humidity(z, **kwargs)),
            'T': (['z'], temperature(z, **kwargs)),
            'p': (['z'], pressure(z, **kwargs))
        })
    
    # Potential temperature.
    profile['theta'] = profile.T * (1000/profile.p)**(2/7)
    
    profile.z.attrs = {'name': 'Height', 'units': 'm'}
    profile.q.attrs = {'name': 'Specific humidity', 'units': 'kg kg-1'}
    profile.T.attrs = {'name': 'Temperature', 'units': 'K'}
    profile.p.attrs = {'name': 'Pressure', 'units': 'hPa'}
    profile.theta.attrs = {'name': 'Potential temperature', 'units': 'K'}
    return(profile)

def suggested_heights():
    """
    Return the heights suggested for RCEMIP layers as defined in Wing et al 2018 Table 3.
    """
    
    z = np.array([37,      112,   194,   288,   395,   520,   667,   843,  1062,  
                  1331,   1664,  2055,  2505,  3000,  3500,  4000,  4500,  5000,  
                  5500,   6000,  6500,  7000,  7500,  8000,  8500,  9000,  9500, 
                  10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 
                  14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500,
                  19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 
                  23500, 24000, 24500, 25000, 25500, 26000, 26500, 27000, 27500,
                  28000, 28500, 29000, 29500, 30000, 30500, 31000, 31500, 32000, 
                  32500, 33000])
    return(z)

def default_profile(**kwargs):
    """
    Return the default RCEMIP profile defined by Wing et al 2018 using the suggested heights.
    
    Arguments: 
    **kwargs: Optional arguments to initial_profile().
    """
    
    return(initial_profile(z=suggested_heights(), **kwargs))

def plot_default_profile(z_max=18600, **kwargs):
    """
    Plot the profiles made by default_profile().
    
    Arguments:
    zmax: The maximum height to plot to.
    **kwargs: Arguments to default_profile().
    """
    
    profile = default_profile(**kwargs)
    fig, ax = plt.subplots(ncols=3)
    profile.T.sel(z=slice(0,z_max)).plot(ax=ax[0], y='z')
    profile.q.sel(z=slice(0,z_max)).plot(ax=ax[1], y='z') 
    profile.p.sel(z=slice(0,z_max)).plot(ax=ax[2], y='z')

    for i in np.arange(1, len(ax)):
        ax[i].tick_params(labelleft=False)
        
    plt.tight_layout()
    plt.show()
    
def surface_q_interp(surface_T):
    """
    Use a simple linear interpolation to determine a surface value for specific humidity for a given 
    surface temperature, based on values given in Wing et al 2018.
    
    Arguments:
    surface_T: The surface temperature to interpolate for [K].
    """
    
    Ts = np.array([295, 300, 305])
    qs = np.array([0.012, 0.01865, 0.024])
    

    Ts = np.array([295, 300, 305])
    qs = np.array([0.012, 0.01865, 0.024])

    interp = sp.interpolate.interp1d(Ts, qs, fill_value='extrapolate' )
    return(interp(surface_T))
    
    