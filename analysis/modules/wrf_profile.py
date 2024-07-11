# Functions to compute profiles as per the WRF equations in the WRF tech note version
# (https://www2.mmm.ucar.edu/wrf/users/docs/technote/contents.html).
# Author: Tim Raupach <t.raupach@unsw.edu.au>.

import os
import xarray
import numpy as np
import pandas as pd
import modules.atmosphere as atm

# A note on full and half levels: WRF has staggered coordinates. Full levels refer to 
# grid boundaries or layer boundaries, while half levels are the centre of each grid point.
# In the vertical case:
#
#  ----- - level top height, full level
#    x   - half level height 
#  ----- - level bottom height, full level
#
# NOTE: Most data structures in this code are assumed to be simple np.arrays. Some 
# xarray objects are used, but arithmetic operations between xarray structures 
# use automatic broadcasting to match coordinates, which can silently produce
# unexpected results if the operations are supposed to be across coordinates.

def integrate_moist_wrf(rho_bottom, p_bottom, q_bottom, q_top, theta_top, dz, it=10, g=9.81, **kwargs):
    """
    Integrate the values of density and pressure up a column by calculating their values
    a height dz m above known values.
    
    rho_bottom: Density [kg m-3] at height z.
    p_bottom: Moist pressure [hPa] at height z. 
    q_bottom: Mixing ratio for water vapour [kg kg-1] at height z.
    q_top: Mixing ratio for water vapour [kg kg-1] at height z+dz.
    theta_top: Input potential temperature [K].
    dz: Upward distance to integrate above z [m].
    it: Number of iterations to use in estimation.
    g: Gravitational constant [m s-2].
    **kwargs: Arguments to atm.density().
    """

    rho_top = rho_bottom
    for i in range(it):
        # Note p_s is in hPa, and density() expects hPa, hence /100 compared to WRF code.
        p_top = p_bottom - (0.5 * dz * (rho_bottom + rho_top)*g*(1 + (q_bottom + q_top)/2))/100 
        rho_top = atm.density(p=p_top, theta=theta_top, q_v=q_top, **kwargs)
    return(p_top, rho_top)

def integrate_dry_wrf(p_top, rho_top, rho_bottom, dz, g=9.81):
    """
    Given a pressure at the top of a layer, the layer defined by two density levels 
    separated by a height difference, calculate the dry pressure [hPa] at the layer bottom.
    
    Arguments:
    p_top: Pressure at the layer top [hPa].
    rho_top: Density at the layer top [kg m-3].
    rho_bottom: Density at the layer bottom [kg m-3].
    dz: Layer thickness [m].
    g: Gravitational constant [m s-2].
    """
    return(p_top + (0.5*dz*(rho_bottom+rho_top)*g)/100)

def compute_wrf_profile(p_s, T_s, q_s, z, theta, q_v, dry=False, verbose=True):
    """
    Compute and return pressure and density for each sounding point using WRF equations.
    To calculate a dry sounding, set q_v = 0 to ignore moisture.
    
    p_s: Surface pressure [hPa].
    T_s: Surface temperature [K].
    q_s: Surface water vapour mixing ratio [kg kg-1].
    z: The input sounding heights above the surface [m].
    theta: Input sounding dry potential temperatures for each height [K].
    q_v: Input sounding water vapour mixing ratios for each height [kg kg-1].
    dry: If True, compute dry sounding by ignoring moisture in q_v.
    verbose: Print verbose information while running? (Default: True).
    """
    
    # Calculate surface potential temperature.
    theta_s = T_s * (1000/p_s)**(2/7) 
    
    # Add the surface values into the profile arrays.
    z = np.insert(z, 0, 0)
    theta = np.insert(theta, 0, theta_s)
    q_v = np.insert(q_v, 0, q_s)
    
    if dry:
        q_v = np.zeros(len(q_v))

    # Set up output arrays.
    p_moist = np.zeros(len(z)) # Moist pressure [hPa].
    rho = np.zeros(len(z))     # Density [kg m-3].
    p_dry = np.zeros(len(z))   # Dry pressure [Pa].

    # Set surface values.
    p_moist[0] = p_s
    rho[0] = atm.density(p=p_s, theta=theta_s, q_v=q_v[0])

    # Integrate up the column finding moist profile values.
    for x in range(len(z)-1):
        p_moist[x+1], rho[x+1] = integrate_moist_wrf(rho_bottom=rho[x], p_bottom=p_moist[x], 
                                                     q_bottom=q_v[x], q_top=q_v[x+1], 
                                                     theta_top=theta[x+1], dz=z[x+1]-z[x])
    
    # Integrate down the column to find dry pressure values, starting with the top
    # moist pressure value.
    p_dry[len(z)-1] = p_moist[len(z)-1]
    for x in reversed(range(len(z)-1)):
        p_dry[x] = integrate_dry_wrf(p_top=p_dry[x+1], rho_top=rho[x+1], 
                                     rho_bottom=rho[x], dz=z[x+1]-z[x])
    
    # Return values at input z heights including the surface.
    profile = xarray.Dataset(
        data_vars={
            'z': z,
            'q_v': (['z'], q_v),
            'theta': (['z'], theta),
            'rho': (['z'], rho),
            'p': (['z'], p_moist),
            'p_dry': (['z'], p_dry)
        })
   
    profile.z.attrs = {'name': 'Height', 'units': 'm'}
    profile.q_v.attrs = {'name': 'Water vapour mixing ratio', 'units': 'kg kg-1'}
    profile.theta.attrs = {'name': 'Dry potential temperature', 'units': 'K'}
    profile.rho.attrs = {'name': 'Density', 'units': 'kg m-3'}
    profile.p.attrs = {'name': 'Moist hydrostatic pressure', 'units': 'hPa'}
    profile.p_dry.attrs = {'name': 'Dry hydrostatic pressure', 'units': 'hPa'}
    
    # Print verbose output if required.
    if verbose:
        if np.all(q_v == 0):
            print('Dry WRF profile from sounding:')
        else:
            print('Moist WRF profile from sounding:')
        print('Surface pressure:\t\t\t' + str(p_s) + ' hPa')
        print('Surface temperature:\t\t\t' + str(T_s) + ' K')
        print('Surface mixing ratio:\t\t\t' + str(q_v[0]) + ' kg kg-1')
        print('Surface potential temperature:\t\t' + str(theta_s) + ' K')
        print('Surface density:\t\t\t' + str(rho[0]) + ' kg m-3')
    
    return(profile)

def interpolate(v, z, h, verbose=True):
    """
    Interpolate or extrapolate values.
    
    Arguments:
    v: Values of variable to interpolate, at each height.
    z: Height or pressure for each profile value [m].
    h: Height or pressure at which to interpolate [m].
    """

    increasing = None
    if np.all(z[1:] > z[:-1]):
        increasing = True
    elif np.all(z[1:] < z[:-1]):
        increasing = False
        
    assert increasing is not None, 'z must be monotone increasing or decreasing.'
    assert len(z) == len(v), 'z and v must have equal lengths.'
    
    if (increasing and h > z[len(z)-1]) or ((not increasing) and h < z[len(z)-1]): 
        # Extrapolate above profile.
        w2 = (z[len(z)-1] - h) / (z[len(z)-1] - z[len(z)-2])
        w1 = 1 - w2
        return(w1*v[len(z)-1] + w2*v[len(z)-2])

    elif (increasing and h < z[0]) or ((not increasing) and h > z[0]): 
        # Extrapolate below profile.
        w2 = (z[1] - h) / (z[1] - z[0])
        w1 = 1 - w2
        return(w1*v[1] + w2*v[0])

    else:
        # Interpolate within profile.
        if increasing:
            k = np.searchsorted(z, h) 
            kp = k-1
        else:
            k = len(z) - np.searchsorted(np.flip(z), h) - 1
            kp = k+1

        assert z[kp] <= h and z[k] >= h, 'Error in searchsorted result' 
        w2 = (z[k]-h) / (z[k] - z[kp])
        w1 = 1 - w2
        return(w1*v[k] + w2*v[kp])

def base_state_pressures(p, z, ztop, verbose=True, unit='hPa'):
    """
    Interpolate pressure profile to find model-top and surface pressures and 
    column mass.
    
    Arguments:
    p: Pressure profile.
    z: Heights for each profile value [m].
    ztop: Model-top height [m].
    verbose: Print output information? (Default: True).
    unit: Unit for display.
    """
    
    top_p = interpolate(v=p, z=z, h=ztop)
    surface_p = interpolate(v=p, z=z, h=0)
    p_c = surface_p - top_p
    
    if(verbose):
        print('Interpolated model-top pressure:\t' + str(top_p) + ' ' + unit)
        print('Interpolated surface pressure:\t\t' + str(surface_p) + ' ' + unit)
        print('Column mass:\t\t\t\t' + str(p_c) + ' ' + unit)
    
    return(top_p, surface_p, p_c)
       
def base_state_geopotential(eta, dry_profile, moist_profile, p_surface_dry, p_surface_moist, p_top, p_0=100000, verbose=True):
    """
    Use the WRF method to calculate base-state (dry) and perturbation (moist) geopotential [m2 s-2] 
    for full eta-levels. 
    
    NOTE: Differences between output from this function and WRF's values of PHB are attributed 
    to floating point differences because python uses 64 bit floats and WRF uses 32 bit floats.
    
    Geopotential (phi) is calculated for each full level; ie a phi value for each value of eta.
   
    Arguments:
    eta: eta-level values for which to find geopotential [-] (full levels).
    dry_profile: Dry sounding profile returned from compute_wrf_profile() with dry=True.
    moist_profile: Full sounding profile returned from compute_wrf_profile() with dry=False.
    p_surface_dry: Dry surface pressure based on the dry profile [hPa].
    p_surface_moist: Dry surface pressure based on the moist profile [hPa].
    p_top: Dry model-top pressure [hPa].
    p_0: Reference sea-level pressure [Pa].
    verbose: Print output information? (Default: True).
    
    Note that p_level_moist and p_surface_moist both refer to *dry* hydrostatic pressures,
    but based on the dry pressures calculted in the moist profile. 
    """
   
    assert len(eta) >= 2, 'Requires multiple full eta values.'
 
    # Convert pressures to Pa. From here on in this function, work in Pa.
    p_surface_dry = p_surface_dry * 100
    p_surface_moist = p_surface_moist * 100
    p_top = p_top * 100
    
    # Find dry pressure for each half eta level.
    eta_half = (eta[1:] + eta[:-1]) / 2
    p_level_dry = pressure_for_eta(eta=eta_half, p_surface=p_surface_dry, p_top=p_top)
    p_level_moist = pressure_for_eta(eta=eta_half, p_surface=p_surface_moist, p_top=p_top)

    rho_dry = np.zeros(len(eta_half))
    theta = np.zeros(len(eta_half))
    q_v = np.zeros(len(eta_half))
    pert_alpha = np.zeros(len(eta_half))
    rho_moist = np.zeros(len(eta_half))
    pert_pressure = np.zeros(len(eta_half))
    phi = np.zeros(len(eta))
    phi_pert = np.zeros(len(eta))
 
    # Calculate column masses.
    column_mass_dry = p_surface_dry - p_top
    column_mass_moist = p_surface_moist - p_top
    column_mass_perturb = column_mass_moist - column_mass_dry

    if verbose:
        print('Model-top pressure:\t\t' + str(p_top) + ' Pa')
        print('Dry column mass:\t\t' + str(column_mass_dry) + ' Pa')
        print('Moist column mass:\t\t' + str(column_mass_moist) + ' Pa')
        print('Perturbation column mass:\t' + str(column_mass_perturb) + ' Pa')
        
    for k in range(len(eta_half)):
        # Calculate dry density at each half eta level.
        theta_dry = interpolate(v=dry_profile.theta.values, z=dry_profile.p_dry.values*100, h=p_level_dry[k])
        rho_dry[k] = atm.density(p=p_level_dry[k]/100, theta=theta_dry, q_v=0)

        # Interpolate theta and q_v at each half eta level.
        theta[k] = interpolate(v=moist_profile.theta.values, z=moist_profile.p_dry.values*100, h=p_level_moist[k])
        q_v[k] = interpolate(v=moist_profile.q_v.values, z=moist_profile.p_dry.values*100, h=p_level_moist[k])
        
    # Calculate perturbation (including moisture) geopotentials, down the column.
    for x in reversed(range(len(eta_half))):
        if(x == len(eta_half)-1):
            q_v_half = q_v[x]
            delta_eta = eta[x+1] - eta[x]
            pert_pressure[x] = -0.5*(column_mass_perturb+(q_v_half/(1+q_v_half))*
                                    (column_mass_dry))/(1/delta_eta)/(1/(1+q_v_half))
        else:
            delta_eta = eta_half[x+1] - eta_half[x]
            q_v_half = (q_v[x] + q_v[x+1])/2
            
            pert_pressure[x] = pert_pressure[x+1] - (column_mass_perturb+(q_v_half/(1+q_v_half))*
                                                     (column_mass_dry))/(1/delta_eta)/(1/(1+q_v_half))
        
        rho_moist[x] = atm.density(p=(p_level_dry[x]+pert_pressure[x])/100, theta=theta[x], q_v=q_v[x])
        pert_alpha[x] = (1/rho_moist[x]) - (1/rho_dry[x])
   
    # Up the column to calculate geopotentials.
    for x in range(1, len(eta)):
        # Base-state (dry) geopotential.
        phi[x] = phi[x-1] - (eta[x] - eta[x-1]) * column_mass_dry * (1/rho_dry[x-1])
   
        # Perturbation (moist) geopotential.
        phi_pert[x] = phi_pert[x-1] - (eta[x] - eta[x-1]) * (
            (column_mass_dry + column_mass_perturb) * pert_alpha[x-1] +
            (column_mass_perturb * 1/rho_dry[x-1]))
    
    return(phi, phi_pert)
    
def heights(eta, phi_base, phi_pert, g=9.81):
    """
    Calculate Z_BASE heights [m] from base and perturbation geopotentials.
    
    Arguments:
    eta: eta value for each value of phi_base and phi_pert.
    phi_base: Base geopotential [m2 s-2].
    phi_pert: Perturbation geopotential [m2 s-2].
    g: Gravitational acceleration [m s-2].
    """
    
    assert len(eta) == len(phi_base), 'Expecting equal array lengths for eta and phi.'
    assert len(eta) == len(phi_pert), 'Expecting equal array lengths for eta and phi.'
    
    eta_half = (eta[1:] + eta[:-1]) / 2
    
    z = (phi_base[1:]+phi_base[:-1])/2 + (phi_pert[1:]+phi_pert[:-1])/2
    z = z/g
    
    # Return values at input z heights including the surface.
    heights = xarray.Dataset(
        data_vars={
            'eta': eta_half,
            'z_base': ((phi_base[1:]+phi_base[:-1])/2)/g,
            'z_full': ((phi_base[1:]+phi_base[:-1])/2 + (phi_pert[1:]+phi_pert[:-1])/2)/g
        })
   
    heights.eta.attrs = {'name': 'Half-level eta', 'units': '-'}
    heights.z_base.attrs = {'name': 'Base geopotential height', 'units': 'm'}
    heights.z_full.attrs = {'name': 'Base+perturbation geopotential height', 'units': 'm'}
    
    return(heights)
    
def destagger_etas(half):
    """
    Destagger staggered eta values defined at mass point to get full-level values.
    Produce an error if eta values that are monotonically decreasing cannot be found.
    
    Arguments:
    half: Staggered eta values.
    """
    
    etas = np.ones(len(half)+1)

    # Iterate from surface to top, making the first value 1 for the surface.
    for x in range(1,len(etas)):
        etas[x] = etas[x-1] - 2*(etas[x-1] - half[x-1])
        
        if x < len(half):
            assert etas[x] > half[x], 'Impossible to create monotonically decreasing etas.'

    return(etas)

def compare_arrays(x, ref, unit, digits=2):
    """
    Show a simple comparison of two np.arrays.
    
    Arguments:
    x: Values to test against the reference.
    ref: Reference values.
    unit: The unit of x and ref values.
    digits: Number of digits to round results to.
    """
    
    diffs = x - ref
    rel_diffs = diffs / ref * 100
    
    print('Maximum absolute difference:\t\t' + str(np.round(np.max(np.abs(diffs)), digits)) + ' ' + unit)
    print('Bias:\t\t\t\t\t' + str(np.round(np.mean(diffs), digits)) + ' ' + unit)
    print('Maximum relative absolute difference:\t' + 
          str(np.round(np.max(np.abs(rel_diffs)), digits)) + '%')
    
def pressure_for_eta(eta, p_surface, p_top):
    """
    Return the dry hydrostatic pressure for a given eta level in the same units 
    surface/top pressures.
    
    Arguments:
    eta: Eta value(s) to calculate pressure for.
    p_surface: The dry surface pressure.
    p_top: The dry model-top pressure.
    """

    return(eta * (p_surface - p_top) + p_top)
        
def eta_for_pressure(p_dry, p_top):
    """
    For a pressure profile with values at half-eta heights, determine the full-eta values that 
    correspond to each sounding point.
    
    Arguments:
    p_dry: Dry hydrostatic pressure per half eta level from the surface upwards (hPa).
    top_p_dry: Model-top pressure [hPa].
    """

    eta_half = (p_dry[1:] - p_top) / (p_dry[0] - p_top)
    eta = destagger_etas(eta_half)
    
    assert np.all(eta[1:] - eta[:-1] < 0), "eta values are not monotonic decreasing."
    return(eta)

def write_input_sounding(out_dir, surface_p, surface_T, surface_q, profile, indexed=True):
    """
    Write an input_sounding file to a given output directory.
    
    Arguments:
    out_dir: Output directory.
    surface_p: Surface pressure [hPa].
    surface_T: Surface temperature [K].
    surface_q: Surface specific humidity [kg kg-1].
    profile: Profile to write, must contain z, theta, q (U and V are (re)set to zeros).
    indexed: Is the profile indexed (argument 'index' to pandas.to_csv().
    """
    
    assert not os.path.exists(out_dir + '/input_sounding'), 'Output file already exists.'
    
    # Calculate surface potential temperature.
    # Note R_d constant value used here is from RCEMIP specification (Wing et al 2018), not WRF.
    surface_theta = atm.potential_temp(T=surface_T, p=surface_p, R_d=287.04)
    
    # Write surface information to the first line. q is converted from kg kg-1 to g kg-1.
    surf = pd.DataFrame(data={'p': [surface_p], 'theta': [surface_theta], 'q': [surface_q*1000]})
    surf.to_csv(out_dir+'/input_sounding', sep='\t', header=False, index=False)
    
    # Append profile.
    sounding_file = open(out_dir+'/input_sounding', mode='a', newline='')
    prof = profile[['z', 'theta', 'q']]
    
    # Convert q from kg kg-1 to g kg-1.
    assert prof.q.attrs['units'] == 'kg kg-1'
    prof['q'] = prof.q * 1000
    
    prof['U'] = 0
    prof['V'] = 0
    prof.to_dataframe().to_csv(sounding_file, sep='\t', header=False, index=indexed)
    sounding_file.close()

