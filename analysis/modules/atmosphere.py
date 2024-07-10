### Functions to compute basic atmospheric properties. ###

import xarray
import numpy as np

def density(p, theta, q_v, R_d=287, R_v=461.6, p_0=100000):
    """
    Calculate density rho [kg m-3] using Equation 5.2 from the WRF tech note.
    
    Arguments:
    p: Pressure [hPa].
    theta: Dry potential temperature [K].
    q_v: Mixing ratio for water vapour [kg kg-1].
    p_0: Reference sea-level pressure [Pa].
    R_d: Gas constant for dry air [J kg-1 K-1].
    R_v: Gas constant for water vapour [J kg-1 K-1].
    """
    
    p = p * 100 # Convert pressure from hPa to Pa.
    c_p = 7 * R_d/2 # Specific heat of dry air at constant pressure [J kg-1 K-1].
    c_v = c_p - R_d # Specific heat of dry air at constant volume [J kg-1 K-1].
    
    # Calculate moist potential temperature.
    theta_m = theta * (1 + (R_v/R_d)*q_v)
    
    # Calculate density.
    rho = 1 / (((R_d * theta_m) / p_0) * (p/p_0)**(-c_v/c_p)) 
    return(rho)

def saturation_vapour_pressure(T, T_0=273.15, over='water', method='wexler'):
    """
    Calculate saturatino vapour pressure over water or ice using a specified method.
    
    Arguments:
    T: Temperature(s) [K] for which to calculate the saturation vapour pressure.
    T_0: Conversion factor for Kelvin to Celsius (zero C in K).
    over: Calculate the saturation vapour pressure over 'water' or 'ice'. 
    method: Method to use; one of 'bolton', 'wexler', 'murphy', or 'flatau'.
    """
    
    method = method.lower() 
    assert (method == 'bolton' or method == 'wexler' or
            method == 'flatau' or method == 'murphy'), 'invalid method name.'
    
    if method == 'bolton':
        es = saturation_vapour_pressure_bolton(T, T_0=T_0, over=over)
    elif method == 'wexler':
        es = saturation_vapour_pressure_wexler(T, over=over)
    elif method == 'murphy':
        es = saturation_vapour_pressure_murphy(T, over=over)
    elif method == 'flatau':
        es = saturation_vapour_pressure_flatau(T, T_0=T_0, over=over)
        
    if isinstance(es, xarray.DataArray):
        es.attrs = {'long_name': 'Saturation vapour pressure', 'units': 'hPa'}
    return(es)
    
def saturation_vapour_pressure_bolton(T, T_0=273.15, over='water'):
    """
    Calculate saturation vapour pressure over water, using the expression of Bolton 1980, 
    equation 10. This equation is accurate within 0.3% for 238.15 <= T <= 308.15.
    
    Arguments:
    T: Temperature(s) [K] for which to calculate the saturation vapour pressure.
    T_0: Conversion factor for Kelvin to Celsius (zero C in K).
    over: Calculate the saturation vapour pressure over 'water' or 'ice'. 
    """
    
    assert over == 'water', 'saturation_vapour_pressure_bolton: \'over\' must be \'water\'.'          
                
    if np.nanmin(T) < 238.15 or np.nanmax(T) > 308.15:
        print('saturation_vapour_pressure_bolton: warning T outside valid range.')
        
    es = 6.112 * np.exp(17.67*(T - T_0)/(T - T_0 + 243.5))
    return(es)
    
def saturation_vapour_pressure_flatau(T, T_0=273.15, over='water', const=True):
    """ 
    Calculate saturation vapour pressure over water or ice, using the polynomial fits 
    of Flatau et al. 1992 (Table 4, right-hand column). Polynomial fits valid over water
    for 188.15 <= T <= 343.15 K and over ice for 183.15 <= T <= 273.15 K.
    
    A constant value of saturation vapour pressure is returned for temperatures below -80C
    over water, and -90C over ice. Flatau states their water relationship is valid down to 
    -85C, but the returned SVP goes negative around -84C, so here the minimum is set to -80C 
    for water (this matches with the approach taken in code for the Morrison two-moment 
    microphysics scheme in WRF which uses -80C for both ice and water).
    
    Arguments:
    T: Temperature(s) [K] for which to calculate the saturation vapour pressure.
    T_0: Conversion factor for Kelvin to Celsius (zero C in K).
    over: Calculate the saturation vapour pressure over 'water' or 'ice'. 
    const: Truncate temperature values at low temperatures? (Default: True).
    """
    
    assert over == 'water' or over == 'ice', 'parameter \'over\' must be \'water\' or \'ice.\''
    
    # Convert K to deg. C.
    T_C = T-T_0
    
    if over == 'water':
        if np.nanmin(T) < 188.15 or np.nanmax(T) > 343.15:
            print('saturation_vapour_pressure_flatau: warning T outside valid range.')
        
        coefs = [6.11239921, 
                 0.443987641,
                 0.142986287e-1,
                 0.264847430e-3,
                 0.302950461e-5,
                 0.206739458e-7,
                 0.640689451e-10,
                 -0.952447341e-13,
                 -0.976195544e-15]
        
        if const:
            T_C = np.maximum(-80, T_C)

    elif over == 'ice':
        if np.nanmin(T) < 183.15 or np.nanmax(T) > 273.15:
            print('saturation_vapour_pressure_flatau: warning T outside valid range.')
        
        coefs = [6.11147274,
                 0.503160820,
                 0.188439774e-1,
                 0.420895665e-3,
                 0.615021634e-5,
                 0.602588177e-7,
                 0.385852041e-9,
                 0.146898966e-11,
                 0.252751365e-14]
        
        if const:
            T_C = np.maximum(-90, T_C)
        
    es = np.polynomial.polynomial.polyval(T_C, coefs)
    return(es)
    
def saturation_vapour_pressure_wexler(T, over='water'):
    """
    Calculate saturation vapour pressure [hPa] over water or ice, using the Wexler formation of 
    Wexler 1976, 1977, as it is stated in Flatau 1992 (Equations 2.1 and 2.2 and 
    coefficients in Table 1).
    
    Arguments:
    T: Temperature(s) [K] for which to calculate the saturation vapour pressure.
    over: Calculate the saturation vapour pressure over 'water' or 'ice'.
    """
    
    assert over == 'water' or over == 'ice', 'parameter \'over\' must be \'water\' or \'ice.\''
    
    if over == 'water':
        g0 = -0.29912729e4
        g1 = -0.60170128e4
        g2 = 0.1887643854e2
        g3 = -0.28354721e-1
        g4 = 0.17838301e-4
        g5 = -0.84150417e-9
        g6 = 0.44412543e-12
        g7 = 0.2858487e1
        
        es = np.exp((g0 + 
                     (g1 + (g2 + g7*np.log(T) +
                            (g3 + (g4 + (g5 + g6*T)*T)*T)*T)*T)*T) / T**2)
        
    if over == 'ice':
        k0 = -0.58653696e4
        k1 = 0.2224103300e2
        k2 = 0.13749042e-1
        k3 = -0.34031775e-4
        k4 = 0.26967687e-7 
        k5 = 0.6918651e0
        
        es = np.exp((k0 + (k1 + k5*np.log(T) + 
                          (k2 + (k3 + k4*T)*T)*T)*T)/T)
        
    # Convert Pa to hPa.
    es = es / 100
    return(es)

def saturation_vapour_pressure_murphy(T, over='water'):
    """
    Calculate saturation vapour pressure [hPa] over water or ice, using the extended temperature 
    formulations of Murphy and Koop 2005 (Equation 10 for over water and Equation 7 for over 
    ice).
    
    Arguments:
    T: Temperature(s) [K] for which to calculate the saturation vapour pressure.
    over: Calculate the saturation vapour pressure over 'water' or 'ice'.
    """
    
    assert over == 'water' or over == 'ice', 'parameter \'over\' must be \'water\' or \'ice.\''
    
    if over == 'water':
        if np.nanmin(T) <= 123 or np.nanmin(T) >= 332:
            print('saturation_vapour_pressure_murphy: warning T outside valid range.')
        
        es = np.exp(54.842763 - 6763.22/T - 4.210*np.log(T) + 0.000367*T + 
                     np.tanh(0.0415*(T - 218.8))*
                     (53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T))
     
    if over == 'ice':
        if np.nanmin(T) <= 110:
            print('saturation_vapour_pressure_murphy: warning T outside valid range.')
        
        es = np.exp(9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T)
            
    # Convert Pa to hPa.
    es = es / 100
    return(es)

def temp_from_theta(theta, p, R=287, P_0=1000):
    """
    Calculate temperature [K] from potential temperature.
    
    Arguments:
    theta: Potential temperature [K].
    p: Pressure [hPa].
    R: Gas constant for dry air [J kg-1 K-1].
    P_0: Reference pressure [hPa].
    """
    
    # Specific heat of dry air at constant pressure [J kg-1 K-1].              
    Cp = 7*R/2 
    
    temp = theta / ((P_0/p)**(R/Cp))
    
    if isinstance(temp, xarray.DataArray):
        temp.attrs = {'long_name': 'Temperature', 'units': 'K'}
        
    return(temp)

def potential_temp(T, p, p_0=1000, R_d=287):
    """
    Calculate potential temperature using Poisson's equation.
    
    Arguments:
    
    T: Temperature [K].
    p: Pressure [hPa].
    p_0: Reference pressure [hPa].
    R_d: Gas constant for dry air [J kg-1 K-1].
    """
    
    c_p = 7 * R_d/2 # Specific heat of dry air at constant pressure [J kg-1 K-1].
    theta = T * (p_0/p)**(R_d/c_p)
    return(theta)

def specific_humidity(es, p):
    """
    Calculate saturation specific humidity [kg kg-1].
    
    Arguments:
    es: Vapour pressure [hPa].
    p: Pressure [hPa].
    """
    
    qs = 0.622 * es/(p - 0.378*es)
    
    if isinstance(qs, xarray.DataArray):
        qs.attrs = {'long_name': 'Saturation specific humidity', 'units': 'kg kg-1'}

    return(qs)        
        
def saturation_mixing_ratio(es, p):
    """
    Calculate saturation mixing ratio [kg kg-1].
    
    Arguments:
    es: Saturation vapour pressure [hPa].
    p: Pressure [hPa].
    """
    
    ws = 0.622 * es/(p-es)
    
    if isinstance(ws, xarray.DataArray):
        ws.attrs = {'long_name': 'Saturation water vapour mixing ratio', 'units': 'kg kg-1'}
       
    return(ws)

def relative_humidity(theta, p, q, es_method='murphy'):
    """
    Calculate relative humidity [%].
    
    Arguments:
    theta: Potential temperature [K].
    p: Pressure [hPa].
    q: Water vapour mixing ratio [kg kg-1].
    es_method: Method to use for calculation of saturation vapour pressure.
    """
    
    T = temp_from_theta(theta=theta, p=p)              
    es = saturation_vapour_pressure(T=T, over='water', method=es_method)
    ws = saturation_mixing_ratio(es=es, p=p)
    
    rh = q/ws * 100
    
    if isinstance(rh, xarray.DataArray):
        rh.attrs = {'long_name': 'Relative humidity', 'units': '%'}
        
    return(rh)
