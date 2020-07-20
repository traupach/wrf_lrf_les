# wrf_perturbation.py
# 
# Functions for analysing output of WRF perturbation experiments.
#
# Author: T. Raupach <t.raupach@unsw.edu.au>

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap
import xarray
import glob
import wrf

FIGURE_SIZE = [15, 5] # Default figure size [horizontal, vertical]. 

def read_wrfvars(inputs):
    """
    Read all wrfvars files in multiple datasets.
    
    Arguments:
        inputs: A dictionary containing dataset names as keys and input directories as values.
        
    Returns: an xarray.DataArray with all data from wrfvars*.nc files, labelled by dataset.
    """
    
    datasets = []

    for setname, directory in inputs.items():
        datasets.append(xarray.open_mfdataset(directory+"/wrfvars*.nc", combine="nested", concat_dim="time"))
        datasets[-1]['Dataset'] = setname

    dat = xarray.combine_nested(datasets, concat_dim='Dataset')
    return(dat)

def analyse_wrfinput(wrfinput_file, sounding_file=None, ideal=True, plot_profiles=True):
    """
    Print information to show basic setup options stored in a wrfinput file, and show summary plots of input profiles.
    
    Arguments:
        wrfinput_file: The file name to analyse.
        ideal: If true, check ideal-case assumptions.
        plot_f: Plot T, QV and P profiles.
        sounding_file: If specified, a sounding file to pass to plot_wrfinput_profiles().
    """
    
    wrfin = xarray.open_dataset(wrfinput_file)

    if ideal:
        check_wrfinput_ideal(wrfin=wrfin)
    
    # Calculate total geopotential height (staggered).
    hgt = ((wrfin.PH + wrfin.PHB) / 9.81).isel(Time=0).mean(['south_north', 'west_east'])
    
    # Print important initialisation values.
    print('Surface skin temperature (TSK):\t\t\t' + str(np.unique(wrfin.TSK)[0]) + ' ' + wrfin.TSK.attrs['units'])
    print('Soil temperature at lower boundary (TMN):\t' + str(np.unique(wrfin.TMN)[0]) + ' ' + wrfin.TMN.attrs['units'])
    print('Sea surface temperature (SST):\t\t\t' + str(np.unique(wrfin.SST)[0]) + ' ' + wrfin.SST.attrs['units'])
    print('Horizontal grid spacing (DX):\t\t\t' + str(wrfin.DX) + ' m')
    print('Horizontal (S-N) grid spacing (DY):\t\t\t' + str(wrfin.DY) + ' m')
    print('Horizontal (W-E) domain size:\t\t\t' + str(wrfin.attrs['WEST-EAST_GRID_DIMENSION']-1) + ' mass points')
    print('Horizontal (S-N) domain size:\t\t\t' + str(wrfin.attrs['SOUTH-NORTH_GRID_DIMENSION']-1) + ' mass points')
    print('Vertical domain size:\t\t\t\t' + str(wrfin.attrs['BOTTOM-TOP_GRID_DIMENSION']-1) + ' mass points')
    print('Maximum base-state height (mass point):\t\t' + str(np.round(wrfin.Z_BASE.isel(Time=0).max().data, 1)) + ' m')
    print('Maximum geopotential height (top of grid):\t' + str(np.round(hgt.max().data, 1)) + ' m')
    print('Model-top pressure:\t\t\t\t' + str(np.round(wrfin.P_TOP.isel(Time=0).data, 1)) + ' ' + wrfin.P_TOP.attrs['units'])
    print('Coriolis sine latitude term (F):\t\t' + str(np.unique(wrfin.F)[0]) + ' ' + wrfin.F.attrs['units'])
    print('Coriolis cosine latitude term (E):\t\t' + str(np.unique(wrfin.E)[0]) + ' ' + wrfin.E.attrs['units'])
    print('Use light nudging on U and V:\t\t\t' + str(wrfin.USE_LIGHT_NUDGING))
    print('Ideal evaporation:\t\t\t\t' + str(wrfin.IDEAL_EVAP_FLAG))
    print('Surface wind for fluxes:\t\t\t' + str(wrfin.SURFACE_WIND) + ' m s-1')
    print('Constant radiative cooling profile:\t\t' + str(wrfin.CONST_RAD_COOLING))
    print('Relax U and V to set profiles?\t\t\t' + str(wrfin.RELAX_UV_WINDS))
    if wrfin.RELAX_UV_WINDS > 0:
        print('Wind relaxation time:\t\t\t\t' + str(wrfin.WIND_RELAXATION_TIME) + ' s')
    print('Physics schemes:')
    print('\tMicrophysics:\t\t\t\t' + wrf_mp_scheme(wrfin))
    print('\tRadiation (longwave):\t\t\t' + wrf_ra_lw_scheme(wrfin))
    print('\tRadiation (shortwave):\t\t\t' + wrf_ra_sw_scheme(wrfin))
    print('\tSurface layer:\t\t\t\t' + wrf_sf_sfclay_scheme(wrfin))
    print('\tLand-surface:\t\t\t\t' + wrf_sf_surface_scheme(wrfin))
    print('\tPBL:\t\t\t\t\t' + wrf_pbl_scheme(wrfin))
    print('\tCumulus:\t\t\t\t' + wrf_cu_scheme(wrfin))
    
    if plot_profiles:
        plot_wrfinput_profiles(wrfin=wrfin, sounding_file=sounding_file)

def perturbation_details(wrfinput_file, comparison_file=None):
    """
    Print perturbation settings stored in a WRF input file.
    
    Arguments:
        wrfinput_file: The wrfinput file to open.
        comparison_file: If specified, throw an error if the wrfinput file does not match 
                         comparison_file except in perturbation details and simulated times.
        
    Return: the value of K_PERT, the level index to which perturbations would be/were applied.
    """

    if not comparison_file is None:
        if not nc_equal(wrfinput_file, comparison_file, ignore_fields=['PERTURB_T','PERTURB_Q',
                                                                        'K_PERT','TTENDAMP',
                                                                        'QTENDAMP','Times']):
            print('Error: perturbed input file does not match comparison input file.')
            return(None)
    
    pert_in = xarray.open_dataset(wrfinput_file)
    print('Perturb temperature T:\t\t\t' + str(pert_in.PERTURB_T))
    print('Perturb moisture q:\t\t\t' + str(pert_in.PERTURB_Q)) 
    print('Perturbation vertical level (1-based):\t' + str(pert_in.K_PERT))
    print('Perturbation T tendency amplitude:\t' + str(pert_in.TTENDAMP) + ' K day-1')
    print('Perturbation Q tendency amplitude:\t' + str(pert_in.QTENDAMP) + ' kg kg-1 day-1')
    
    k_pert = pert_in.K_PERT-1
    print('Value of k_pert returned (0-based):\t' + str(k_pert))
 
    return(k_pert)

def plot_wrfinput_profiles(wrfin, sounding_file=None):
    """
    Plot grid-mean temperature, water vapour mixing ratio, and pressure by height in a facetted plot.
    
    Arguments:
        wrfin: The wrfinput file as an xarray.DataArray.
        sounding_file: If specified, a sounding file (text) to plot for comparison. Should contain
                       on the first line "base_P base_T base_QV" then on next lines 
                       "Z P T QV U V". 
    """
    
    base_height = wrfin.Z_BASE.isel(Time=0) / 1000
    fig, ax = plt.subplots(ncols=4)

    # Potential temperature profile.
    potential_temp = wrfin.T.isel(Time=0).mean(['south_north', 'west_east']) + 300 
    ax[0].plot(potential_temp, base_height, label='WRF input')
    ax[0].set_xlabel('Total potential temperature [K]')
    ax[0].set_ylabel('Base state height [km]')

    # Water vapour mixing ratio (converted from kg kg-1 to g kg-1).
    ax[1].plot(wrfin.QV_BASE.isel(Time=0)*1000, base_height, label='WRF input')
    ax[1].set_xlabel('Water vapour mixing ratio [g kg-1]')
       
    # Wind U and V.
    U = wrfin.U.isel(Time=0).mean(['south_north', 'west_east_stag'])
    V = wrfin.V.isel(Time=0).mean(['south_north_stag', 'west_east'])
    ax[2].plot(U, base_height, label='WRF input U')
    ax[2].set_xlabel('Wind speed [m s-1]')
    ax[2].plot(V, base_height, label='WRF input V')
    ax[2].set_xlabel('Wind speed V [m s-1]')
    ax[2].legend()
    
    # Full pressure.
    pressure = (wrfin.P + wrfin.PB).isel(Time=0).mean(['south_north', 'west_east']) / 100
    ax[3].plot(pressure, base_height, label='WRF input')
    ax[3].set_xlabel('Full pressure [hPa]')
    
    for i in np.arange(1, len(ax)):
        ax[i].tick_params(labelleft=False)
    
    # Plot comparison sounding profiles 
    if not sounding_file is None:
        prof = pd.read_table(sounding_file, header=None, skiprows=1, sep=r'\s+')
        prof.columns = ['Z', 'T', 'QV', 'U', 'V']
        prof['Z'] = prof['Z'] / 1000 
        
        ax[0].plot(prof['T'], prof['Z'], label='Sounding')
        ax[1].plot(prof['QV'], prof['Z'], label='Sounding')
        ax[2].plot(prof['U'], prof['Z'], label='Sounding U')
        ax[2].plot(prof['V'], prof['Z'], label='Sounding V')
        
        for i in np.arange(0, len(ax)):
            ax[i].legend()
        
    plt.tight_layout()
    plt.show()
    
def check_wrfinput_ideal(wrfin):
    """
    Check ideal-case assumptions - flat water domain with fixed surface temp and no 
    coriolis effect - on a wrfinput file. 
    
    Arguments:
        wrfin: The wrfinput file as an xarray.DataArray.
    """
   
    assert np.all(wrfin.LU_INDEX == wrfin.ISWATER), 'Domain must be all water.'
    assert len(np.unique(wrfin.TSK)) == 1, 'TSK must be constant.'
    assert len(np.unique(wrfin.TMN)) == 1, 'TMN must be constant.'
    assert len(np.unique(wrfin.SST)) == 1, 'SST must be constant.'
    assert np.unique(wrfin.HGT) == 0, 'Height must be constant zero.'
    assert np.unique(wrfin.F) == 0, 'Coriolis sine term must be constant zero.'
    assert np.unique(wrfin.E) == 0, 'Coriolis cosine term must be constant zero.'
    
    ## Ensure that mean QV, U, and V are the same as QV_BASE, U_BASE, and V_BASE respectively.
    mean_qvapour = wrfin.QVAPOR.isel(Time=0).mean(['south_north','west_east'])
    mean_U = wrfin.U.isel(Time=0).mean(['south_north','west_east_stag'])
    mean_V = wrfin.V.isel(Time=0).mean(['south_north_stag','west_east'])
    
    assert np.max(np.abs(mean_qvapour - wrfin.QV_BASE)) < 1e-5, 'Grid-mean QVAPOR must equal QV_BASE.'
    assert np.max(np.abs(mean_U - wrfin.U_BASE)) < 1e-5, 'Grid-mean U must equal U_BASE.'
    assert np.max(np.abs(mean_V - wrfin.V_BASE)) < 1e-5, 'Grid-mean V must equal V_BASE.'
    
def diff_means(dat, control_start, control_end, other_start, other_end, control_name='Control'):
    """
    Select data within a time period, take the mean control and mean perturb values, and return 
    differences as perturb - control.
    
    Arguments:
        dat: The data (xarray.DataArray). 
        control_start, control_end: Start and end times to select for control dataset (used as a slice).
        other_start, other_end: Start and end time slice for all other datasets.
        control_name: Name for control values of 'Dataset' in dat.  
        
    Returns: perturb - control differences in temporal means.
    """
    
    control = dat.sel(Dataset=control_name, time=slice(control_start, control_end)).mean('time', keep_attrs=True)
    other = dat.drop_sel(Dataset=control_name).sel(time=slice(other_start, other_end)).mean('time', keep_attrs=True)
    
    def take_diff(x): 
        with xarray.set_options(keep_attrs=True):
            diffs = x - control
        return(diffs)
        
    diffs = other.groupby('Dataset').map(take_diff)
    return(diffs)

def keep_left_axis(axes, invert_y=False):
    """
    Loop through a set of axes, invert y if required, and turn off labels on all but the left-most axis.
    
    Arguments:
        axes: A list of axes.
        invert_y: Invert the y axes?
    """
    
    for i, ax in enumerate(axes):
        if invert_y: ax.invert_yaxis()
        if i > 0: ax.set_ylabel('')

def rewrap_labels(axes, length_x=25, length_y=45):
    """
    Re-linewrap plot x and y labels.
    
    Arguments:
        axes: List of plot axes.
        length_x: The length to wrap x labels at [chars].
        length_y: The length to wrap y labels at [chars].
    """
    
    for ax in axes:
        ax.set_xlabel('\n'.join(textwrap.wrap(ax.get_xlabel().replace('\n',' '), length_x)))
        ax.set_ylabel('\n'.join(textwrap.wrap(ax.get_ylabel().replace('\n',' '), length_y)))
            
def compare_profiles(dat, control_start, control_end, other_start, other_end, control_name='Control', 
                     variables=['tk','q','ua','va','rh']):
    """
    Find temporal means of water vapour mixing ratio (q) and temperature (tk), and plot the differences
    between them by pressure level.
    
    Arguments:
        dat: The data (xarray.DataArray). 
        control_start, control_end: Start and end times to select for control dataset (used as a slice).
        other_start, other_end: Start and end time slice for all other datasets.
        control_name: Name for control values of 'Dataset' in dat.  
        variables: The variables to compare.
    """
    
    diffs = diff_means(dat=dat[variables], control_start=control_start, control_end=control_end,
                       other_start=other_start, other_end=other_end, control_name=control_name)

    fig, ax = plt.subplots(ncols=len(variables), sharey=True)
    
    for i in range(len(variables)):
        diffs[variables[i]].plot(hue='Dataset', y='level', ax=ax[i], yincrease=False)
        ax[i].axvline(x=0, color='red')
        ax[i].set_title('')
        
    keep_left_axis(ax)
    rewrap_labels(ax)
    plt.suptitle('Differences in temporal means (perturbed - control)', y=1.02)
    plt.tight_layout()
    plt.show()
    
def plot_profiles(dat, control_start, control_end, other_start, other_end, control_name='Control', 
                  variables=['tk','q','ua','va','rh']):
    """
    Plot temporals means of water vapour mixing ratio (q) and temperature (tk) by Dataset.
    
    Arguments:
        dat: The data (xarray.DataArray). 
        control_start, control_end: Start and end times to select for control dataset (used as a slice).
        other_start, other_end: Start and end time slice for all other datasets.
        control_name: Name for control values of 'Dataset' in dat. 
        variables: The variables to plot.
    """
    
    profs = dat[variables]
    profs_control = profs.sel(Dataset=control_name, time=slice(control_start, control_end)).mean('time', keep_attrs=True)
    profs_other = profs.drop_sel(Dataset=control_name).sel(time=slice(other_start, other_end)).mean('time', keep_attrs=True)
    profs = xarray.concat([profs_control, profs_other], dim='Dataset')
    
    fig, ax = plt.subplots(ncols=5, sharey=True)
    
    for i in range(len(variables)):
        profs[variables[i]].plot(hue='Dataset', y='level', ax=ax[i], yincrease=False)
        
    keep_left_axis(ax)
    rewrap_labels(ax)
    plt.tight_layout()
    plt.show()
    
def get_OLR(inputs, patterns, file_idx=0, time_idx=1):
    """
    Read wrfout files ; load one file from the sorted list, and retrieve an outgoing longwave 
    radiation (OLR) field for the last time step. 
    
    Arguments:
        inputs: A dictionary containing dataset names as keys and input directories as values.
        patterns: A dictionary with dataset name/file pattern combinations.
        file_idx: the index of the file to read, from the list of matching files (default 0, first file).
        time_idx: the time index of the OLR field to read, backwards from the end of the file.
                  By default time_idx=1 and the last OLR field will be returned.
        
    Returns: The specified OLR fields labelled by dataset.
    """
    
    olr = []
    for setname, directory in inputs.items():
        file = sorted(glob.glob(directory+'/'+patterns[setname]))[file_idx]
        olr.append(xarray.open_dataset(file, decode_times=False).OLR.tail(Time=time_idx))
        olr[-1]['Dataset'] = setname
    return(xarray.combine_nested(olr, concat_dim='Dataset'))

def plot_OLRs(inputs, patterns):
    """
    Plot outgoing longwave radiation fields for each input. Patterns should match a single 
    file from which the last OLR field will be plotted.
    
    Variables:
        inputs: A dictionary with dataset name/output directory combinations.
        patterns: A dictionary with dataset name/file pattern combinations.
    """

    OLRs = get_OLR(inputs=inputs, patterns=patterns)

    p = OLRs.plot(col='Dataset', figsize=FIGURE_SIZE).set_titles('{value}')
    for ax in p.axes.flat:
        ax.set_aspect('equal')
    plt.show()

def get_wind(dat):
    """
    Calculate wind magnitude.
    
    Arguments:
        dat: Data set, must contain 'ua' and 'va' wind fields.
        
    Returns: wind magnitude fields.
    """
    
    wind = np.sqrt(dat.ua**2 + dat.va**2)
    assert dat.ua.units == dat.va.units
    wind.attrs['units'] = dat.ua.units
    wind.attrs['long_name'] = 'Wind magnitude'
    return(wind)

def pressure_at_kth_eta_level(wrfvars, k_pert, dataset='Perturbed'):
    """
    Plot and print the minimum and maximum pressure at a certain vertical level.
    
    Arguments:
        wrfvars: The data to plot, containing time, P_HYD_min and P_HYD_max.
        k_pert: The level to analyse (zero-based index in bottom_top).
        dataset: The dataset to subset to (default: 'Perturbed').
    """
    
    min_pres = wrfvars.P_HYD_min.sel(Dataset=dataset).isel(bottom_top=k_pert)/100
    max_pres = wrfvars.P_HYD_max.sel(Dataset=dataset).isel(bottom_top=k_pert)/100

    print('Pressure at the perturbed level (eta level 0-based index ' + str(k_pert) + ') ranged from ' + \
          str(np.round(min_pres.min().values,1)) + ' hPa to ' + \
          str(np.round(max_pres.max().values,1)) + ' hPa.')
    
    fig, ax = plt.subplots()
    ax.plot(wrfvars.time.values, min_pres, color='black', linewidth=0.2)
    ax.plot(wrfvars.time.values, max_pres, color='black', linewidth=0.2)
    ax.fill_between(wrfvars.time.data, min_pres, max_pres, facecolor='orange', alpha=.5)
    ax.set_ylabel('Hydrostatic pressure [hPa]')
    ax.set_xlabel('Simulation time')
    ax.invert_yaxis()
    plt.show()

def plot_radiative_cooling_profiles(dat):
    """
    Plot the radiative cooling profiles for each dataset.
    
    Arguments:
        dat: The data to plot, containing RTHRATEN and Dataset.
    """
    
    rthprofile = dat.RTHRATEN.mean('time', keep_attrs=True)
    
    # Convert from K s-1 to K day-1.
    assert rthprofile.attrs['units'] == 'K s-1', 'Unexpected units on RTHRATEN.'
    with xarray.set_options(keep_attrs=True):
        rthprofile = rthprofile * 86400 
    rthprofile.attrs['units'] = 'K day-1'
    
    rthprofile.plot(y='level', col='Dataset', figsize=FIGURE_SIZE, yincrease=False).set_titles('{value}')
    plt.show()
    
def plot_radiative_cooling_by_level(dat, plot_levels):
    """
    Plot the radiative cooling profiles for each dataset.
    
    Arguments:
        dat: The data to plot, containing RTHRATEN and Dataset.
        plot_levels: The pressure levels to show [hPa].
    """
    
    levdata = dat.RTHRATEN.isel(time=slice(1,len(dat.time))).sel(level=plot_levels)
    
    # Convert from K s-1 to K day-1.
    assert levdata.attrs['units'] == 'K s-1', 'Unexpected units on RTHRATEN.'
    with xarray.set_options(keep_attrs=True):
        levdata = levdata * 84600 
        
    levdata.plot(col='level', col_wrap=2, hue='Dataset', sharey=False,
                 figsize=FIGURE_SIZE).set_titles('{value} hPa')
    plt.show()
    
def nc_equal(file1, file2, ignore_fields=[]):
    """
    Test whether two NetCDF files are equal, while ignoring certain fields.
    
    Arguments:
        file1, file2: The two files to test.
        ignore_fields: Optional names of fields or attributes to ignore.
        
    Returns: whether file1 == file2 when ignoring specified fields.
    """
    nc1 = xarray.open_dataset(file1)
    nc2 = xarray.open_dataset(file2)

    for field in ignore_fields:
        if field in nc1.data_vars:
            nc1 = nc1.drop_vars(field)
        if field in nc2.data_vars:
            nc2 = nc2.drop_vars(field)
        if field in nc1.attrs:
            nc1.attrs.pop(field)
        if field in nc2.attrs:
            nc2.attrs.pop(field)

    return(nc1.equals(nc2))

def wrf_mp_scheme(wrfin):
    """
    Lookup the microphysics scheme information in a wrfinput file and return a description string.
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {1:  'Kessler',
               2:  'Purdue Lin',
               3:  'WSM3',
               4:  'WSM5',
               5:  'Eta (Ferrier)',
               6:  'WSM6',
               7:  'Goddard',
               8:  'Thompson',
               9:  'Milbrandt 2-moment',
               10: 'Morrison 2-moment',
               11: 'CAM 5.1',
               13: 'SBU-YLin',
               14: 'WDM5',
               16: 'WDM6',
               17: 'NSSL 2-moment',
               18: 'NSSL 2-moment with CCN prediction',
               19: 'NSSL 1-moment',
               21: 'NSSL 1-moment lfo',
               22: 'NSSL 2-moment without hail',
               28: 'Thompson aerosol- aware',
               30: 'HUJI SBM fast',
               32: 'HUJI SBM full',
               40: 'Morrison+CESM aerosol',
               50: 'P3',
               51: 'P3 nc',
               52: 'P3 2ice'}
    
    return(str(wrfin.MP_PHYSICS) + ' (' + schemes[wrfin.MP_PHYSICS] + ')')

def wrf_ra_lw_scheme(wrfin):
    """
    Lookup the longwave radiation scheme information in a wrfinput file and return a description string.
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {1: 'RRTM',
               3: 'CAM',
               4: 'RRTMG',
               24: 'RRTMG fast',
               14: 'RRTMG-K',
               5: 'New Goddard',
               7: 'FLG',
               31: 'Held-Suarez',
               99: 'GFDL'}
        
    return(str(wrfin.RA_LW_PHYSICS) + ' (' + schemes[wrfin.RA_LW_PHYSICS] + ')')
          
def wrf_ra_sw_scheme(wrfin):
    """
    Lookup the shortwave radiation scheme information in a wrfinput file and return a description string.
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {1: 'Dudhia',
               2: 'Goddard',
               3: 'CAM',
               4: 'RRTMG',
               24: 'RRTMG',
               14: 'RRTMG-K',
               5: 'New Goddard',
               7: 'FLG',
               99: 'GFDL'}
        
    return(str(wrfin.RA_SW_PHYSICS) + ' (' + schemes[wrfin.RA_SW_PHYSICS] + ')')
        
def wrf_sf_sfclay_scheme(wrfin):
    """
    Lookup the surface layer scheme information in a wrfinput file and return a description string.
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {0: 'No surface-layer',
               1: 'Revised MM5 Monin-Obukhov',
               2: 'Monin-Obukhov (Janjic Eta)',
               3: 'NCEP GFS',
               4: 'QNSE',
               5: 'MYNN',
               7: 'Pleim-Xiu',
               91: 'Old MM5 surface layer'}
        
    return(str(wrfin.SF_SFCLAY_PHYSICS) + ' (' + schemes[wrfin.SF_SFCLAY_PHYSICS] + ')')
       
def wrf_sf_surface_scheme(wrfin):
    """
    Lookup the land-surface scheme information in a wrfinput file and return a description string.
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {0: 'No surface temp prediction',
               1: 'Thermal diffusion',
               2: 'Unified Noah',
               3: 'RUC',
               4: 'Noah-MP',
               5: 'CLM4',
               7: 'Pleim-Xiu',
               8: 'SSiB'}
        
    return(str(wrfin.SF_SURFACE_PHYSICS) + ' (' + schemes[wrfin.SF_SURFACE_PHYSICS] + ')')   

def wrf_pbl_scheme(wrfin):
    """
    Lookup the PBL scheme information in a wrfinput file and return a description string.
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {1: 'YSU',
               2: 'MYJ',
               3: 'GFS (hwrf)',
               4: 'QNSE-EDMF',
               5: 'MYNN2',
               6: 'MYNN3',
               7: 'ACM2',
               8: 'BouLac',
               9: 'UW',
               10: 'TEMF',
               11: 'Shin-Hong',
               12: 'GBM',
               99: 'MRF'}
        
    return(str(wrfin.BL_PBL_PHYSICS) + ' (' + schemes[wrfin.BL_PBL_PHYSICS] + ')') 

def wrf_cu_scheme(wrfin):
    """
    Lookup the cumulus scheme information in a wrfinput file and return a description string.
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {0: 'No cumulus parameterization',
               1: 'Kain-Fritsch (new Eta)',
               2: 'Betts-Miller-Janjic',
               3: 'Grell-Freitas',
               4: 'Scale-aware GFS Simplified Arakawa-Schubert (SAS)',
               5: 'New Grell (G3)',
               6: 'Tiedtke',
               7: 'Zhang-McFarlane from CESM',
               10: 'Modified Kain-Fritsch',
               11: 'Multi-scale Kain-Fritsch',
               14: 'New GFS SAS from YSU',
               16: 'A newer Tiedke',
               93: 'Grell-Devenyi ensemble',
               94: '2015 GFS Simplified Arakawa-Schubert (HWRF)',
               95: 'Previous GFS Simplified Arakawa-Schubert (HWRF)',
               99: 'Previous Kain-Fritsch'}
        
    return(str(wrfin.CU_PHYSICS) + ' (' + schemes[wrfin.CU_PHYSICS] + ')')  

def wrf_shcu_scheme(wrfin):
    """
    Lookup the shallow cumulus scheme information in a wrfinput file and return a description string.
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {0: 'No independent shallow cumulus',
               2: 'Park and Bretherton from CAM5',
               3: 'GRIMS'}
        
    return(str(wrfin.SHCU_PHYSICS) + ' (' + schemes[wrfin.SHCU_PHYSICS] + ')')  
