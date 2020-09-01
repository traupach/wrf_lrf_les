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

FIGURE_SIZE = [15, 4]   # Default figure size [horizontal, vertical]. 

def read_wrfvars(inputs, resample=None):
    """
    Read all wrfvars files in multiple datasets.
    
    Arguments:
        inputs: A dictionary containing dataset names as keys and input directories as values.
        resample: If defined, resample each dataset using no tolerance (use e.g. "1H" to keep only hourly records).
        
    Returns: an xarray.DataArray with all data from wrfvars*.nc files, labelled by dataset.
    """
    
    datasets = []

    for setname, directory in inputs.items():
        datasets.append(xarray.open_mfdataset(directory+"/wrfvars*.nc", combine="nested", concat_dim="time"))
        if not resample is None:
            datasets[-1] = datasets[-1].resample(time=resample).nearest(tolerance=0)
        datasets[-1]['Dataset'] = setname

    dat = xarray.combine_nested(datasets, concat_dim='Dataset')
    dat = prettify_long_names(dat)
    return(dat)

def prettify_long_names(dat):
    """
    Make the long_name attribute for selected variables pretty for plotting.
    
    Arguments:
    dat: Data read by read_wrfvars().
    """
    
    dat.q.attrs['long_name'] = 'Water vapor mixing ratio'
    dat.RTHRATEN.attrs['long_name'] = 'Theta tendency due to radiation'
    dat.eta_tk.attrs['long_name'] = 'Temperature'
    dat.eta_q.attrs['long_name'] = 'Water vapor mixing ratio'
    dat.tk.attrs['long_name'] = 'Temperature'
    dat.rh.attrs['long_name'] = 'Relative humidity'
    dat.ua.attrs['long_name'] = 'Destaggered u-wind component'
    dat.va.attrs['long_name'] = 'Destaggered v-wind component'
    
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
    
    # Calculate difference in base-levels (mass-points).
    zdiffs = wrfin.Z_BASE.values[0][1:] - wrfin.Z_BASE.values[0][:-1]
    
    # Print important initialisation values.
    print('Sea surface temperature (SST):\t\t\t' + str(np.unique(wrfin.SST)[0]) + ' ' + wrfin.SST.attrs['units'])
    print('Surface skin temperature (TSK):\t\t\t' + str(np.unique(wrfin.TSK)[0]) + ' ' + wrfin.TSK.attrs['units'])
    print('Soil temperature at lower boundary (TMN):\t' + str(np.unique(wrfin.TMN)[0]) + ' ' + wrfin.TMN.attrs['units'])
    print('Horizontal grid spacing (DX):\t\t\t' + str(wrfin.DX) + ' m')
    print('Horizontal (S-N) grid spacing (DY):\t\t' + str(wrfin.DY) + ' m')
    print('Horizontal (W-E) domain size:\t\t\t' + str(wrfin.attrs['WEST-EAST_GRID_DIMENSION']-1) + ' mass points')
    print('Horizontal (S-N) domain size:\t\t\t' + str(wrfin.attrs['SOUTH-NORTH_GRID_DIMENSION']-1) + ' mass points')
    print('Vertical domain size:\t\t\t\t' + str(wrfin.attrs['BOTTOM-TOP_GRID_DIMENSION']-1) + ' mass points')
    print('Maximum geopotential height (model-top):\t' + str(np.round(hgt.max().values, 1)) + ' m')
    print('Maximum base-state height (on mass points):\t' + str(np.round(wrfin.Z_BASE.isel(Time=0).max().values, 1)) + ' m')
    print('Minimum, mean, maximum between-level distance:\t' + str(np.round(np.min(zdiffs), 1)) + ', ' + 
          str(np.round(np.mean(zdiffs), 1)) + ', ' + str(np.round(np.max(zdiffs), 1)) + ' m')
    print('Model-top pressure:\t\t\t\t' + str(np.round(wrfin.P_TOP.isel(Time=0).data, 1)) + ' ' + wrfin.P_TOP.attrs['units'])
    print('Coriolis sine latitude term (F):\t\t' + str(np.unique(wrfin.F)[0]) + ' ' + wrfin.F.attrs['units'])
    print('Coriolis cosine latitude term (E):\t\t' + str(np.unique(wrfin.E)[0]) + ' ' + wrfin.E.attrs['units'])
    print('Use light nudging on U and V:\t\t\t' + true_false(wrfin.LIGHT_NUDGING))
    print('Ideal evaporation/surface fluxes:\t\t' + true_false(wrfin.IDEAL_EVAPORATION))
    if wrfin.IDEAL_EVAPORATION > 0:
        print('Surface wind for ideal surface fluxes:\t\t' + str(wrfin.SURFACE_WIND) + ' m s-1')
    print('Constant radiative cooling profile:\t\t' + true_false(wrfin.CONST_RAD_COOLING))
    print('Relax stratsopheric T and q profiles?\t\t' + true_false(wrfin.RELAX_STRATOSPHERE))
    print('Relax U and V to set profiles?\t\t\t' + true_false(wrfin.RELAX_UV_WINDS))
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

def perturbation_details(inputs):
    """
    Print perturbation settings stored in a WRF input file.
    
    Arguments:
        wrfout_dir: The directory containing wrfout files to test (the first file is used).
    """
    
    for input_set, input_dir in inputs.items():
        wrfout_file = sorted(glob.glob(input_dir+'/wrfout*'))[0]    
        pert_in = xarray.open_dataset(wrfout_file)
        
        out = (input_set+':').ljust(24)
        
        if bool(pert_in.PERTURB_T) or bool(pert_in.PERTURB_Q):
            out = out + 'Perturbed '
            if bool(pert_in.PERTURB_T):
                out = out + 'T with amplitude ' + str(pert_in.TTENDAMP) + ' K day-1 '
                if bool(pert_in.PERTURB_Q):
                    out = out + 'and '
        
            if bool(pert_in.PERTURB_Q):
                out = out + 'q with amplitude ' + str(pert_in.QTENDAMP) + ' kg kg-1 day-1 '
    
            out = out + 'at ' + str(pert_in.P_PERT) + ' hPa.'
        else:
            out = out + 'No perturbation.'
        
        print(out)
        
def compare_perturbation_forcing(dat, p_pert, k_pert):
    """
    Compare perturbation forcing functions for either perturbing around a level or around a pressure.
    
    Arguments:
        dat: Dataset that contains at least 'pres', pressure per level in Pa.
        p_pert: The pressure to perturb around (hPa).
        k_pert: The zero-based level to perturb around.
    """
    
    # Use mean pressure per level in data, convert to hPa.
    p = (dat.pres.sel(Dataset='RCE').mean(['time'])/100).values # [hPa]
    
    forcing_level = np.zeros(len(p))
    forcing_pressure = np.zeros(len(p))
    
    for i in range(len(p)):
        if i == k_pert:
            delta = 1
        else:
            delta = 0
            
        forcing_level[i] = 0.5 * (delta + np.exp(-((p[k_pert] - p[i])/75)**2))
        forcing_pressure[i] = np.exp(-((p_pert - p[i])/75)**2)
            
    plt.plot(forcing_level, p, label='Around model level')
    plt.plot(forcing_pressure, p, label='Around specific pressure')
    plt.xlabel('Perturbation forcing f')
    plt.ylabel('Pressure [hPa]')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()

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

    # Potential temperature profile. 300 K is added as per instruction in WRF user guide
    # under 'special WRF output variables'.
    potential_temp = wrfin.T.isel(Time=0).mean(['south_north', 'west_east']) + 300 
    ax[0].plot(potential_temp, base_height, label='WRF input')
    ax[0].set_xlabel('Total potential\ntemperature [K]')
    ax[0].set_ylabel('Base state height [km]')

    # Water vapour mixing ratio (converted from kg kg-1 to g kg-1).
    ax[1].plot(wrfin.QV_BASE.isel(Time=0)*1000, base_height, label='WRF input')
    ax[1].set_xlabel('Water vapour\nmixing ratio [g kg-1]')
       
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

def rewrap_labels(axes, length_x=20, length_y=24):
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
            
def compare_profiles(dat, start, end, variables=['tk','q','ua','va','rh'], figsize=FIGURE_SIZE):
    """
    Find temporal means of water vapour mixing ratio (q) and temperature (tk), and plot the differences
    between them by pressure level.
    
    Arguments:
        dat: The data (xarray.DataArray). 
        start: Dictionary with start times as values, dataset values as keys.
        end: Dictionary with end times as values, dataset values as keys.
        control_name: Name for control values of 'Dataset' in dat.  
        variables: The variables to compare.
        figsize: Figure size [width, height].
    """
    
    diffs = diff_means(dat=dat[variables], start=start, end=end)

    fig, ax = plt.subplots(ncols=len(variables), sharey=True, figsize=figsize)
    
    for i in range(len(variables)):
        diffs[variables[i]].plot(hue='Dataset', y='level', ax=ax[i], yincrease=False, add_legend=False)
        ax[i].axvline(x=0, color='red')
        ax[i].set_title('')
        
    plt.legend(labels=diffs.Dataset.values, bbox_to_anchor=(1.05, 1))
    keep_left_axis(axes=ax)
    rewrap_labels(axes=ax, length_x=12)
    plt.tight_layout()
    plt.show()
        
def diff_means(dat, start, end, control_name='Control'):
    """
    Select data within a time period, take the mean control and mean perturb values, and return 
    differences as perturb - control.
    
    Arguments:
        dat: The data (xarray.DataArray). 
        start: Dictionary with start times as values, dataset values as keys.
        end: Dictionary with end times as values, dataset values as keys.
        control_name: Name for control values of 'Dataset' in dat.  
        
    Returns: perturb - control differences in temporal means.
    """
    
    profs = profiles_by_dataset(dat=dat, start=start, end=end)
    
    def take_diff(x): 
        with xarray.set_options(keep_attrs=True):
            diffs = x - profs.sel(Dataset=control_name)
        return(diffs)
        
    diffs = profs.drop_sel(Dataset=control_name).groupby('Dataset').map(take_diff)
    return(diffs)
    
def profiles_by_dataset(dat, start, end):
    """
    Find mean profiles by dataset, with different start and end points per dataset.
    
    Arguments:
    dat: The data to find averages of, with Dataset as a coordinate.
    start: Dictionary with start times as values, dataset values as keys.
    end: Dictionary with end times as values, dataset values as keys.
    """
    
    all_profiles = []
    
    for dataset in dat.Dataset.values:
        p = dat.sel(Dataset=dataset, time=slice(start[dataset], end[dataset])).mean('time', keep_attrs=True)
        p['Dataset'] = dataset
        all_profiles.append(p)
        
    profs = xarray.combine_nested(all_profiles, concat_dim='Dataset')
    return(profs)
    
def plot_profiles(dat, start, end, control_name='Control', variables=['tk','q','ua','va','rh'], 
                  figsize=FIGURE_SIZE):
    """
    Plot temporals means of water vapour mixing ratio (q) and temperature (tk) by Dataset.
    
    Arguments:
        dat: The data (xarray.DataArray). 
        start: RCE start times (per Dataset), used as slice start.
        end: RCE end times (per Dataset), used as slice end.
        control_name: Name for control values of 'Dataset' in dat. 
        variables: The variables to plot.
        figsize: Figure size [width, height].
    """
    
    profs = profiles_by_dataset(dat=dat, start=start, end=end)
    
    fig, ax = plt.subplots(ncols=len(variables), sharey=True, figsize=figsize)
    for i in range(len(variables)):
        plot = profs[variables[i]].plot(hue='Dataset', y='level', ax=ax[i], yincrease=False, add_legend=False)
        
    plt.legend(labels=profs.Dataset.values, bbox_to_anchor=(1.05, 1))
    keep_left_axis(axes=ax)
    rewrap_labels(axes=ax, length_x=12)
    plt.tight_layout()
    plt.show()

def plot_daily_rain(inputs, patterns, figsize=FIGURE_SIZE):
    """
    Plot daily rain by dataset. Patterns should match a single 
    file from which the last-first accumulated rain field will be plotted.
    
    Variables:
        inputs: A dictionary with dataset name/output directory combinations.
        patterns: A dictionary with dataset name/file pattern combinations.
        var: The field name to plot.
        figsize: Figure size [width, height].
    """

    fields = []
    for setname, directory in inputs.items():
        file = sorted(glob.glob(directory+'/'+patterns[setname]))
        assert len(file) == 1, 'Pattern does not match a single file.'
        
        rain = xarray.open_dataset(file[0]).RAINNC
        rain = rain.isel(Time=len(rain.Time)-1) - rain.isel(Time=0)
        fields.append(rain)
        fields[-1]['Dataset'] = setname
    
    fields = xarray.combine_nested(fields, concat_dim='Dataset')
    fields.attrs['long_name'] = 'Daily rain accumulation'
    fields.attrs['units'] = 'mm'
    
    if len(fields.Dataset) > 1:
        p = fields.plot(col='Dataset', figsize=figsize).set_titles('{value}')
        for ax in p.axes.flat:
            ax.set_aspect('equal')
    else:
        fields.plot(figsize=figsize)
        plt.gca().set_aspect('equal')
        
    plt.show()

def wind(dat):
    """
    Calculate wind magnitude.
    
    Arguments:
        dat: Data set, must contain 'ua' and 'va' wind fields.
        
    Returns: wind magnitude fields.
    """
    
    wind = np.sqrt(dat.ua**2 + dat.va**2)
    assert dat.ua.units == dat.va.units
    wind.attrs['units'] = dat.ua.units
    wind.attrs['long_name'] = 'Wind'
    wind.level.attrs['long_name'] = 'Level'
    return(wind)

def plot_wind(wind, sepVar='Dataset', figsize=[15,7]):
    """
    Plot wind fields at all levels.
    
    Arguments:
        wind: Wind values returned by wind().
        sepVar: Variable by which to divide the plot.
        figsize: Size for plot [width, height].
    """
    
    wind.plot(row=sepVar, x='time', figsize=figsize, cmap='plasma', yincrease=False).set_titles('{value}')
    plt.show()
    
def plot_wind_levels(wind, sepVar='Dataset', figsize=[15,7], plot_levels=[850, 500, 350, 200]):
    """
    Plot wind fields at selected levels.
    
    Arguments:
        wind: Wind values returned by wind().
        sepVar: Variable by which to divide the plot.
        figsize: Size for plot [width, height].
        plot_levels: Pressure levels for which to plot timeseries [hPa].
    """
        
    wind.sel(level=plot_levels).plot(hue='Dataset', row='level', figsize=figsize).set_titles('{value} hPa')
    plt.show()
    
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

def plot_radiative_cooling_profiles(dat, figsize=FIGURE_SIZE):
    """
    Plot the radiative cooling profiles for each dataset.
    
    Arguments:
        dat: The data to plot, containing RTHRATEN and Dataset.
        figsize: Figure size [width, height] - note may change due to tight_layout called by xarray.
    """
    
    rthprofile = dat.RTHRATEN.mean('time', keep_attrs=True)
    
    # Convert from K s-1 to K day-1.
    assert rthprofile.attrs['units'] == 'K s-1', 'Unexpected units on RTHRATEN.'
    with xarray.set_options(keep_attrs=True):
        rthprofile = rthprofile * 86400 
    rthprofile.attrs['units'] = 'K day-1'
    
    rthprofile.plot(y='level', hue='Dataset', figsize=figsize, yincrease=False)
    rewrap_labels(axes=[plt.gca()], length_x=60)
    plt.show()
    
def plot_radiative_cooling_by_level(dat, plot_levels, figsize=FIGURE_SIZE):
    """
    Plot the radiative cooling time series at selected levels.
    
    Arguments:
        dat: The data to plot, containing RTHRATEN and Dataset.
        plot_levels: The pressure levels to show [hPa].
        figsize: Figure size [width, height] - note may change due to tight_layout called by xarray.
    """
    
    levdata = dat.RTHRATEN.isel(time=slice(1,len(dat.time))).sel(level=plot_levels)
    
    # Convert from K s-1 to K day-1.
    assert levdata.attrs['units'] == 'K s-1', 'Unexpected units on RTHRATEN.'
    with xarray.set_options(keep_attrs=True):
        levdata = levdata * 84600 
        
    levplot = levdata.plot(col='level', hue='Dataset', sharey=False, col_wrap=2,
                           figsize=figsize).set_titles('{value} hPa')
    for ax in levplot.axes:
        rewrap_labels(axes=ax, length_y=12)
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

def compare_nc(file1, file2):
    """
    Show any differences in contents between two netCDF files.
    
    Arguments:
        file1, file2: The two files to compare.
    """
    
    nc1 = xarray.open_dataset(file1)
    nc2 = xarray.open_dataset(file2)
    
    diffs_found = False
    
    common_fields = [field for field in nc1.keys() if field in nc2.keys()]
    nc1_only_fields = [field for field in nc1.keys() if field not in nc2.keys()]
    nc2_only_fields = [field for field in nc2.keys() if field not in nc1.keys()]
    
    if len(nc1_only_fields) > 0:
        print("The following fields are in file1 but not file2:")
        print('\n'.join(nc1_only_fields))
            
    if len(nc2_only_fields) > 0:
        print("The following fields are in file2 but not file1:")
        print('\n'.join(nc2_only_fields))
    
    nc1.close()
    nc2.close()
    del(nc1)
    del(nc2)
    
    # Which fields are different?
    for key in common_fields:
        nc1 = xarray.open_dataset(file1)
        nc2 = xarray.open_dataset(file2)
        
        if not nc1[key].equals(nc2[key]):
            if not diffs_found:
                format_str = "Differences stats{:<15} {:>15} {:>15} {:>20}"
                print(format_str.format(*[':','bias','max abs','med rel bias [%]']))
            diffs_found = True
            diffs = nc1[key] - nc2[key]
            rel_diff = np.nanmedian((diffs / pre[key]).values)*100
            mean_diff = np.mean(diffs.values)
            max_diff = np.max(np.abs(diffs.values))
            print("Differences in {:<15} {:>15} {:>15} {:>20}".format(*[
                key+':', str(np.round(mean_diff, 8)), str(np.round(max_diff, 8)), 
                str(np.round(rel_diff, 2))]))
            
        nc1.close()
        nc2.close()
        del(nc1)
        del(nc2)
            
    if not diffs_found:
        print("No differences found.")

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

def true_false(v):
    """
    Return 'true' if v != 0 and 'false' if v == 0.
    """

    if v == 0:
        return('False')
    else:
        return('True')
        
def compare_vert_levels(dat, start, end):
    """
    Plot the vertical levels in the first timestep of a dataset, compared to the average 
    levels between a start and end time.
    
    Arguments:
    dat: The dataset containing z.
    start: Start time for RCE period (inclusive).
    end: End time for RCE period (exclusive).
    """

    fig, ax = plt.subplots()
    initial_heights = dat.z.isel(time=0)
    rce_heights = dat.z.sel(time=slice(start, end)).mean(['time'])

    initial_heights.plot(ax=ax, label='Initial levels')
    rce_heights.plot(ax=ax, label='Mean RCE period levels')
    plt.xlabel('Vertical level')
    plt.ylabel('Mean height of level points [m]')
    plt.title('')
    plt.legend()
    plt.show()

def calc_profiles(dat, start, end, plot=True, plotVars=['eta_tk', 'eta_q']):
    """
    Calculate mean T and water vapour mixing ratio q profiles.
    
    Arguments:
    dat: Dataset containing at least 'tk' and 'q' per height.
    start: Averaging start time (inclusive).
    end: Averaging end time (exclusive).
    
    Returns: Mean profiles over the specified time period.
    """
    
    dat['pres'] = dat.pres / 100 # Convert Pa to hPa.
    dat.pres.attrs['units'] = 'hPa'
    dat.pres.attrs['long_name'] = 'Hydrostatic pressure'
    
    profiles = dat.sel(time=slice(start, end)).mean(['time'], keep_attrs=True)
    profiles = profiles.set_coords('pres')
    profiles = profiles.swap_dims({'bottom_top': 'pres'})
    
    if plot:
        fig, ax = plt.subplots(ncols=2)
        for i, var in enumerate(plotVars):
            profiles[var].plot(ax=ax[i], y='pres')
            ax[i].invert_yaxis()
            ax[i].set_title('')
        rewrap_labels(axes=ax)
        plt.show()
        
    return profiles

def write_Tq_profiles(profiles, T_file='T_profile', q_file='q_profile'):
    """
    Write temperature T and water vapour mixing ratio q profiles to files with one value per line.
    
    Arguments:
    profiles: Profiles returned by calc_profiles().
    """
    
    profiles.eta_tk.to_dataframe().to_csv(T_file, header=False, index=False)
    profiles.eta_q.to_dataframe().to_csv(q_file, header=False, index=False)
                     
def surface_temps(wrf_file):
    """
    Print out the (unique) value of SST, TSK, and TMN from a wrfout file.
    
    Arguments:
    wrf_file: The wrfout file to open.
    """
    
    nc = xarray.open_dataset(wrf_file)
    
    sst = np.unique(nc.tail(Time=1).SST.values)
    tsk = np.unique(nc.tail(Time=1).TSK)              
    tmn = np.unique(nc.tail(Time=1).TMN)
    
    assert len(sst) == 1, 'Error: SST is not unique.'
    assert len(tsk) == 1, 'Error: TSK is not unique.'
    assert tsk == tmn, 'ERROR: TSK does not equal TMN.'
    
    print('Last time stamp in WRF file has SST of ' + str(sst[0]) + 
          ' K and TSK of ' + str(tsk[0]) + ' K.')
    
def plot_profile_range(dat, var, ignoreDatasets=['RCE'], figsize=[15,6], overplot_x=None, overplot_y=None):
    """
    Plot a mean profile per Dataset, with a shaded region to show min->max area.
    
    Arguments:
    dat: The dataset to plot (wrfvars).
    var: The variable to plot the profiles of.
    ignoreDatasets: A list of datasets to ignore.
    figsize: The figure size.
    overplot_x, overplot_y: If specified, points to overplot in red.
    """

    assert (overplot_x is None) == (overplot_y is None), 'overplot_x and overplot_y are both required.'
    
    sets = [ds for ds in dat.Dataset.values if ds not in ignoreDatasets]
    fig, ax = plt.subplots(ncols=len(sets), figsize=figsize)

    means = dat.sel(Dataset=sets).mean('time', keep_attrs=True)
    mins = dat.sel(Dataset=sets).min('time', keep_attrs=True)
    maxs = dat.sel(Dataset=sets).max('time', keep_attrs=True)

    for axnum, dataset in enumerate(sets):
        means[var].sel(Dataset=dataset).plot(ax=ax[axnum], yincrease=False, color='black', y='level')
        ax[axnum].fill_betweenx(mins.level.values, mins[var].sel(Dataset=dataset).values, 
                               maxs[var].sel(Dataset=dataset).values, color='lightblue')
        ax[axnum].set_title(dataset)
        
        if not overplot_x is None:
            ax[axnum].scatter(overplot_x, overplot_y, color='red')

    rewrap_labels(axes=ax)
    plt.tight_layout()
    
def plot_tq_stratosphere(dat, RCE_profiles, p_from=200, p_to=80, figsize=[13,4], **kwargs):
    """
    Plot T and q profiles by Dataset, showing mean and range of each profile, with overlaid RCE profiles.
    
    Arguments: 
    dat: wrfvars data to plot.
    RCE_profiles: RCE profile data.
    p_from, p_to: Pressure range to plot [hPa].
    figsize: Figure size [width, height].
    kwargs: Extra arguments to plot_profile_range().
    """
    
    RCE = RCE_profiles.sel(pres=slice(p_from,p_to))
    dat = dat.sel(level=slice(p_from,p_to))
    plot_profile_range(dat=dat, var='tk', figsize=figsize, overplot_x=RCE.eta_tk, overplot_y=RCE.pres, **kwargs)
    plot_profile_range(dat=dat, var='q', figsize=figsize, overplot_x=RCE.eta_q, overplot_y=RCE.pres, **kwargs)
    