# wrf_perturbation.py
# 
# Functions for analysing output of WRF perturbation experiments.
#
# Author: T. Raupach <t.raupach@unsw.edu.au>

import matplotlib.pyplot as plt
import modules.atmosphere as atm
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap
import datetime
import xarray
import glob
import os

FIGURE_SIZE = [15, 4]   # Default figure size [horizontal, vertical]. 

def read_wrfvars(inputs, resample=None, drop_vars=None, calc_rh=True, quiet=False):
    """
    Read all wrfvars files in multiple datasets.
    
    Arguments:
        inputs: A dictionary containing resolutions as keys and dictionaries as values. 
                Each subdictionary should have dataset names as keys and input directories as values.
        resample: If defined, resample each dataset using no tolerance (use e.g. "1H" to keep only hourly records).
        drop_vars: If defined, drop selected variables if they exist in any dataset.
        calc_rh: Calculate relative humidity?
        quiet: Be quiet.
        
    Returns: an xarray.DataArray with all data from wrfvars*.nc files, labelled by dataset.
    """

    res_datasets = {}
    
    for res in inputs.keys():

        datasets = []

        for setname, directory in inputs[res].items():
            
            if not quiet:
                print(f'Reading {res} dataset ({directory}): ' + setname + '...')
            
            datasets.append(xarray.open_mfdataset(directory+'/wrfvars*.nc', combine='nested', concat_dim='time', parallel=True))
            if not resample is None:
                datasets[-1] = datasets[-1].resample(time=resample).nearest(tolerance=0)

            if not drop_vars is None:
                dvars = [v for v in drop_vars if v in datasets[-1].keys()]
                datasets[-1] = datasets[-1].drop_vars(dvars)

            datasets[-1]['Dataset'] = setname

        # Check all datasets have the same keys before merging. Merge only uses keys from first dataset.
        keys = datasets[0].keys()
        for i in range(1, len(datasets)):
            keydiffs = set(keys).symmetric_difference(set(datasets[i].keys()))
            assert len(keydiffs) == 0, ('Dataset keys have differences: ' + 
                                        str(keydiffs) + '. Consider using drop_vars.')

        dat = xarray.combine_nested(datasets, concat_dim='Dataset', compat='equals', 
                                    combine_attrs='drop_conflicts')
        dat = prettify_long_names(dat)

        dat['rh'] = atm.relative_humidity(theta=dat.T+300, p=dat.level, q=dat.q)
        dat.rh.attrs['long_name'] = 'Relative humidity'
        dat.rh.attrs['units'] = '%'

        res_datasets[res] = dat
        
    return(res_datasets)

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
    if 'RELAX_UV_WINDS' in wrfin.attrs:
        print('Relax U and V to set profiles?\t\t\t' + true_false(wrfin.RELAX_UV_WINDS))
        if wrfin.RELAX_UV_WINDS > 0:
            print('Wind relaxation time:\t\t\t\t' + str(wrfin.WIND_RELAXATION_TIME) + ' s')
    elif 'RELAX_U_WINDS' in wrfin.attrs:
        print('Relax U to set profile?\t\t\t\t' + true_false(wrfin.RELAX_U_WINDS))
        print('RELAX V to set profile?\t\t\t\t' + true_false(wrfin.RELAX_V_WINDS))
        if wrfin.RELAX_U_WINDS > 0 or wrfin.RELAX_V_WINDS > 0:
            print('Wind relaxation time:\t\t\t\t' + str(wrfin.WIND_RELAXATION_TIME) + ' s')
    print('Physics schemes:')
    print('\tMicrophysics:\t\t\t\t' + wrf_mp_scheme(wrfin))
    print('\tRadiation (longwave):\t\t\t' + wrf_ra_lw_scheme(wrfin))
    print('\tRadiation (shortwave):\t\t\t' + wrf_ra_sw_scheme(wrfin))
    print('\tSurface layer:\t\t\t\t' + wrf_sf_sfclay_scheme(wrfin))
    print('\tLand-surface:\t\t\t\t' + wrf_sf_surface_scheme(wrfin))
    print('\tPBL:\t\t\t\t\t' + wrf_pbl_scheme(wrfin))
    print('\tCumulus:\t\t\t\t' + wrf_cu_scheme(wrfin))
    print('Turbulence options:')
    print('\tDiffusion (diff_opt):\t\t\t' + wrf_diff_opt(wrfin))
    print('\tEddy coefficient (km_opt):\t\t' + wrf_km_opt(wrfin))
    
    if plot_profiles:
        plot_wrfinput_profiles(wrfin=wrfin, sounding_file=sounding_file)

def perturbation_details(inputs):
    """
    Print perturbation settings stored in a WRF input file.
    
    Arguments:
        inputs: Dictionary (res/dict) with dataset/directory combinations per resolution.
    """
    
    for i, res in enumerate(inputs.keys()):
        if not i == 0:
            print('')
        
        print('Perturbation details for ' + res + ':')
    
        for input_set, input_dir in inputs[res].items():
            wrfout_file = sorted(glob.glob(input_dir+'/wrfout*'))[0]    
            pert_in = xarray.open_dataset(wrfout_file)

            out = (input_set+':').ljust(30)

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
    p = (dat.pres.mean(['time'])/100).values # [hPa]
    
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

def calc_profile_diffs(profs, control_name='Control', variables=['tk','q','rh'], neg=[]):
    diffs = diff_means(profs=profs, control_name=control_name)
    pos_diffs = diffs.drop_sel(Dataset=neg)
    neg_diffs = -1 * diffs.sel(Dataset=neg)
    diffs = xarray.concat([pos_diffs, neg_diffs], dim='Dataset')
    return diffs
        
def compare_profiles(profs, control_name='Control', variables=['tk','q','rh'], xlims=None,
                     figsize=FIGURE_SIZE, title='', neg=[], loc='best'):
    """
    Find temporal means of variables, and plot the differences
    between them by pressure level.
    
    Arguments:
        profs: Profiles to compare.
        control_name: Name for control values of 'Dataset' in dat.  
        variables: The variables to compare.
        xlims: Limits for the x axis (dict with limits per variable).
        figsize: Figure size [width, height].
        title: Plot title.
        neg: Which values of Dataset should have their differences multiplied by -1?
        loc: Loc argument to plt.legend.
    """
    
    diffs = calc_profile_diffs(profs=profs, control_name=control_name, neg=neg)
    
    return plot_profiles(profs=diffs, variables=variables, figsize=figsize, 
                         vline=0, title=title, xlims=xlims, loc=loc)
        
def diff_means(profs, control_name='Control'):
    """
    Select data within a time period, take the mean control and mean perturb values, and return 
    differences as perturb - control.
    
    Arguments:
        profs: Profiles by Dataset.
        control_name: Name for control values of 'Dataset' in dat.  
        
    Returns: perturb - control differences in temporal means.
    """
    
    def take_diff(x): 
        with xarray.set_options(keep_attrs=True):
            diffs = x - profs.sel(Dataset=control_name)
        return(diffs)
        
    diffs = profs.drop_sel(Dataset=control_name).groupby('Dataset').map(take_diff)
    return(diffs)

def pairwise_diffs(dat, start, end, comp_pairs, relative=False):
    """
    Select data within a time period, take the mean profiles, and return pair-wise differences.
    
    Arguments:
        dat: The data (xarray.DataArray). 
        start: Averaging start time.
        end: Averaging end time.
        comp_pairs: Dictionary with ref: test pairs of Dataset names. 
                    Differences will be test - ref.  
        relative: Calculate relative differences (test - ref) / ref * 100 as percent?
        
    Returns: perturb - control differences in temporal means.
    """
     
    profs = dat.sel(time=slice(start, end)).mean('time', keep_attrs=True)
    
    diffs = []
    
    for name, comb in comp_pairs.items():
        ref, test = comb
        with xarray.set_options(keep_attrs=True):   
            diffs.append(profs.sel(Dataset=test) - profs.sel(Dataset=ref))
            
        if relative:
            diffs[-1] = diffs[-1] / profs.sel(Dataset=ref) * 100
            for key in diffs[-1].keys():
                diffs[-1][key].attrs['units'] = '%'
            
        diffs[-1]['Dataset'] = name
        
    diffs = xarray.combine_nested(diffs, concat_dim='Dataset')
    return(diffs)

def plot_pairwise_diffs(dat, start, end, comp_pairs, variables=['tk','q','ua','va','rh'],
                        figsize=FIGURE_SIZE, relative=False, title=''):
    """
    Plot pairwise differences.
    
        Arguments:
        dat: The data (xarray.DataArray). 
        start: Averaging start time.
        end: Averaging end time.
        comp_pairs: Dictionary with ref: test pairs of Dataset names. 
                    Differences will be test - ref. 
        variables: Variables to plot differences of.
        relative: Calculate relative differences?
    """
    
    diffs = pairwise_diffs(dat=dat, start=start, end=end, comp_pairs=comp_pairs, relative=relative)
    plot_profiles(profs=diffs, variables=variables, figsize=figsize, vline=0, title=title)
    
def plot_profiles(profs, variables=['tk','q','ua','va','rh'], ylims=[1000, 200], 
                  xlims=None, figsize=FIGURE_SIZE, vline=None, title='', loc='best'):
    """
    Plot simple profiles by Dataset.
    
    Arguments:
        profs: Profiles by dataset.
        variables: The variables to plot.
        figsize: Figure size [width, height].
        xlims: Limits for the x axis (dictionary with limits per variable).
        ylims: Limits for the y axis (list applied to all plots).
        vline: x coordinate for a vertical line in red.
        title: Plot title.
        loc: Loc argument to plt.legend.
        
    Returns: fig, ax for plot.
    """
    
    fig, ax = plt.subplots(ncols=len(variables), sharey=True, figsize=figsize)
    for i, var in enumerate(variables):
        plot = profs[var].plot(hue='Dataset', y='level', ax=ax[i], ylim=ylims, 
                               xlim=xlims[var] if xlims is not None else None,
                               yincrease=False, add_legend=False)
        ax[i].set_title('')
        if vline is not None:
            ax[i].axvline(x=vline, color='red')
        
    plt.legend(labels=profs.Dataset.values, bbox_to_anchor=(1.05, 1), loc=loc)
    keep_left_axis(axes=ax)
    rewrap_labels(axes=ax, length_x=30)
    plt.suptitle(title)
    
    return fig, ax
    
def mean_profiles(dat, start, end, variables=['tk','q','ua','va','rh'], 
                  plot=True, figsize=FIGURE_SIZE, title=''):
    """
    Calculate temporal means of variables by Dataset.
   
    Arguments:
        dat: The data (xarray.DataArray). 
        start: RCE start time, used as slice start.
        end: RCE end time, used as slice end.
        variables: The variables to plot.
        figsize: Figure size [width, height].
        title: Plot title.
    """
    
    profs = dat.sel(time=slice(start, end)).mean('time', keep_attrs=True)
    if plot:
        plot_profiles(profs=profs, variables=variables, figsize=figsize, title=title)
    return profs

def plot_daily_rain(inputs, patterns, figsize=FIGURE_SIZE, ncols=5):
    """
    Plot daily rain by dataset. Pattern should match a files 
    from which the last-first accumulated rain field will be plotted.
    
    Variables:
        inputs: A dictionary with res/dict with dict=dataset name/output directory combinations.
        patterns: File pattern to match as dictionary similar to inputs.
        figsize: Figure size [width, height].
    """

    assert inputs.keys() == patterns.keys(), 'Inputs and patterns should have same keys.'
    
    for res in inputs.keys():

        fields = []
        for setname, directory in inputs[res].items():
            files = sorted(glob.glob(directory+'/'+patterns[res]))
            assert len(files) != 0, 'Pattern ' + patterns[res] + ' does not match any files.'

            rain = xarray.open_mfdataset(files, concat_dim='Time', combine='nested').RAINNC
            rain = rain.isel(Time=len(rain.Time)-1) - rain.isel(Time=0)
            fields.append(rain)
            fields[-1]['Dataset'] = setname

        fields = xarray.combine_nested(fields, concat_dim='Dataset')
        fields.attrs['long_name'] = 'Daily rain accumulation'
        fields.attrs['units'] = 'mm'

        if len(fields.Dataset) > 1:
            p = fields.plot(col='Dataset', col_wrap=ncols, figsize=figsize).set_titles('{value}')
            for ax in p.axes.flat:
                ax.set_aspect('equal')
        else:
            fields.plot(figsize=figsize)
            plt.gca().set_aspect('equal')

        plt.suptitle(res, y=1.01)
        plt.show()

def plot_fields(inputs, pattern, var, timeidx=0, figsize=FIGURE_SIZE, meanover=None):
    """
    Plot fields by dataset. Pattern should match a single file for each dataset.
    
    Variables:
        inputs: A dictionary with res/dict with dict=dataset name/output directory combinations.
        pattern: File pattern to match as dictionary similar to inputs.
        var: The field name to plot.
        timeidx: Index of time within the file to plot.
        figsize: Figure size [width, height].
        meanover: Dimensions over which to take the mean, or None.
    """

    assert inputs.keys() == pattern.keys(), 'Inputs and patterns should have same keys.'
    
    for res in inputs.keys():

        fields = []
        for setname, directory in inputs[res].items():
            files = sorted(glob.glob(directory+'/'+pattern[res]))
            assert len(files) != 0, 'No files matched.'

            field = xarray.open_mfdataset(files, concat_dim='Time', combine='nested')[var].isel(Time=timeidx)
            
            
            if not meanover is None:
                field = field.mean(dim=meanover)
            
            fields.append(field)
            fields[-1]['Dataset'] = setname

        fields = xarray.combine_nested(fields, concat_dim='Dataset')
        if len(fields.Dataset) > 1:
            p = fields.plot(col='Dataset', col_wrap=5, figsize=figsize).set_titles('{value}')
            for ax in p.axes.flat:
                ax.set_aspect('equal')
        else:
            fields.plot(figsize=figsize)
            plt.gca().set_aspect('equal')

        plt.suptitle(res, y=1.01)
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

def plot_wind(wind, sepVar='Dataset', figsize=[15,7], title=''):
    """
    Plot wind fields at all levels.
    
    Arguments:
        wind: Wind values returned by wind().
        sepVar: Variable by which to divide the plot.
        figsize: Size for plot [width, height].
        title: Plot title.
    """
    
    if(len(wind[sepVar]) > 1):
        wind.plot(row=sepVar, x='time', figsize=figsize, cmap='plasma', yincrease=False).set_titles('{value}')
        plt.suptitle(title, y=1.01)
    else:
        wind.plot(x='time', figsize=figsize, cmap='plasma', yincrease=False)
        plt.title(title)
        
    plt.show()
    
def plot_wind_levels(wind, sepVar='Dataset', figsize=[15,7], plot_levels=[850, 500, 350, 200], title=''):
    """
    Plot wind fields at selected levels.
    
    Arguments:
        wind: Wind values returned by wind().
        sepVar: Variable by which to divide the plot.
        figsize: Size for plot [width, height].
        plot_levels: Pressure levels for which to plot timeseries [hPa].
        title: Plot title.
    """
        
    wind.sel(level=plot_levels).plot(hue='Dataset', row='level', figsize=figsize).set_titles('{value} hPa')
    plt.suptitle(title, y=1.02)
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

def plot_radiative_cooling_profiles(dat, figsize=FIGURE_SIZE, title=''):
    """
    Plot the radiative cooling profiles for each dataset.
    
    Arguments:
        dat: The data to plot, containing RTHRATEN and Dataset.
        figsize: Figure size [width, height] - note may change due to tight_layout called by xarray.
        title: Plot title.
    """
    
    rthprofile = dat.RTHRATEN.mean('time', keep_attrs=True)
    
    # Convert from K s-1 to K day-1.
    assert rthprofile.attrs['units'] == 'K s-1', 'Unexpected units on RTHRATEN.'
    with xarray.set_options(keep_attrs=True):
        rthprofile = rthprofile * 86400 
    rthprofile.attrs['units'] = 'K day-1'
    
    rthprofile.plot(y='level', hue='Dataset', figsize=figsize, yincrease=False)
    rewrap_labels(axes=[plt.gca()], length_x=60)
    plt.title(title)
    plt.show()
    
def plot_radiative_cooling_by_level(dat, plot_levels, figsize=FIGURE_SIZE, title=''):
    """
    Plot the radiative cooling time series at selected levels.
    
    Arguments:
        dat: The data to plot, containing RTHRATEN and Dataset.
        plot_levels: The pressure levels to show [hPa].
        figsize: Figure size [width, height] - note may change due to tight_layout called by xarray.
        title: Plot title.
    """
    
    levdata = dat.RTHRATEN.isel(time=slice(1,len(dat.time))).sel(level=plot_levels)
    
    # Convert from K s-1 to K day-1.
    assert levdata.attrs['units'] == 'K s-1', 'Unexpected units on RTHRATEN.'
    with xarray.set_options(keep_attrs=True):
        levdata = levdata * 84600 
        
    levplot = levdata.plot(col='level', hue='Dataset', sharey=False, col_wrap=2,
                           figsize=figsize).set_titles('{value} hPa')
    #for ax in levplot.axes:
    #    rewrap_labels(axes=ax, length_y=12)
        
    plt.suptitle(title)
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
    
    # Which fields are different?
    for key in common_fields:
        if not nc1[key].equals(nc2[key]):
            if not diffs_found:
                format_str = "Differences stats{:<15} {:>15} {:>15} {:>20}"
                print(format_str.format(*[':','bias','max abs','med rel bias [%]']))
            diffs_found = True
            diffs = nc1[key] - nc2[key]
            rel_diff = np.nanmedian((diffs / nc1[key]).values)*100
            mean_diff = np.mean(diffs.values)
            max_diff = np.max(np.abs(diffs.values))
            print("Differences in {:<15} {:>15} {:>15} {:>20}".format(*[
                key+':', str(np.round(mean_diff, 8)), str(np.round(max_diff, 8)), 
                str(np.round(rel_diff, 2))]))
            
    # Which attributes are different?
    for key in nc1.attrs.keys():
        if not nc1.attrs[key] == nc2.attrs[key]:
            print('Attribute ' + key + ' differs.')
            diffs_found = True
            
    if not diffs_found:
        print("No differences found.")
        
    nc1.close()
    nc2.close()

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

def wrf_diff_opt(wrfin):
    """
    Report the diff_opt value in the wrfinput file. 
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {0: 'No turbulence',
               1: 'Simple diffusion',
               2: 'Full diffusion'}
    
    return(str(wrfin.DIFF_OPT) + ' (' + schemes[wrfin.DIFF_OPT] + ')') 

def wrf_km_opt(wrfin):
    """
    Report the km_opt value in the wrfinput file. 
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {1: 'Constant K',
               2: '3D TKE',
               3: '3D Smagorinsky',
               4: '2D (horiz) Smagorinsky'}
    
    return(str(wrfin.KM_OPT) + ' (' + schemes[wrfin.KM_OPT] + ')')

def wrf_pbl_scheme(wrfin):
    """
    Lookup the PBL scheme information in a wrfinput file and return a description string.
    
    Arguments:
        wrfin: The open wrfinput file as an xarray object.
    """
    
    schemes = {0: 'No PBL scheme',
               1: 'YSU',
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
    
    schemes = {0: 'No cumulus parameterisation',
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
        
def compare_vert_levels(init, RCE, start, end, title=''):
    """
    Plot the vertical levels in the first timestep of a dataset, compared to the average 
    levels between a start and end time.
    
    Arguments:
    init: The 'initial time' dataset containing z.
    RCE: The RCE dataset containing z.
    start: Start time for RCE period.
    end: End time for RCE period.
    """

    fig, ax = plt.subplots()
    initial_heights = init.z.isel(time=0)
    rce_heights = RCE.z.sel(time=slice(start, end)).mean(['time'])

    initial_heights.plot(ax=ax, label='Initial levels')
    rce_heights.plot(ax=ax, label='Mean RCE period levels')
    plt.xlabel('Vertical level')
    plt.ylabel('Mean height of level points [m]')
    plt.title(title)
    plt.legend()
    plt.show()

def target_tk_profiles(wrfout_file, pres, start, end):
    """
    Calculate profiles of values on eta levels.
    
    Arguments:
    wrfout_file: A wrfout_file from which to get the target profiles.
    pres: A pressure field (with times) to use for vertical pressure values between start/end times.
    start: Averaging start time.
    end: Averaging end time.
    
    Returns: Mean profiles over the specified time period.
    """
    
    targets = xarray.open_dataset(wrfout_file)
    targets = targets[['RELAX_T_TARGET_PROFILE', 'RELAX_Q_TARGET_PROFILE']].isel(Time=-1)
    targets['pres'] = pres.sel(time=slice(start, end)).mean('time') / 100 # Convert Pa to hPa.
    targets.pres.attrs['units'] = 'hPa'
    targets = targets.set_coords('pres')
    targets = targets.swap_dims({'bottom_top': 'pres'})
    
    targets = targets.rename_vars({'RELAX_T_TARGET_PROFILE': 'target_T',
                                   'RELAX_Q_TARGET_PROFILE': 'target_q'})
    return targets
                     
def eta_profiles(dat, pres, start, end, variables=['eta_tk', 'eta_q', 'z', 'eta_T']):
    """
    Calculate pressure-level profiles for processed variables.
    
    Arguments:
    dat: wrfvar data to process.
    pres: Pressure field to use (ie a single Dataset).
    start, end: The time slice to use.
    variables: Variables for which to calculate the mean profiles.
    """
    
    assert pres.Dataset.size == 1, 'Expecting 1 dataset in pres.'
    res = dat.sel(time=slice(start, end))[variables].mean('time', keep_attrs=True)
    res['pres'] = pres.sel(time=slice(start, end)).mean('time') / 100
    res.pres.attrs['long_name'] = 'Pressure'
    res.pres.attrs['units'] = 'hPa'
    res = res.set_coords('pres')
    res = res.swap_dims({'bottom_top': 'pres'})
    return(res)
      
def surface_temps(wrf_file, title=''):
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
    
    title = title + ': ' if not title == '' else ''
    print(title + 'Last time stamp in WRF file has SST of ' + str(sst[0]) + 
          ' K and TSK of ' + str(tsk[0]) + ' K.')
    
def plot_profile_range(dat, var, ignoreDatasets=[], figsize=[15,6], overplot_x=None, 
                       overplot_y=None, ncols=None, nrows=1, title=''):
    """
    Plot a mean profile per Dataset, with a shaded region to show min->max area.
    
    Arguments:
    dat: The dataset to plot (wrfvars).
    var: The variable to plot the profiles of.
    ignoreDatasets: A list of datasets to ignore.
    figsize: The figure size.
    overplot_x, overplot_y: If specified, points to overplot in red.
    ncols/nrows: Number of columns/rows to produce.
    title: Plot title.
    """

    assert (overplot_x is None) == (overplot_y is None), 'overplot_x and overplot_y are both required.'
    
    sets = [ds for ds in dat.Dataset.values if ds not in ignoreDatasets]
    if ncols is None:
        ncols = len(sets)
        
    assert ncols * nrows >= len(sets), 'Not enough columns/rows for data.'
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    means = dat[var].sel(Dataset=sets).mean('time', keep_attrs=True)
    mins = dat[var].sel(Dataset=sets).min('time', keep_attrs=True)
    maxs = dat[var].sel(Dataset=sets).max('time', keep_attrs=True)
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
        
    for axnum, dataset in enumerate(sets):
        means.sel(Dataset=dataset).plot(ax=ax.flat[axnum], yincrease=False, color='black', y='level')
        ax.flat[axnum].fill_betweenx(mins.level.values, mins.sel(Dataset=dataset).values, 
                               maxs.sel(Dataset=dataset).values, color='lightblue')
        ax.flat[axnum].set_title(dataset)
        
        if not overplot_x is None:
            ax.flat[axnum].scatter(overplot_x, overplot_y, color='red')

    rewrap_labels(axes=ax.flat)
    plt.suptitle(title)
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
    plot_profile_range(dat=dat, var='tk', figsize=figsize, overplot_x=RCE.target_T, overplot_y=RCE.pres, **kwargs)
    plot_profile_range(dat=dat, var='q', figsize=figsize, overplot_x=RCE.target_q, overplot_y=RCE.pres, **kwargs)
    
def as_date(d, date_format='%Y-%m-%d'):
    """
    Transform a string into a datetime64 object.
    
    Arguments:
    d: The date string.
    date_format: The format the date is in.
    """
    
    return(datetime.datetime.strptime(d, date_format))

def dat_properties(dat, variables, start, end, datasets=None):
    """
    For each dataset, show the min, max, mean and standard deviation values for a variable 
    between start and end times.
    
    Arguments:
    dat: wrfvars data with time, Dataset and pw.
    variables: The variables to average.
    start: Averaging start time.
    end: Averaging end time.
    datasets: Datasets to include in the printout, or None for all.
    """
    
    if datasets is None:
        datasets = dat.Dataset.values
    
    for var in variables:
        print(dat[var].attrs['long_name'].capitalize() + ' [' + dat[var].attrs['units'] + '] from ' +
              str(start) + ' to ' + str(end) + ':')
        print(f'{"Dataset":25} {"min":7} {"max":7} {"mean":7} {"sd":7}')

        for dataset in datasets:
            v = dat[var].sel(Dataset=dataset, time=slice(start, end), drop=True)
            
            vmin = str(np.round(np.nanmin(v.values), 2))
            vmax = str(np.round(np.nanmax(v.values), 2))
            vmean = str(np.round(np.nanmean(v.values), 2))
            vsd = str(np.round(np.sqrt(np.nanvar(v.values)), 2))
            print(f'{dataset:25} {vmin:7} {vmax:7} {vmean:7} {vsd:7}')
            
        if var != variables[-1]:
            print(' ')

def model_setups(inputs, dataset='RCE'):
    """
    Print information about each model setup in inputs.
    
    Arguments:
        inputs: A dictionary of dictionaries; key = res, value = dictionary with key = dataset, value = dir.
        dataset: The dataset to show information for.
    """
    
    for res in inputs.keys():
        print('Model setup for ' + res + ' (' + dataset + '):')
        analyse_wrfinput(wrfinput_file=inputs[res][dataset]+'wrfinput_d01',
                         sounding_file=inputs[res][dataset]+'input_sounding')
        
def plot_profiles_by_time(dat, variable, figsize=(14,10), col_wrap=3, yincrease=False):
    """
    Plot variable vs time vs height for each key in 'dat'.
    
    Arguments:
        dat: Data to plot. Should be dictionary.
        variable: The variable to plot.
        figsize: Figure size width x height.
        col_wrap: Number of columns per row.
        yincrease: y axis increases from bottom to top?
    """
    
    for res in dat.keys():
        dat[res][variable].plot(col='Dataset', x='time', figsize=figsize, col_wrap=col_wrap,
                                yincrease=yincrease).set_titles('{value}')
        plt.suptitle(res, y=1.01)
        plt.show()
        
def plot_levels_by_time(dat, variable, plot_levels, figsize=(10,5), col_wrap=2):
    """
    Plot variable vs time by level for each key in 'dat'.
    
    Arguments:
        dat: Data to plot. Should be dictionary.
        variable: The variable to plot.
        plot_levels: Values of 'level' to plot from dat.
        figsize: Figure size width x height.
        col_wrap: Number of columns per row.
    """
    
    for res in dat.keys():
        dat[res][variable].sel(level=plot_levels).plot(hue='Dataset', col='level', col_wrap=col_wrap, 
                                                       figsize=figsize, sharey=False).set_titles('{value} hPa')
        plt.suptitle(res, y=1.01)
        plt.show()
           
def plot_RCE_tq_profiles(inputs, wrfvars, start_time, end_time, 
                         p_from=200, p_to=80, ncols=5, nrows=2, figsize=[13,6]):
    """
    Plot the RCE T and q profiles and the profiles from the output data to compare.
    
    Arguments:
        inputs: Input directories by resolution then dataset (dict of dicts).
        wrfvars: Data to plot (dictionary by resolution).
        start_time, end_time: Times over which to find the mean profile.
        p_from, p_to: Pressure ranges to plot [hPa].
        nrows, ncols: Rows/columns to use.
        figsize: Figure size width x height.
    """

    for res in inputs.keys():
        RCE_profiles = target_tk_profiles(
            wrfout_file=(inputs[res]['Control'] + '/wrfout_d01_' + start_time[res] + '_00:00:00'),
            pres=wrfvars[res].pres.sel(Dataset='Control'), start=start_time[res], end=end_time[res])
        
        # Show values in all non-RCE datasets.
        plot_tq_stratosphere(dat=wrfvars[res].drop_sel(Dataset='RCE').sel(time=slice(start_time[res], end_time[res])), 
                             title=res, RCE_profiles=RCE_profiles, p_from=p_from, p_to=p_to, ncols=ncols, 
                             nrows=nrows, figsize=figsize)
        
def input_map(perts, basedir):
    """
    Construct an inputs dictionary based on a base input directory and a list of perturbations for T and q.
    
    Arguments:
        perts: Dictionary of perturbations containing res, levels, T, and q. 
        basedir: Base directory to use.
    """
    
    inputs = {}
    for i, res in enumerate(perts['res']):
        inputs[res] = {'RCE': basedir + perts['dir'][i] + '/RCE/',
                     'Control': basedir + perts['dir'][i] + '/control/'}
        for level in perts['levels']:        
            for T in perts['T']: 
                pert_name = 'T ' + T + ' @' + level
                d = basedir + perts['dir'][i] + '/pert_' + level + 'hPa_T_' + T + 'K/'
                if os.path.exists(d):
                    inputs[res][pert_name] = d
            for q in perts['q']:
                pert_name = 'q ' + q + ' @' + level
                d = basedir + perts['dir'][i] + '/pert_' + level + 'hPa_q_' + q + 'kgkg-1/'
                if os.path.exists(d):
                    inputs[res][pert_name] = d
    
    return(inputs)

def shear_profile(z, z_s=3000, U_s=22):
    """
    Return a straight shear profile for a horizontal wind (U or V) at different heights. 
    The shear amount is formulated using Equation 14 in Richardson et al., 2007 (DOI: 10.1175/MWR3463.1). 
    
    Arguments:
        z: The heights for which to calculate the wind magnitude in either U or V [m].
        z_s: Parameter one [m].
        U_s: Parameter two (maximum wind speed at about z_s*2) [m s-1].
    """
    
    wind = U_s * np.tanh(z/z_s)
    return wind

def add_mass_flux(wrfvars):
    """
    Add mass flux to the datasets.
    
    Arguments:
        wrfvars: Data to add to (dictionary by resolution).
        
    Returns:
        wrfvars with mass flux added.
    """
    
    for res in wrfvars.keys():
        wrfvars[res]['level_pressure'], _ = xarray.broadcast(wrfvars[res].level, wrfvars[res].wa)
        wrfvars[res]['air_density'] = atm.density(p=wrfvars[res].level_pressure, 
                                                  q_v=wrfvars[res].q, 
                                                  theta=wrfvars[res].T+300)
        wrfvars[res]['conv_mass_flux'] = (wrfvars[res].air_density * 
                                          wrfvars[res].updraft_proportion * 
                                          wrfvars[res].mean_updraft)
        
        wrfvars[res] = wrfvars[res].drop('level_pressure')
        wrfvars[res].air_density.attrs['long_name'] = 'Density of air'
        wrfvars[res].air_density.attrs['units'] = 'kg m-3'
        
        wrfvars[res].conv_mass_flux.attrs['long_name'] = 'Convective mass flux'
        wrfvars[res].conv_mass_flux.attrs['units'] = 'kg m-2 s-1'
        
    return wrfvars

def plot_mean_profiles(profs, variables, figsize=(13,4), dataset='Control', resolutions=['4 km', '1 km', '100 m'], ylim=(1000,200), relabel={}, retick={}, file=None):
    """
    Plot mean profiles for a given dataset, by resolution.
    
    Arguments:
        profs: Mean profiles to plot.
        variables: List of variables to plot.
        figsize: Figure size.
        dataset: The dataset to plot.
        resolutions: The resolutions to plot.
        ylim: Y limits.
        relabel: {variable: label} dictionary with new labels for x axes.
        retick: {variable: ticks} dictionary with new ticks for x axes.
        file: File to save plot to.
    """
    
    fig, axs = plt.subplots(ncols=len(variables), nrows=1, figsize=figsize, gridspec_kw={'wspace': 0.1})
    
    for res in resolutions:
        for i, variable in enumerate(variables):

            p = profs[res][variable]
            p.sel(Dataset=dataset).plot(ax=axs[i], y='level', yincrease=False, hue=res, label=f'{res} (WRF)')
            axs[i].set_title('')
            axs[i].set_ylabel('')
            if i > 0:
                axs[i].set_yticks([])

            if variable in retick:
                axs[i].set_xticks(retick[variable])

            if variable in relabel:
                axs[i].set_xlabel(relabel[variable])

            axs[i].set_ylim(ylim)
            axs[i].ticklabel_format(style='sci', axis='x', useMathText=True, scilimits=(-4, 5))
            

    axs[-1].legend()
    axs[0].set_ylabel('Pressure [hPa]')
    sns.move_legend(axs[-1], "upper left", bbox_to_anchor=(1, 1))

    if file is not None:
        plt.savefig(file, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.show()
        
def MONC_data(path='analysis/data/',
              files={'Chimene_MONC_1km/Responses_from_perturbations_of_minus_0point5_K_per_day_at_415_hPa.csv': ['T -0.5 @412', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_minus_0point5_K_per_day_at_500_hPa.csv': ['T -0.5 @500', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_minus_0point5_K_per_day_at_600_hPa.csv': ['T -0.5 @600', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_minus_0point5_K_per_day_at_730_hPa.csv': ['T -0.5 @730', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_minus_0point5_K_per_day_at_850_hPa.csv': ['T -0.5 @850', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_minus_0point2_g_per_kg_per_day_at_415_hPa.csv': ['q -0.0002 @412', '1 km'], 
                     'Chimene_MONC_1km/Responses_from_perturbations_of_minus_0point2_g_per_kg_per_day_at_500_hPa.csv': ['q -0.0002 @500', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_minus_0point2_g_per_kg_per_day_at_600_hPa.csv': ['q -0.0002 @600', '1 km'], 
                     'Chimene_MONC_1km/Responses_from_perturbations_of_minus_0point2_g_per_kg_per_day_at_730_hPa.csv': ['q -0.0002 @730', '1 km'], 
                     'Chimene_MONC_1km/Responses_from_perturbations_of_minus_0point2_g_per_kg_per_day_at_850_hPa.csv': ['q -0.0002 @850', '1 km'], 
                     'Chimene_MONC_1km/Responses_from_perturbations_of_plus_0point2_g_per_kg_per_day_at_415_hPa.csv': ['q 0.0002 @412', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_plus_0point2_g_per_kg_per_day_at_500_hPa.csv': ['q 0.0002 @500', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_plus_0point2_g_per_kg_per_day_at_600_hPa.csv': ['q 0.0002 @600', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_plus_0point2_g_per_kg_per_day_at_730_hPa.csv': ['q 0.0002 @730', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_plus_0point2_g_per_kg_per_day_at_850_hPa.csv': ['q 0.0002 @850', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_plus_0point5_K_per_day_at_415_hPa.csv': ['T 0.5 @412', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_plus_0point5_K_per_day_at_500_hPa.csv': ['T 0.5 @500', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_plus_0point5_K_per_day_at_600_hPa.csv': ['T 0.5 @600', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_plus_0point5_K_per_day_at_730_hPa.csv': ['T 0.5 @730', '1 km'],
                     'Chimene_MONC_1km/Responses_from_perturbations_of_plus_0point5_K_per_day_at_850_hPa.csv': ['T 0.5 @850', '1 km'],
                     'Chimene_MONC_250m/Responses_from_perturbations_of_plus_0point5_K_per_day_at_500_hPa_HorRes_of_250m.csv': ['T 0.5 @500', '250 m'], 
                     'Chimene_MONC_250m/Responses_from_perturbations_of_plus_0point5_K_per_day_at_600_hPa_HorRes_of_250m.csv': ['T 0.5 @600', '250 m'],
                     'Chimene_MONC_250m/Responses_from_perturbations_of_plus_0point5_K_per_day_at_730_hPa_HorRes_of_250m.csv': ['T 0.5 @730', '250 m'],
                     'Chimene_MONC_250m/Responses_from_perturbations_of_plus_0point5_K_per_day_at_850_hPa_HorRes_of_250m.csv': ['T 0.5 @850', '250 m'],
                     'Chimene_MONC_500m/Responses_from_perturbations_of_plus_0point5_K_per_day_at_850_hPa_HorRes_of_500m.csv': ['T 0.5 @850', '500 m']}):
    all_dat = []
    for file, [level, res] in files.items():
        dat = pd.read_csv(f'{path}/{file}')
        dat['pert'] = level
        dat['res'] = res
        all_dat.append(dat.reset_index())

    monc = pd.concat(all_dat).reset_index()
    monc = monc.rename(columns={'Pressure (hPa)': 'pressure',
                                'Delta_theta (K)': 'T',
                                'Delta_Temp (K)': 'tk',
                                'Delta_qv (g/kg)': 'q',
                                'Delta_qliq (g/kg)': 'qcloud',
                                'Delta_qice (g/kg)': 'qice',
                                'Delta_qsnow (g/kg)': 'qsnow',
                                'Delta_qrain (g/kg)': 'qrain',
                                'Delta_qgraupel (g/kg)': 'qgraup'})
    monc = monc.drop(columns=['level_0', 'index'])
    return monc

def responses_grid(diffs, comp, pos_perts, neg_perts, variables, figsize=(20,20), lims={'tk': [-0.2, 1.2], 
                                                                                        'q': [-0.1, 0.55],
                                                                                        'T': [-0.2, 1.2],
                                                                                        'qcloud': np.array([-1.5, 1.4])*1e-3,
                                                                                        'qice': np.array([-0.6, 0.2])*1e-3,
                                                                                        'qsnow': np.array([-20, 20])*1e-3,
                                                                                        'qrain': np.array([-1, 1.8])*1e-3,
                                                                                        'qgraup': np.array([-2.5, 1])*1e-3}):
    """
    Plot a grid of responses with perturbations in rows, variables in columns.
    
    Arguments:
        diff: 
    """
    
    # Perts in rows, variables in columns.
    fig, axs = plt.subplots(nrows=len(pos_perts), 
                            ncols=len(variables), figsize=figsize,
                            gridspec_kw={'hspace':0.35})

    for i in np.arange(len(pos_perts)):
        for j, v in enumerate(variables):
            (diffs.sel(Dataset=neg_perts[i])[v]*-1).plot(y='level', color='green', ax=axs[i,j])
            diffs.sel(Dataset=pos_perts[i])[v].plot(y='level', color='blue', ax=axs[i,j])

            att=diffs.sel(Dataset=pos_perts[i])[v].attrs
            lab = f'{att["long_name"].replace("mixing ratio", "MR")} [{att["units"]}]'
            axs[i,j].set_xlabel(lab)
            axs[i,j].axvline(0, color='red')
            axs[i,j].set_title(pos_perts[i])
            axs[i,j].invert_yaxis()
            axs[i,j].xaxis.set_major_locator(plt.MaxNLocator(3))
            axs[i,j].ticklabel_format(useOffset=False, style='plain', scilimits=(-3,4)) 
            axs[i,j].set_xlim(lims[v][0], lims[v][1])
            axs[i,j].set_ylim(1000,200)
            if j > 0:
                axs[i,j].set_ylabel('')
                axs[i,j].set_yticks([])
                
            if comp is not None:
                pos = comp[comp.pert == pos_perts[i]]
                neg = comp[comp.pert == neg_perts[i]]
                
                if len(pos) > 0:
                    axs[i,j].plot(pos[v], pos.pressure, color='blue', linestyle='--')
                    
                if len(neg) > 0:
                    axs[i,j].plot(neg[v]*-1, neg.pressure, color='green', linestyle='--')
            