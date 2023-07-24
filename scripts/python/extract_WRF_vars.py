#!/usr/bin/env python3

# extract_WRF_vars.py
# Extract, interpolate and average WRF variables. Write output to current dir.
# Author: Tim Raupach <t.raupach@unsw.edu.au>

from netCDF4 import Dataset
import numpy as np
import xarray
import wrf
import os
import sys

if(len(sys.argv) != 2):
    print("extract_WRF_vars.py -- extract, interpolate and average WRF " +
          "variables. Write output to current dir.")
    print("")
    print("Usage: extract_WRF_vars.py <wrfout_file>")
    sys.exit(1)
    
infile = str(sys.argv[1])
print("Processing file: " + infile)

outfile = os.path.basename(infile).replace('wrfout', 'wrfvars') + '.nc'
if(os.path.exists(outfile)):
    print("ERROR: output file already exists: " + outfile)
    sys.exit(1)

# Get full pressure at each 3D model point, to determine interpolation.
nc = Dataset(infile)
pres = wrf.getvar(nc, "pres", timeidx=wrf.ALL_TIMES, units="hPa")

def horiz_summary(nc, varname,
                  operation='mean',
                  rename=None, 
                  interp=True,
                  vert_pres=pres,
                  vert_levels=np.arange(1000, 90, step=-10),
                  units=None, long_name=None):
    """
    Calculate horizontal summary values for a given WRF variable and
    return an xarray.dataArray including metadata.
    
    Keyword arguments:
    nc -- An open netcdf file.
    varname -- The variable name to read, an argument to wrf.getvar.
    operation -- The summary operation to compute (can be 'mean', 'min', 'max').
    interp -- Interpolate to pressure levels?
    vert_pres -- Pressure values for each coordinate. See wrf.interplevel.
    vert_levels -- Levels to interpolate to in same unit as vert_pres.
                   (by default, 1000 hPa to 200 hPa in 10 hPa increments)
    units, long_name: Metadata to overwrite original data with.

    """

    # Retreive the variable from WRF files.
    orig_var = wrf.getvar(wrfin=nc, varname=varname, timeidx=wrf.ALL_TIMES)

    if(interp):
        # Interpolate it to pressure levels.
        var = wrf.interplevel(field3d=orig_var, vert=vert_pres,
                              desiredlev=vert_levels)
    else:
        var = orig_var
        
    # Take the spatial summary to get values per time and vertical level.
    if operation == 'mean':
        horiz = var.mean(['south_north', 'west_east'])
    elif operation == 'max':
        horiz = var.max(['south_north', 'west_east'])
    elif operation == 'min':
        horiz = var.min(['south_north', 'west_east'])
    elif operation == 'scaled_var':
        horiz = var.var(['south_north', 'west_east']) / var.mean(['south_north', 'west_east'])
    elif operation == 'positive_prop':
        # Determine proportion of region with positive values.
        horiz = xarray.ones_like(var).where(var > 0,
                                            other=0).mean(['south_north',
                                                           'west_east'])
    elif operation == 'positive_mean':
        # Determine mean of only positive values.
        horiz = var.where(var > 0).mean(['south_north', 'west_east'])
    #elif operation == 'mass_weighted_
    else:
        print('ERROR: horiz_summary: unknown operation ' + operation + '.')
        sys.exit(1)    

    # Remove malformed time strings, keeping only XTIME as the time
    # coordinate array.
    horiz = horiz.drop_vars('Time')

    # Rename the Time dimension to lower case, and rename XTIME
    # coordinates to have the same name.
    horiz = horiz.rename({'Time': 'time', 
                          'XTIME': 'time'})

    # Assign metadata.
    if(interp):
        horiz.level.attrs['units'] = pres.attrs['units']
        horiz.level.attrs['long_name'] = 'Interpolated pressure level'
        
    horiz.time.attrs['units'] = nc.variables['XTIME'].units
    horiz.time.attrs['long_name'] = "Simulation time"
    horiz.name = varname
    if not rename is None:
        horiz.name = rename
    horiz.attrs['units'] = orig_var.attrs['units']
    if not units is None:
        horiz.attrs['units'] = units
    horiz.attrs['long_name'] = orig_var.attrs['description'].lower()
    if not long_name is None:
        horiz.attrs['long_name'] = long_name
    
    desc = ('Values retrieved using wrf-python getvar with varname=\'' +
            varname + '\', ')
    if(interp):
        desc = (desc + 'interpolated to vertical levels using ' +
                'wrf-python.interplevel, ')
    desc = (desc + 'and ' + operation +
            ' calculated across south_north and west_east dimensions.')
    horiz.attrs['description'] = desc
        
    return(horiz)

# Get all the variables required.
dat = xarray.merge([
    # Interpolate the following fields to even pressure levels.
    horiz_summary(nc=nc, varname='QVAPOR', rename='q'),      # Water vapour mixing ratio [kg kg-1].
    horiz_summary(nc=nc, varname='QCLOUD', rename='qcloud'), # Cloud water mixing ratio [kg kg-1].
    horiz_summary(nc=nc, varname='QRAIN', rename='qrain'),   # Rain mixing ratio [kg kg-1].
    horiz_summary(nc=nc, varname='QICE', rename='qice'),     # Ice mixing ratio [kg kg-1].
    horiz_summary(nc=nc, varname='QSNOW', rename='qsnow'),   # Snow mixing ratio [kg kg-1].
    horiz_summary(nc=nc, varname='QGRAUP', rename='qgraup'), # Graupel mixing ratio [kg kg-1].
    horiz_summary(nc=nc, varname='tk'),                      # Temperature [K]. 
    horiz_summary(nc=nc, varname='T'),                       # Perturbation potential temperature [K].   
    horiz_summary(nc=nc, varname='ua'),                      # Wind U on mass points [m s-1].
    horiz_summary(nc=nc, varname='va'),                      # Wind V on mass points [m s-1].
    horiz_summary(nc=nc, varname='RTHRATEN'),                # Theta tendency due to radiation [K s-1].
    horiz_summary(nc=nc, varname='RTHFORCETEN'),             # Theta forcing for LRF [K s-1].
    horiz_summary(nc=nc, varname='RQVFORCETEN'),             # Moisture forcing for LRF [kg kg-1 s-1].

    horiz_summary(nc=nc, varname='wa'),                                # Mean vertical wind on mass points [m s-1].
    horiz_summary(nc=nc, varname='wa', operation='positive_prop',      # Proportion of points with updraft.
                  rename='updraft_proportion', units='-',
                  long_name='Proportion of points with updraft.'),
    horiz_summary(nc=nc, varname='wa', operation='positive_mean',      # Mean updraft.
                  rename='mean_updraft',
                  long_name='Mean vertical wind on updraft points.'),
    
    # Fields for which no interpolation is required, return horizontal mean per (mass-point) eta-level.
    horiz_summary(nc=nc, varname='z', interp=False),                      # Full geopotential height [m].
    horiz_summary(nc=nc, varname='T', rename='eta_T', interp=False),      # Perturbation potential temp [K].
    horiz_summary(nc=nc, varname='P_HYD', rename='pres', interp=False),   # Pressure [hPa].
    horiz_summary(nc=nc, varname='tk', rename='eta_tk', interp=False),    # Temperature [K].
    horiz_summary(nc=nc, varname='QVAPOR', rename='eta_q', interp=False), # Water vapour mixing ratio [kg kg-1].
    
    # 2D fields.
    horiz_summary(nc=nc, varname='pw', interp=False), # Precipitable water [kg m-2].
    horiz_summary(nc=nc, varname='pw', interp=False, # Scaled variance of PW as % of mean.
                  operation='scaled_var', rename='pw_scaled_var', 
                  long_name='Variance/mean of precipitable water.')
])


if 'RTHRELAXTEN' in nc.variables.keys():
    dat = xarray.merge([dat,
                        horiz_summary(nc=nc, varname='RTHRELAXTEN'),  # Theta stratospheric relaxation [K s-1].
                        horiz_summary(nc=nc, varname='RQVRELAXTEN')]) # Moisture stratospheric relaxation [kg kg-1 s-1].

if 'RURELAXTEN' in nc.variables.keys():
    dat = xarray.merge([dat,
                        horiz_summary(nc=nc, varname='RURELAXTEN'),  # U wind relaxation tendency [m s-2].
                        horiz_summary(nc=nc, varname='RVRELAXTEN')]) # V wind relaxation tendency [m s-2].
else:
    # If older version of input file, rename RUFORCETEN to RURELAXTEN in the output.
    dat = xarray.merge([dat,
                        horiz_summary(nc=nc, varname='RUFORCETEN', rename='RURELAXTEN'),  
                        horiz_summary(nc=nc, varname='RVFORCETEN', rename='RVRELAXTEN')]) 

# Check that the time values are the same as XTIME in the original file.
assert all(nc.variables['XTIME'][:] == dat.time.values), "Times do not match input file."

# Record the name of the input file and from where this script was run.
dat.attrs['input_file'] = infile
dat.attrs['run_from_dir'] = os.getcwd()

# Write compressed netcdf output to 'wrfvars' file with same file suffix as input.
comp = dict(zlib=True, shuffle=True, complevel=4)
encoding = {var: comp for var in dat.data_vars}
dat.to_netcdf(path=outfile, encoding=encoding)

