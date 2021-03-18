#!/bin/bash

# setup_wrf_run.sh: Set up a new WRF runtime directory.
# Author: Tim Raupach <t.raupach@unsw.edu.au>

# Exit if any error occurs.
set -e

if [ $# -ne 4 ]; then
    echo 'setup_wrf_run.sh: Set up a new WRF runtime directory.'
    echo 'Usage: setup_wrf_run.sh <outdir> <gitdir> <wrfdir> <configuration>'
    echo ' outdir: The new directory to create.'
    echo ' gitdir: The git directory for wrf_lrf_les.'
    echo ' wrfdir: The directory where WRF is installed.'
    echo ' configuration: A configuration for the setup; see <gitdir>/runtime.'
    exit
fi

outdir=$1
gitdir=$2
wrfdir=$3
config=$4

if [ -d $outdir ]; then
    echo 'Error: output directory already exists.'
    exit
fi

dir=`pwd`

mkdir $outdir
cd $outdir

echo 'Linking WRF files.'
ln -s $wrfdir/WRFV3/run/bulkdens.asc_s_0_03_0_9 .
ln -s $wrfdir/WRFV3/run/bulkradii.asc_s_0_03_0_9 .
ln -s $wrfdir/WRFV3/run/capacity.asc .
ln -s $wrfdir/WRFV3/run/coeff_p.asc .
ln -s $wrfdir/WRFV3/run/coeff_q.asc .
ln -s $wrfdir/WRFV3/run/constants.asc .
ln -s $wrfdir/WRFV3/run/kernels.asc_s_0_03_0_9 .
ln -s $wrfdir/WRFV3/run/kernels_z.asc .
ln -s $wrfdir/WRFV3/run/masses.asc .
ln -s $wrfdir/WRFV3/run/termvels.asc .
ln -s $wrfdir/WRFV3/run/RRTMG_LW_DATA .
ln -s $wrfdir/WRFV3/run/RRTMG_LW_DATA_DBL .
ln -s $wrfdir/WRFV3/run/RRTMG_SW_DATA .
ln -s $wrfdir/WRFV3/run/RRTMG_SW_DATA_DBL .
ln -s $wrfdir/WRFV3/run/RRTM_DATA .
ln -s $wrfdir/WRFV3/run/RRTM_DATA_DBL .
ln -s $wrfdir/WRFV3/run/LANDUSE.TBL .
ln -s $wrfdir/WRFV3/run/ozone.formatted .
ln -s $wrfdir/WRFV3/run/ozone_lat.formatted .
ln -s $wrfdir/WRFV3/run/ozone_plev.formatted .

echo 'Linking executables.'
ln -s $wrfdir/WRFV3/main/ideal.exe .
ln -s $wrfdir/WRFV3/main/wrf.exe .

echo 'Copying runtime scripts.'
cp $gitdir/scripts/sh/run_ideal.sh .
cp $gitdir/scripts/sh/run_wrf.sh .

echo 'Copying configuration files.'
cp $gitdir/runtime/namelist.input.$config namelist.input
cp $gitdir/runtime/input_sounding.$config input_sounding
cp $gitdir/runtime/U_target.$config U_target
cp $gitdir/runtime/V_target.$config V_target
cp $gitdir/runtime/RCE_T.$config RCE_T
cp $gitdir/runtime/RCE_q.$config RCE_q

echo 'Remember to edit namelist.input and run_wrf.sh.'
cd $dir
