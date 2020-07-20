#!/bin/bash
# Load modules and run a script to extract selected variables from a given file.
# Author: Tim Raupach <t.raupach@unsw.edu.au>

. /etc/bashrc # Required to load environment/module command.
module load python3/3.7.4
module load netcdf/4.7.3
module load hdf5/1.10.5

SCRIPT="$HOME/git/wrf_lrf_les/scripts/python/extract_WRF_vars.py"

if [ $# -ne 1 ]; then
    echo "extract_WRF_variables.sh: Extract analysis variables from wrfout file."
    echo "Usage: extract_WRF_variables.sh <wrfout_file>."
    exit
fi
WRF_FILE=$1

# Extract the variables using the python script.
${SCRIPT} ${WRF_FILE}

