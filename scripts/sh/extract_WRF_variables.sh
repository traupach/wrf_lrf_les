#!/bin/bash
# Load modules and run a script to extract selected variables from a given file.
# Author: Tim Raupach <t.raupach@unsw.edu.au>

. /etc/bashrc # Required to load environment/module command.

export USER=tr2908
module use /g/data3/hh5/public/modules
module load conda/analysis3-24.01

SCRIPT="$HOME/git/wrf_lrf_les/scripts/python/extract_WRF_vars.py"

if [ $# -ne 1 ]; then
    echo "extract_WRF_variables.sh: Extract analysis variables from wrfout file."
    echo "Usage: extract_WRF_variables.sh <wrfout_file>."
    exit
fi
WRF_FILE=$1

# Extract the variables using the python script.
${SCRIPT} ${WRF_FILE}

