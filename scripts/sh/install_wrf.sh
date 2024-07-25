#!/bin/bash

# install_wrf.sh
#
# Replace files in a WRF installation with symlinks to the modified
# files in this directory.
#
# Author: Tim Raupach <t.raupach@unsw.edu.au>

if [ $# -ne 1 ]; then
    echo "install_wrf.sh: replace files in a WRF install with symlinks to modified versions."
    echo 'Usage: install.sh <wrf_base_directory>'
    exit
fi

dir=$1

if [ ! -d WRFV3 ]; then
    echo "This script should be run from the modified WRF code directory containing WRFV3."
    exit
fi

if [ ! -d $dir/WRFV3 ]; then
    echo "WRF directory must contain WRFV3 directory."
    exit
fi

echo "Installing modified files into directory: ${dir}"

files=`find WRFV3 -type f`

for file in $files; do
    echo $file

    if [ -e ${dir}/${file}_original ]; then
	echo "ERROR: expecting that ${dir}/${file}_original does not already exist. Stopping."
	exit
    fi
	

    if [ -e $dir/$file ]; then
	cp $dir/$file ${dir}/${file}_original
    fi
    ln -rsf $file $dir/$file 
done
