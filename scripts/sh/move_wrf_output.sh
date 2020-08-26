#!/bin/bash

if [ $# -ne 1 ]; then
    echo "move_wrf_output.sh: Move all WRF output after a run to a specified directory.";
    echo "Usage: move_wrf_output.sh <dir>.";
    exit
fi

dir=$1

if [ ! -d $dir ]; then
    echo "Creating output directory.";
    mkdir $dir
fi

## Settings.
cp namelist.input $dir
mv -i namelist.output $dir
cp -i input_sounding $dir
cp -i U_target $dir
cp -i V_target $dir

if [ -e T_target ]; then
    cp -i T_target $dir
fi
if [ -e q_target ]; then
    cp -i q_target $dir
fi

## Output and restart files.
if [ -e wrfinput_d01 ]; then
    mv -i wrfinput_d01 $dir
fi

mv -i wrfout_* $dir
mv -i wrfrst_* $dir

## Log files.
if [ -e ideal.error.0000 ]; then
    mv -i ideal.error.0000 $dir
    mv -i ideal.out.0000 $dir
fi
mv -i rsl* $dir
mv -i wrf_job.* $dir
mv -i run_environment.txt $dir
