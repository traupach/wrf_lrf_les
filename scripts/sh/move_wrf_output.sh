#!/bin/bash

if [ $# -ne 1 ]; then
    echo "move_wrf_output.sh: Move all WRF output after a run to a specified directory.";
    echo "Usage: move_wrf_output.sh <dir>.";
    exit
fi

dir=$1

if [ ! -d $dir ]; then
    echo "Error: output directory does not exist.";
    exit;
fi

if [ ! -z "`ls $dir`" ]; then
    echo "Error: output directory must be empty.";
    exit;
fi

## Settings.
cp namelist.input $dir
mv namelist.output $dir
cp input_sounding $dir
cp U_target $dir
cp V_target $dir

if [ -e T_target ]; then
    cp T_target $dir
fi
if [ -e q_target ]; then
    cp q_target $dir
fi

## Output and restart files.
mv wrfinput_d01 $dir
mv wrfout_* $dir
mv wrfrst_* $dir

## Log files.
mv ideal.error.0000 $dir
mv ideal.out.0000 $dir
mv rsl* $dir
mv wrf_job.* $dir
mv run_environment.txt $dir
