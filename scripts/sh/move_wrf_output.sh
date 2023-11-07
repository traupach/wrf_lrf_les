#!/bin/bash

if [ $# -ne 2 ]; then
    echo "move_wrf_output.sh: Move all WRF output after a run to a specified directory.";
    echo "Logs and run-specific information will be moved to <dir>/runs/<runname>."
    echo "Usage: move_wrf_output.sh <dir> <runname>";
    exit
fi

dir=$1
run=$2

if [ ! -d $dir ]; then
    echo "Creating output directory.";
    mkdir $dir
fi


if [ ! -d $dir/runs ]; then
    echo "Creating runs subdirectory.";
    mkdir $dir/runs
fi

if [ ! -d $dir/runs/$run ]; then
    echo "Creating run subdirectory.";
    mkdir $dir/runs/$run
fi

## Settings.
cp -i namelist.input $dir/runs/$run
mv -i namelist.output $dir/runs/$run

## Ask whether to over-write profiles.
cp -i input_sounding $dir
cp -i U_target $dir
cp -i V_target $dir
cp -i RCE_T $dir
cp -i RCE_q $dir

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

## Log files.
if [ -e ideal.error.0000 ]; then
    mv -i ideal.error.0000 $dir
    mv -i ideal.out.0000 $dir
fi
tar cvfz $dir/runs/$run/rsl_logs.tar.gz rsl* --remove-files
mv -i wrf_job.* $dir/runs/$run
mv -i run_environment.txt $dir/runs/$run

mv -i wrfout_* $dir
mv -i wrfrst_* $dir
