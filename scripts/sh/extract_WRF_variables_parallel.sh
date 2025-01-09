#!/bin/bash
# Extract selected variables from wrfout files in the current directory, in parallel.
# Author: Tim Raupach <t.raupach@unsw.edu.au>

#PBS -q normal
#PBS -P up6
#PBS -l storage=gdata/up6+gdata/hh5
#PBS -l ncpus=19
#PBS -l walltime=03:00:00
#PBS -l mem=192gb
#PBS -j oe
#PBS -W umask=0022
#PBS -l wd
#PBS -N extract_var_job

module load parallel

SCRIPT="bash ~/git/wrf_lrf_les/scripts/sh/extract_WRF_variables.sh"

# ${PBS_NCPUS} is total number of CPUs to run on; {%} is replaced with
# the CPU slot from 1..${PBS_NCPUS}.
parallel -j ${PBS_NCPUS} pbsdsh -n {%} -- ${SCRIPT} {} ::: wrfout*
