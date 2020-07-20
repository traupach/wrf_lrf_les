#!/bin/tcsh
#PBS -P up6
#PBS -l storage=gdata/up6
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=192GB       
#PBS -l ncpus=48
#PBS -j oe
#PBS -l wd
#PBS -W umask=0022
#PBS -N wrf_job

module load openmpi
ulimit -s unlimited
limit stacksize unlimited

echo 'Running in directory:' `pwd`
env > run_environment.txt

echo 'Running wrf.exe using $PBS_NCPUS mpi nodes...'
time mpirun -np $PBS_NCPUS -report-bindings ./wrf.exe
