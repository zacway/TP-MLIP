#!/bin/bash

#PBS -q beta
#PBS -l select=4:ncpus=24
#PBS -l walltime=23:00:00

#PBS -o output
#PBS -e error

#PBS -N w_bulk

CP2K_EXE=/home/francela/apps/cp2k-7.1/exe/Linux-x86-64-intelx/cp2k.popt

module purge
module load intel/intel-cc-18.0/18.0
module load intel/intel-cmkl-18.0/18.0
module load intel/intel-fc-18.0/18.0
module load python-2.7
module load intel/intel-mpi/2018.2

export I_MPI_COLL_INTRANODE=pt2pt
export I_MPI_ADJUST_REDUCE=1
export I_MPI_ADJUST_BCAST=1

cd ${PBS_O_WORKDIR}

time -p mpirun -np 96 ${CP2K_EXE} -i bulk.inp  -o bulk.out
