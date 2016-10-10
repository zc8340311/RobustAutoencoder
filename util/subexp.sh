#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#

date
#ln -s /usr/lib64/libblas.so.3 /home/czhou2/root/lib/libblas.so.3
#ln -s /usr/lib64/liblapack.so.3 /home/czhou2/root/lib/liblapack.so.3
export BLAS=/home/czhou2/root/lib
export LAPACK=/home/czhou2/root/lib
module load python27
python experiment.py
date
