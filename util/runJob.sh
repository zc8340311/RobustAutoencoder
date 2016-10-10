#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#

date
module load python27
python DeepRAE.py
date
echo "success"
