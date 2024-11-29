#!/bin/bash
#$ -S /bin/sh
#$ -cwd
#$ -V
#$ -q all.q
#$ -pe gau 1
#$ -o $JOB_NAME.o$JOB_ID.log
#$ -e $JOB_NAME.e$JOB_ID.err

module purge
module load gau/g16.c01

g16 < MOL_1_6.com  > MOL_1_6.log
