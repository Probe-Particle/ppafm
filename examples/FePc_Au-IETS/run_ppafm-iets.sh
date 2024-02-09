#!/bin/bash
OMP=0	# 0 = 'False' , 1 = 'True'
if [ $OMP -eq 1 ]
then
    export OMP_NUM_THREADS=8
fi

echo "OMP_NUM_THREADS:"
echo $OMP_NUM_THREADS
echo "Now the tests:"

echo "test for the PP-STM code: IETS"
ppafm-generate-ljff -i geom-cube.in -f npy
ppafm-relaxed-scan --pos -f npy --vib 3
ppafm-plot-results --iets 16 0.0015 0.001 -f npy --atoms #--save_df
#
#python3 PPSTM/IETS_test_FePc.py
#python3 PPdos_simple.py
#Apython3 PPSTM_simple.py

echo "Now all things made, before submiting, please run clean.sh!"
