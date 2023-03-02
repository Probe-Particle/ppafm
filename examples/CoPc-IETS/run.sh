#!/bin/bash
# test for calculating the IETS figures according to one showed in PRL 113, 226101 (2014) Q=0.0
PPAFM_DIR='../../ppafm/cmdline'

python ${PPAFM_DIR}/generateLJFF.py -i answer.xyz
python ${PPAFM_DIR}/relaxed_scan.py --vib 3
python ${PPAFM_DIR}/plot_results.py --iets 16 0.0015 0.0007
# since now there is a proper width (w) of the Gaussian 0.0007 is 0.0007^2 * 2 = 1.0 as in the original paper
