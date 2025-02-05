#!/usr/bin/env python3


from ppafm.data import download_dataset
from ppafm.ocl.AFMulator import quick_afm

download_dataset("PTCDA-Ag", "sample")

quick_afm(
    file_path="./sample/LOCPOT.xsf",
    scan_size=(20, 20),  # Physical size of scan region
    offset=(0, 0),  # The scan region is by default centered on average coordinate, add offset here
    distance=11.0,  # Furthest distance of probe from sample
    scan_step=(0.1, 0.1, 0.1),  # Step size in (x, y, z)
    num_heights=50,  # Number of scan heights (approaching from the furthest distance)
    amplitude=1.0,  # Cantilever oscillation amplitude (peak-to-peak)
    probe_type=8,  # Probe element
    tip="dz2",  # Tip density multipole type
    charge=-0.05,  # Tip charge/multipole magnitude
    sigma=0.71,  # Tip charge distribution width
    out_dir=None,  # Image output directory (None = automatic)
)
