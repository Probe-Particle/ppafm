import os
from pathlib import Path

from ppafm.cli.conv_rho import main as generate_conv_rho
from ppafm.cli.generateDFTD3 import main as generate_dftd3
from ppafm.cli.generateElFF import main as generate_elff
from ppafm.cli.plot_results import main as plot_results
from ppafm.cli.relaxed_scan import main as relaxed_scan
from ppafm.data import download_dataset


def example_pyridine_density_overlap():
    script_location = Path(__file__).absolute().parent

    # Change directory to the location of this script
    os.chdir(script_location)

    download_dataset("pyridine", "sample")
    download_dataset("CO-tip-densities", "tip")

    generate_conv_rho("-s sample/CHGCAR.xsf -t tip/density_CO.xsf -B 1.0 -E".split())
    generate_elff("-i sample/LOCPOT.xsf --tip_dens tip/density_CO.xsf --Rcore 0.7 -E --doDensity".split())
    generate_dftd3("-i sample/LOCPOT.xsf --df_name PBE".split())

    relaxed_scan("-k 0.25 -q 1.0 --noLJ --Apauli 18.0 --bDebugFFtot".split())  # Note the --noLJ for loading separate Pauli and vdW instead of LJ force field
    plot_results("-k 0.25 -q 1.0 -a 2.0 --df".split())


if __name__ == "__main__":
    example_pyridine_density_overlap()
