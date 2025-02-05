import os
from pathlib import Path

from ppafm.cli.generateElFF import main as generate_elff
from ppafm.cli.generateLJFF import main as generate_ljff
from ppafm.cli.plot_results import main as plot_results
from ppafm.cli.relaxed_scan import main as relaxed_scan
from ppafm.data import download_dataset


def example_ptcda_hartree_dz2():
    script_location = Path(__file__).absolute().parent

    # Change directory to the location of this script
    os.chdir(script_location)

    download_dataset("PTCDA-Ag", "sample")

    generate_ljff(["--input", "sample/LOCPOT.xsf"])
    generate_elff(["--input", "sample/LOCPOT.xsf", "--tip", "dz2"])
    relaxed_scan(["--klat", "0.5", "--charge", "-0.10"])
    plot_results(["--klat", "0.5", "--charge", "-0.10", "--Amplitude", "0.5", "--df"])


if __name__ == "__main__":
    example_ptcda_hartree_dz2()
