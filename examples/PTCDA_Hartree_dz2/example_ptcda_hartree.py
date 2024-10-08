import os
import subprocess
from pathlib import Path

from ppafm.cli.generateElFF import main as generate_elff
from ppafm.cli.generateLJFF import main as generate_ljff
from ppafm.cli.plot_results import main as plot_results
from ppafm.cli.relaxed_scan import main as relaxed_scan


def example_ptcda_hartree():
    script_location = Path(__file__).absolute().parent

    # Change directory to the location of this script
    os.chdir(script_location)

    if not Path("LOCPOT.xsf").exists():
        subprocess.run(["wget", "--no-check-certificate", "https://www.dropbox.com/s/18eg89l89npll8x/LOCPOT.xsf.zip"])
        subprocess.run(["unzip", "LOCPOT.xsf.zip"])

    generate_ljff(["--input", "LOCPOT.xsf"])
    generate_elff(["--input", "LOCPOT.xsf", "--tip", "dz2"])
    relaxed_scan(["--klat", "0.5", "--charge", "-0.10"])
    plot_results(["--klat", "0.5", "--charge", "-0.10", "--Amplitude", "0.5", "--df"])


if __name__ == "__main__":
    example_ptcda_hartree()
