import os
from pathlib import Path

from ppafm.cli.generateElFF_point_charges import main as generate_elff_point_charges
from ppafm.cli.generateLJFF import main as generate_ljff
from ppafm.cli.plot_results import main as plot_results
from ppafm.cli.relaxed_scan import main as relaxed_scan


def example_ptcda_singe():
    script_location = Path(__file__).absolute().parent

    # Change directory to the location of this script
    os.chdir(script_location)

    generate_ljff(["--input", "PTCDA.xyz"])
    generate_elff_point_charges(["--input", "PTCDA.xyz", "--tip", "s"])
    relaxed_scan(["-k", "0.5", "-q", "-0.10"])
    plot_results(["--klat", "0.5", "--charge", "-0.10", "--arange", "0.5", "2.0", "2", "--df"])


if __name__ == "__main__":
    example_ptcda_singe()
