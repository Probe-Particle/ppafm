import os
import urllib.request
import zipfile
from pathlib import Path

from ppafm.cli.conv_rho import main as generate_conv_rho
from ppafm.cli.generateDFTD3 import main as generate_dftd3
from ppafm.cli.generateElFF import main as generate_elff
from ppafm.cli.plot_results import main as plot_results
from ppafm.cli.relaxed_scan import main as relaxed_scan


def example_pyridine_density_overlap():
    script_location = Path(__file__).absolute().parent

    # Change directory to the location of this script
    os.chdir(script_location)

    if not Path("LOCPOT.xsf").exists():
        urllib.request.urlretrieve("https://zenodo.org/records/14222456/files/pyridine.zip?download=1", "sample.zip")
        # Unzip the file using python tools only
        with zipfile.ZipFile("sample.zip", "r") as zip_ref:
            zip_ref.extractall(".")

    if not Path("tip/density_CO.xsf").exists():
        urllib.request.urlretrieve("https://zenodo.org/records/14222456/files/CO_tip.zip?download=1", "tip.zip")
        # Unzip the file using python tools only
        with zipfile.ZipFile("tip.zip", "r") as zip_ref:
            # Make "tip" directory if it does not exist
            Path("tip").mkdir(exist_ok=True)
            zip_ref.extractall("tip")

    generate_conv_rho(["-s", "LOCPOT.xsf", "-t", "tip/density_CO.xsf", "-B", "1.0", "-E"])
    generate_elff(["-i", "LOCPOT.xsf", "--tip_dens", "tip/density_CO.xsf", "--Rcore", "0.7", "--energy", "--doDensity"])
    generate_dftd3(["--input", "LOCPOT.xsf", "--df_name", "PBE"])

    relaxed_scan(
        ["--klat", "0.25", "--charge", "1.0", "--noLJ", "--Apauli", "18.0", "--bDebugFFtot"]
    )  # Note the --noLJ for loading separate Pauli and vdW instead of LJ force field
    plot_results(["--klat", "0.25", "--charge", "1.0", "--Amplitude", "2.0", "--df"])


if __name__ == "__main__":
    example_pyridine_density_overlap()
