#!/usr/bin/env python3

import tarfile
import zipfile
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

DATASET_URLS = {
    "CO-tip-densities": "https://zenodo.org/records/10563098/files/CO_tip_densities.tar.gz?download=1",
    "dft-afm": "https://zenodo.org/records/10563098/files/dft-afm.tar.gz?download=1",
    "hartree-density": "https://zenodo.org/records/10563098/files/hartree-density.tar.gz?download=1",
    "FFPB-KPFM-hartree": "https://zenodo.org/records/10563098/files/KPFM_hartree.tar.gz?download=1",
    "BrClPyridine": "https://zenodo.org/records/14222456/files/pyridineBrCl.zip?download=1",
}


def _print_progress(block_num: int, block_size: int, total_size: int):
    if total_size == -1:
        return
    delta = block_size / total_size * 100
    current_size = block_num * block_size
    percent = current_size / total_size * 100
    percent_int = int(percent)
    if (percent - percent_int) > 1.0001 * delta:
        # Only print when crossing an integer percentage
        return
    if block_num > 0:
        print("\b\b\b", end="", flush=True)
    if current_size < total_size:
        print(f"{percent_int:2d}%", end="", flush=True)
    else:
        print("Done")


def _common_parent(paths):
    path_parts = [list(Path(p).parts) for p in paths]
    common_part = Path()
    for parts in zip(*path_parts):
        p = parts[0]
        if all(part == p for part in parts):
            common_part /= p
        else:
            break
    return common_part


def _extract_targz(archive_path, target_dir):
    with tarfile.open(archive_path, "r") as ft:
        print("Reading tar archive files...")
        members = []
        base_dir = _common_parent(ft.getnames())
        for m in ft.getmembers():
            if m.isfile():
                # relative_to(base_dir) here gets rid of a common parent directory within the archive (if any),
                # which makes it so that we can just directly extract the files to the target directory.
                m.name = Path(m.name).relative_to(base_dir)
                members.append(m)
        print(f"Extracting dataset to `{target_dir}`: ", end="", flush=True)
        for i, m in enumerate(members):
            _print_progress(i, 1, len(members) - 1)
            ft.extract(m, target_dir)


def _extract_zip(archive_path, target_dir):
    with zipfile.ZipFile(archive_path, "r") as ft:
        print("Reading zip archive files...")
        members = []
        base_dir = _common_parent(ft.namelist())
        for m in ft.infolist():
            if not m.is_dir():
                # relative_to(base_dir) here gets rid of a common parent directory within the archive (if any),
                # which makes it so that we can just directly extract the files to the target directory.
                m.filename = str(Path(m.filename).relative_to(base_dir))
                members.append(m)
        print(f"Extracting dataset to `{target_dir}`: ", end="", flush=True)
        for i, m in enumerate(members):
            _print_progress(i, 1, len(members) - 1)
            ft.extract(m, target_dir)


def download_dataset(name: str, target_dir: PathLike):
    """
    Download and unpack a dataset to a target directory.

    The following datasets are available:

        - ``'CO-tip-densities'``: https://doi.org/10.5281/zenodo.10563098 - CO_tip_densities.tar.gz
        - ``'dft-afm'``: https://doi.org/10.5281/zenodo.10563098 - dft-afm.tar.gz
        - ``'hartree-density'``: https://doi.org/10.5281/zenodo.10563098 - hartree-density.tar.gz
        - ``'FFPB-KPFM-hartree'``: https://doi.org/10.5281/zenodo.10563098 - KPFM_hartree.tar.gz
        - ``'BrClPyridine'``: https://doi.org/10.5281/zenodo.14222456 - pyridineBrCl.zip: hartree potential and electron density for the BrClPyridine example.

    Arguments:
        name: Name of dataset to download.
        target_dir: Directory where dataset will be unpacked into. The directory and its parents will be created if they
            do not exist already. If the directory already exists and is not empty, then the operation is aborted.
    """
    try:
        dataset_url = DATASET_URLS[name]
    except KeyError:
        raise ValueError(f"Unrecognized dataset name `{name}`")

    target_dir = Path(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Target directory `{target_dir}` exists and is not empty. Skipping downloading dataset `{name}`.")
        return

    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / f"dataset_{name}"
        print(f"Downloading dataset `{name}`: ", end="")
        _, response = urlretrieve(dataset_url, temp_file, _print_progress)
        original_file_name = response.get_filename()
        target_dir.mkdir(exist_ok=True, parents=True)
        if original_file_name.endswith(".tar.gz"):
            _extract_targz(temp_file, target_dir)
        elif original_file_name.endswith(".zip"):
            _extract_zip(temp_file, target_dir)
        else:
            raise RuntimeError(f"Uknown file extension in `{original_file_name}`.")
