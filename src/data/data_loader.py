import tarfile
import os
from tqdm import tqdm


"""
The purpose of this file is to extract the raw data from the .tar file it is provided on.
    - Input: sources.tar
    - Output: 
"""


def extract_tar(source_dir, destination_dir):
    """
    Extracts the signals from the .tar file from the source directory and places the extracted files in the
    destination directory.
    :param source_dir: directory containing the .tar file
    :param destination_dir: directory where extracted files should be placed
    """
    tar_filename = 'core-spectra.tar'
    tar_path = os.path.join(source_dir, tar_filename)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # try:
    #     with tarfile.open(tar_path, 'r') as tar:
    #         tar.extractall(path=destination_dir)
    # except Exception as e:
    #     print(f"Error extracting {tar_filename}: {e}")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=destination_dir)


def extract_signals(base_dir):
    """
    Iterate through all folders in sources.tar and extract the signals present in each folder.
    :param base_dir: base folder containing the all the sources' folders
    """
    n_errors = 0
    for folder in tqdm(os.listdir(base_dir), desc="Extracting files"):
        # Access each folder in sources
        folder_path = os.path.join(base_dir, folder)

        # Add path to the  .tar file
        folder_path = os.path.join(folder_path, 'analysis/spectra/7MTM2TM1')
        try:
            extract_tar(folder_path, f'Data/Raw/sources_extracted/{folder}')
        except Exception as e:
            n_errors += 1


if __name__ == '__main__':
    # Define the base directory
    base_directory = 'Data/sources'
    extract_signals(base_directory)
    n = 0

    for folder in tqdm(os.listdir(base_directory), desc="Counting files"):
        # Access each folder in sources
        folder_path = os.path.join(base_directory, folder)

        # Add path to the  .tar file
        folder_path = os.path.join(folder_path, 'analysis/spectra/7MTM2TM1')
        if not os.path.isfile("core-spectra.tar"):
            n += 1

    print(f'Folders without files: {n}/{len(os.listdir(base_directory))}')

