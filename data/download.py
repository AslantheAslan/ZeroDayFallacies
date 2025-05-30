from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import os

from tqdm import tqdm


def download_CIDDS() -> None:
    url = 'https://www.hs-coburg.de/wp-content/uploads/2024/11/CIDDS-001.zip'
    # Define the target folder and ZIP file name
    target_folder = r'./data/CIDDS'  # Windows

    # Check if the dataset is already downloaded
    if os.path.exists(target_folder) and any(os.scandir(target_folder)):
        print(f"Dataset already exists in {target_folder}. Skipping download.")
    else:
        os.makedirs(target_folder, exist_ok=True)  # Ensure folder exists

        # Download the ZIP file with progress bar
        print("Downloading dataset...")
        with urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB per read
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            zip_data = BytesIO()
            while chunk := response.read(block_size):
                zip_data.write(chunk)
                progress_bar.update(len(chunk))

            progress_bar.close()

        # Extract ZIP file
        print("\nExtracting dataset...")
        with ZipFile(zip_data) as zfile:
            zfile.extractall(target_folder)

        print(f"Files extracted to: {target_folder}")

    print("Process completed!")
