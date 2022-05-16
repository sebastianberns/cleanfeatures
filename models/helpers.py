import logging
from pathlib import Path
import shutil
import urllib.request

import requests


"""
Download any file from a url if it does not exist

    local_dir (Path/str): output folder path
    url (str): the web url to download

Returns the path to the downloaded file
"""
def check_download_url(local_dir, url):
    # Make sure paths are absolute Path objects
    local_dir = Path(local_dir).expanduser().resolve()
    name = Path(url).name  # Get filename
    file_path = local_dir/name  # Build path to file

    if file_path.exists():
        logging.info(f"Found {name} in {local_dir}")
    else:
        if not local_dir.exists():  # Create folder if it does not exist
            logging.info(f"Creating directory {local_dir}")
            local_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading {name} to {local_dir} ...")
        with urllib.request.urlopen(url) as response, open(file_path, 'wb') as f:
            shutil.copyfileobj(response, f)
    return file_path
