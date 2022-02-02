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
    # Make sure paths are Path objects
    local_dir = Path(local_dir)
    name = Path(url).name  # Get filename
    file_path = local_dir/name  # Build path to file

    if not file_path.exists():
        if not local_dir.exists():  # Create folder if it does not exist
            local_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} to {local_dir}")
        with urllib.request.urlopen(url) as response, open(file_path, 'wb') as f:
            shutil.copyfileobj(response, f)
    return file_path


"""
Download a file from google drive

    file_id: id of the google drive file
    out_path (str): output folder path
"""
def download_google_drive(file_id, out_path):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
