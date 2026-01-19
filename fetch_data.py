import os
import requests
import zipfile
import io

def download_wpl_data():
    # URL for Women's Premier League match-by-match CSV data
    url = "https://cricsheet.org/downloads/wpl_csv2.zip"
    target_dir = "data"
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    print(f"Downloading WPL data from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        print("Download successful. Extracting...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(target_dir)
        print(f"Data extracted to {target_dir}")
    else:
        print(f"Failed to download data. Status code: {response.status_code}")

if __name__ == "__main__":
    download_wpl_data()
