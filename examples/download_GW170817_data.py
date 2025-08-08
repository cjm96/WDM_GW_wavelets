import requests
import gzip
import shutil
import os

# File URL
url = "https://gwosc.org/eventapi/html/GWTC-1-confident/GW170817/v3/L-L1_GWOSC_4KHZ_R1-1187006835-4096.txt.gz"

# Local paths
gz_path = "../data/L-L1_GWOSC_4KHZ_R1-1187006835-4096.txt.gz"
txt_path = gz_path[:-3]  # Remove .gz

# Step 1: Download the file
print("Downloading...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(gz_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print(f"Downloaded: {gz_path}")

# Step 2: Unzip the file
print("Unzipping...")
with gzip.open(gz_path, 'rb') as f_in:
    with open(txt_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print(f"Unzipped to: {txt_path}")

# (Optional) Step 3: Delete the .gz file
os.remove(gz_path)
print(f"Deleted: {gz_path}")