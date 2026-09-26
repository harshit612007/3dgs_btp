import urllib.request
import zipfile
import os

url = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
zip_path = "tandt_db.zip"
extract_dir = "tandt"

print(f"Downloading {url}...")
# Add headers to avoid 403 Forbidden
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
    data = response.read()
    out_file.write(data)

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Done! The dataset is located in the 'tandt' folder.")
