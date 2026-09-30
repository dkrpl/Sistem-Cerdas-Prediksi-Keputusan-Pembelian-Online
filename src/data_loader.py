"""
Data Loader module with robust multi-source acquisition for Google Colab and Local environment.
"""

import os
import sys
import glob
from pathlib import Path
import pandas as pd
from src.config import GDRIVE_FOLDER_ID, GDRIVE_FOLDER_URL, DATASET_FILENAME

def load_dataset(data_path: str = None) -> pd.DataFrame:
    """
    Loads online_shoppers_intention.csv using a prioritized fallback strategy:
    1. Specified or default local file path.
    2. Google Drive mounted location (/content/drive/MyDrive/...).
    3. gdown download from Google Drive folder.
    4. UCI Machine Learning Repository fallback.
    """
    # 1. Check local file
    candidate_paths = [
        data_path,
        DATASET_FILENAME,
        f"data/raw/{DATASET_FILENAME}",
        f"../{DATASET_FILENAME}",
        f"/content/{DATASET_FILENAME}",
        f"/content/dataset/{DATASET_FILENAME}",
    ]
    
    for path in candidate_paths:
        if path and os.path.exists(path):
            print(f"[DataLoader] Found dataset locally at: {path}")
            df = pd.read_csv(path)
            return clean_raw_data(df)
            
    # 2. Check mounted Google Drive in Colab
    colab_drive_root = "/content/drive/MyDrive"
    if os.path.exists(colab_drive_root):
        print(f"[DataLoader] Google Drive detected. Searching for '{DATASET_FILENAME}'...")
        matches = glob.glob(f"{colab_drive_root}/**/{DATASET_FILENAME}", recursive=True)
        if matches:
            print(f"[DataLoader] Found in Google Drive at: {matches[0]}")
            df = pd.read_csv(matches[0])
            return clean_raw_data(df)

    # 3. Attempt gdown from user's Google Drive folder
    try:
        import gdown
        print(f"[DataLoader] Attempting download from Google Drive folder ID: {GDRIVE_FOLDER_ID}...")
        out_dir = "/content/dataset" if "google.colab" in sys.modules else "./downloaded_dataset"
        os.makedirs(out_dir, exist_ok=True)
        gdown.download_folder(GDRIVE_FOLDER_URL, output=out_dir, quiet=False, use_cookies=False)
        matches = glob.glob(f"{out_dir}/**/{DATASET_FILENAME}", recursive=True)
        if matches:
            print(f"[DataLoader] Successfully downloaded via gdown: {matches[0]}")
            df = pd.read_csv(matches[0])
            return clean_raw_data(df)
    except Exception as e:
        print(f"[DataLoader] gdown folder download notice: {e}")

    # 4. Fallback to official UCI Machine Learning Repository
    print("[DataLoader] Attempting fallback download from UCI Repository...")
    uci_url = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
    try:
        import urllib.request
        import zipfile
        zip_path = "online_shoppers.zip"
        urllib.request.urlretrieve(uci_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        if os.path.exists(DATASET_FILENAME):
            print("[DataLoader] UCI fallback download successful.")
            df = pd.read_csv(DATASET_FILENAME)
            return clean_raw_data(df)
    except Exception as e:
        print(f"[DataLoader] UCI fallback failed: {e}")

    raise FileNotFoundError(
        f"Could not locate '{DATASET_FILENAME}'. Please ensure the file is present "
        f"in your Google Drive folder ({GDRIVE_FOLDER_URL}) or current directory."
    )

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset by removing exact duplicate records as mandated by data quality standards.
    """
    init_shape = df.shape
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = init_shape[0] - df_cleaned.shape[0]
    print(f"[DataCleaning] Raw rows: {init_shape[0]}, Removed duplicates: {duplicates_removed}, Clean rows: {df_cleaned.shape[0]}")
    return df_cleaned
