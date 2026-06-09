import os
import urllib.request
import zipfile
import subprocess
from scipy.io import loadmat 
import os
import zipfile
import requests
import scipy.io as sio


DATASETS = {
    "UP": {
        "datakey": "Utopia",
        "labelkey": "Utopia_gt",
        "files": [
            ("Utopia.mat", "https://download.scidb.cn/download?fileId=7f0b615fbf9cb869a9b11cfa76841887&path=/V2/Utopia.mat&fileName=Utopia.mat"),
            ("Utopia_gt.mat", "https://download.scidb.cn/download?fileId=74ad946c72ccb701903225ce7311f492&path=/V2/Utopia_gt.mat&fileName=Utopia_gt.mat"),
        ],
    },
    "NF": {
        "datakey": "NiliFossae",
        "labelkey": "NiliFossae_gt",
        "files": [
            ("NiliFossae.mat", "https://download.scidb.cn/download?fileId=ae13bf8fe050b5114fef0f1c03934fe4&path=/V2/NiliFossae.mat&fileName=NiliFossae.mat"),
            ("NiliFossae_gt.mat", "https://download.scidb.cn/download?fileId=2d45d15af1232726c00fea6108fa2eee&path=/V2/NiliFossae_gt.mat&fileName=NiliFossae_gt.mat"),
        ],
    },
    "HC": {
        "datakey": "holden",
        "labelkey": "holden_gt",
        "files": [
            ("Holden.mat", "https://download.scidb.cn/download?fileId=2cba3da0e9e8fd14705f67317bacde15&path=/V2/holden.mat&fileName=holden.mat"),
            ("Holden_gt.mat", "https://download.scidb.cn/download?fileId=a77d0577e822760e90614dc529af07c7&path=/V2/holden_gt.mat&fileName=holden_gt.mat"),
        ],
    },
    "Pavia": {
        "datakey": "paviaU",
        "labelkey": "paviaU_gt",
        "files": [
            ("PaviaU.mat", "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"),
            ("PaviaU_gt.mat", "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat")
        ],
    },
    'Indian':{
        "datakey": "indian_pines", #key of data in .mat file
        "labelkey": "indian_pines_gt", #key of label in .mat file
        "files" : [
            ( "indian_pines_corrected.mat", "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"),
            ( "indian_pines_gt.mat", "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"),
        ],
    },
    # For houston use Houston
}

#
#
# this function will download the dataset if not already downloaded, and load the data and label from the .mat files
# Supported datasets are UP, NF, HC, Pavia, Indian. For Houston dataset

def downloadAndLoadDataset(dataset_name, save_folder='dataset'):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found")

    dataset = DATASETS[dataset_name]

    # load pavia type dataset from kaggle
    if dataset_name == 'Houston':
        return load_houston()

    save_dir = os.path.join(os.getcwd(), save_folder)
    os.makedirs(save_dir, exist_ok=True)

    def download(url, path):
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            urllib.request.urlretrieve(url, path)
            print(f"Done ({os.path.getsize(path)/(1024*1024):.2f} MB)")
        else:
            print(f"{path} already exists, skipping.")

    # Download files
    paths = []
    for filename, url in dataset["files"]:
        filepath = os.path.join(save_dir, filename)
        download(url, filepath)
        paths.append(filepath)

    print(f"\n{dataset_name} download completed.")

    # Load dataset
    data_path, label_path = paths
    return _loadDataset(
        data_path,
        label_path,
        dataset["datakey"],
        dataset["labelkey"]
    )

def load_houston(download_dir="./dataset"):
    os.makedirs(download_dir, exist_ok=True)

    zip_path = os.path.join(download_dir, "houston.zip")

    # Download
    if not os.path.exists(zip_path):
        print("⬇ Downloading Houston dataset...")
        response = requests.get(
            "https://www.kaggle.com/api/v1/datasets/download/mingliu123/houston",
            stream=True
        )
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Downloaded")
    else:
        print("✅ Already downloaded")

    # Unzip
    if not os.path.exists(os.path.join(download_dir, "data.mat")):
        print("📦 Unzipping...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(download_dir)
        print("✅ Unzipped")
    else:
        print("✅ Already unzipped")

    # Load
    data   = sio.loadmat(os.path.join(download_dir, "data.mat"))["data"]
    labels = sio.loadmat(os.path.join(download_dir, "mask_train.mat"))["mask_train"]

    print(f"   Data   shape : {data.shape}   dtype: {data.dtype}")
    print(f"   Labels shape : {labels.shape}  dtype: {labels.dtype}")

    return data, labels


def _loadDataset(data_path, label_path, datakey, labelkey):
    data = loadmat(data_path)[datakey]
    label = loadmat(label_path)[labelkey]

    print(f"Data shape: {data.shape}")
    print(f"Label shape: {label.shape}")
    
    return data, label