import os
import urllib.request
import zipfile
import subprocess
from scipy.io import loadmat 

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
        "url": "https://www.kaggle.com/api/v1/datasets/download/syamkakarla/pavia-university-hsi",
        "type": "zip",
        "unzipped_file_names": ["PaviaU.mat", "PaviaU_gt.mat"] #after unzipping, the files we expect to find in the extracted folder
    },
    'Indian':{
        "datakey": "indian_pines", #key of data in .mat file
        "labelkey": "indian_pines_gt", #key of label in .mat file
        "url" : "https://www.kaggle.com/api/v1/datasets/download/emannasserabdelhafez/indian-pines",
        "type": "zip",
        "unzipped_file_names": ["indian_pines_corrected.mat", "indian_pines_gt.mat"] #after unzipping, the files we expect to find in the extracted folder
    },
    'Houston':{
        "datakey": 'data',
        "labelkey": 'mask_train',
        "url" : "https://www.kaggle.com/api/v1/datasets/download/mingliu123/houston",
        "type" : "zip",
        "unzipped_file_names": ["data.mat", "mask_train.mat"]
    }
}


def downloadAndLoadDataset(dataset_name, save_folder='dataset'):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found")

    dataset = DATASETS[dataset_name]

    # load pavia type dataset from kaggle
    if dataset.get('type') == 'zip':
        return _loadDataFromKaggle(dataset, save_folder)

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


# Todo 
def _loadDataFromKaggle(dataset, save_folder):
    # Handle Kaggle datasets (zip files)
    dataset_name = dataset.get("datakey", "dataset")
    
    # Create full save directory path
    save_dir = os.path.join(os.getcwd(), save_folder)
    os.makedirs(save_dir, exist_ok=True)
    
    zip_filename = f"{dataset_name}-kaggle.zip"
    zip_path = os.path.join(save_dir, zip_filename)
    
    # Download using curl
    if not os.path.exists(zip_path):
        print(f"Downloading {dataset_name} dataset using curl...")
        curl_cmd = ["curl", "-L", "-o", zip_path, dataset["url"]]
        subprocess.run(curl_cmd, check=True)
        print(f"Done ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")
    else:
        print(f"{dataset_name} zip file already exists, skipping download.")
    
    # Create extraction marker to track if already extracted
    extraction_marker = os.path.join(save_dir, f".{dataset_name}_extracted")
    
    # Extract zip file directly to save_dir (no subfolder)
    if not os.path.exists(extraction_marker):
        print(f"Extracting {dataset_name} dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        # Create marker file to indicate extraction is complete
        with open(extraction_marker, 'w') as f:
            f.write("extracted")
        print("Extraction completed.")
    else:
        print(f"{dataset_name} dataset already extracted, skipping.")
    
    # Find files using unzipped_file_names
    unzipped_file_names = dataset.get("unzipped_file_names", [])
    if not unzipped_file_names:
        raise ValueError(f"unzipped_file_names not specified for {dataset_name}")
    
    data_file = None
    label_file = None
    
    # Search recursively for the specified files in save_dir
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if file in unzipped_file_names:
                file_path = os.path.join(root, file)
                if 'gt' in file.lower():
                    label_file = file_path
                else:
                    data_file = file_path
    
    if not data_file or not label_file:
        print(f"ERROR: Could not find expected files in {dataset_name}:")
        print(f"  Looking for: {unzipped_file_names}")
        print(f"  Data file found: {data_file}")
        print(f"  Label file found: {label_file}")
        raise FileNotFoundError(f"Could not find expected files {unzipped_file_names} in extracted {dataset_name} dataset")
    
    print(f"\n{dataset_name} download and extraction completed.")
    return _loadDataset(data_file, label_file, dataset["datakey"], dataset["labelkey"])

def _loadDataset(data_path, label_path, datakey, labelkey):
    data = loadmat(data_path)[datakey]
    label = loadmat(label_path)[labelkey]

    print(f"Data shape: {data.shape}")
    print(f"Label shape: {label.shape}")
    
    return data, label