#!/usr/bin/env python3
"""
Comprehensive Medical Dataset Downloader
Downloads ALL relevant medical datasets from:
1. Kaggle
2. UCI ML Repository
3. Data.gov
4. OpenML
5. PhysioNet (public datasets)
6. WHO Data
7. CDC Data
"""

import os
import requests
import pandas as pd
from pathlib import Path
import json
import time
from bs4 import BeautifulSoup
import zipfile
import io
import subprocess

# Kaggle datasets to download
KAGGLE_DATASETS = [
    # Diabetes
    "uciml/pima-indians-diabetes-database",
    "johndasilva/diabetes",
    "mathchi/diabetes-data-set",
    
    # Heart Disease
    "ronitf/heart-disease-uci",
    "johnsmith88/heart-disease-dataset",
    "rashikrahmanpritom/heart-attack-analysis-prediction-dataset",
    
    # Kidney Disease
    "mansoordaku/ckdisease",
    "akshayksingh/kidney-disease-dataset",
    
    # Liver Disease
    "uciml/indian-liver-patient-records",
    
    # Cancer
    "uciml/breast-cancer-wisconsin-data",
    "yasserh/breast-cancer-dataset",
    "erdemtaha/cancer-data",
    
    # Stroke
    "fedesoriano/stroke-prediction-dataset",
    "jillanisofttech/brain-stroke-dataset",
    
    # General Health
    "mirichoi0218/insurance",
    "cdc/behavioral-risk-factor-surveillance-system",
    "cdc/national-health-and-nutrition-examination-survey",
    
    # Mental Health
    "osmi/mental-health-in-tech-survey",
    
    # COVID-19
    "imdevskp/corona-virus-report",
    
    # Obesity
    "ankanhore545/obesity-levels",
    
    # Sleep
    "uom190346a/sleep-health-and-lifestyle-dataset",
    
    # Alzheimer's
    "jboysen/mri-and-alzheimers",
    
    # Parkinson's
    "vikasukani/parkinsons-disease-data-set",
    
    # Thyroid
    "emmanuelfwerr/thyroid-disease-data",
]

# UCI datasets (direct download links)
UCI_DATASETS = {
    "chronic_kidney_disease": "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/Chronic_Kidney_Disease.zip",
    "hepatitis": "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data",
    "thyroid": "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data",
    "parkinsons": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
    "heart_disease_cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "heart_disease_hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "heart_disease_switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "breast_cancer": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    "lung_cancer": "https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data",
}

# OpenML dataset IDs
OPENML_DATASETS = [
    37,    # diabetes
    1464,  # blood-transfusion
    1480,  # indian_liver_patient
    1590,  # adult (health insurance)
]


def setup_kaggle():
    """Setup Kaggle API credentials"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("⚠️  Kaggle credentials not found")
        print("   Please set up Kaggle API:")
        print("   1. Go to https://www.kaggle.com/settings")
        print("   2. Create new API token")
        print("   3. Place kaggle.json in ~/.kaggle/")
        return False
    
    # Set permissions
    os.chmod(kaggle_json, 0o600)
    return True


def download_kaggle_dataset(dataset_name: str, output_dir: Path):
    """Download a Kaggle dataset"""
    try:
        print(f"\n📥 Downloading from Kaggle: {dataset_name}")
        
        # Create output directory
        dataset_dir = output_dir / dataset_name.replace("/", "_")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download using kaggle CLI
        cmd = f"kaggle datasets download -d {dataset_name} -p {dataset_dir} --unzip"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Downloaded: {dataset_name}")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        return False


def download_uci_dataset(name: str, url: str, output_dir: Path):
    """Download UCI dataset"""
    try:
        print(f"\n📥 Downloading from UCI: {name}")
        
        # Create output directory first
        output_dir.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine file type
        if url.endswith('.zip'):
            # Extract zip
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                extract_dir = output_dir / f"uci_{name}"
                extract_dir.mkdir(parents=True, exist_ok=True)
                z.extractall(extract_dir)
        else:
            # Save as CSV
            output_file = output_dir / f"uci_{name}.csv"
            with open(output_file, 'wb') as f:
                f.write(response.content)
        
        print(f"✓ Downloaded: {name}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")
        return False


def download_openml_dataset(dataset_id: int, output_dir: Path):
    """Download OpenML dataset"""
    try:
        print(f"\n📥 Downloading from OpenML: Dataset {dataset_id}")
        
        # Create output directory first
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from sklearn.datasets import fetch_openml
        
        data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        df = data.frame
        
        output_file = output_dir / f"openml_{dataset_id}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✓ Downloaded: OpenML {dataset_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading OpenML {dataset_id}: {e}")
        return False


def search_data_gov():
    """Search Data.gov for health datasets"""
    print("\n🔍 Searching Data.gov for health datasets...")
    
    datasets = []
    
    try:
        # Data.gov API
        api_url = "https://catalog.data.gov/api/3/action/package_search"
        
        queries = [
            "health", "diabetes", "heart disease", "cancer",
            "kidney disease", "liver disease", "stroke",
            "mental health", "obesity", "sleep"
        ]
        
        for query in queries:
            params = {
                'q': query,
                'rows': 10,
                'fq': 'res_format:CSV'
            }
            
            response = requests.get(api_url, params=params, timeout=30)
            if response.status_code == 200:
                results = response.json()
                if 'result' in results and 'results' in results['result']:
                    for dataset in results['result']['results']:
                        datasets.append({
                            'name': dataset.get('title', 'Unknown'),
                            'url': dataset.get('url', ''),
                            'source': 'Data.gov'
                        })
            
            time.sleep(1)  # Rate limiting
        
        print(f"✓ Found {len(datasets)} datasets on Data.gov")
        return datasets
        
    except Exception as e:
        print(f"❌ Error searching Data.gov: {e}")
        return []


def search_physionet():
    """List public PhysioNet datasets"""
    print("\n🔍 PhysioNet Public Datasets...")
    
    datasets = [
        {
            'name': 'MIMIC-III Demo',
            'url': 'https://physionet.org/content/mimiciii-demo/1.4/',
            'description': '100 ICU patients (no credentials needed)'
        },
        {
            'name': 'PTB Diagnostic ECG Database',
            'url': 'https://physionet.org/content/ptbdb/1.0.0/',
            'description': 'ECG recordings'
        },
        {
            'name': 'MIT-BIH Arrhythmia Database',
            'url': 'https://physionet.org/content/mitdb/1.0.0/',
            'description': 'Cardiac arrhythmia data'
        }
    ]
    
    print(f"✓ Found {len(datasets)} public PhysioNet datasets")
    for ds in datasets:
        print(f"  • {ds['name']}: {ds['description']}")
    
    return datasets


def download_mimic_demo(output_dir: Path):
    """Download MIMIC-III Demo (no credentials needed)"""
    print("\n📥 Downloading MIMIC-III Demo (100 patients)...")
    
    try:
        base_url = "https://physionet.org/files/mimiciii-demo/1.4/"
        
        files = [
            "ADMISSIONS.csv.gz",
            "PATIENTS.csv.gz",
            "LABEVENTS.csv.gz",
            "CHARTEVENTS.csv.gz",
            "DIAGNOSES_ICD.csv.gz"
        ]
        
        mimic_dir = output_dir / "mimic_demo"
        mimic_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            url = base_url + file
            print(f"  Downloading {file}...")
            
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                output_file = mimic_dir / file
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ {file}")
            else:
                print(f"  ❌ Failed: {file}")
            
            time.sleep(1)
        
        print("✓ MIMIC-III Demo downloaded")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading MIMIC Demo: {e}")
        return False


def create_dataset_catalog(output_dir: Path, downloaded: dict):
    """Create catalog of all downloaded datasets"""
    catalog = {
        'total_datasets': sum(len(v) for v in downloaded.values()),
        'sources': {
            'kaggle': len(downloaded.get('kaggle', [])),
            'uci': len(downloaded.get('uci', [])),
            'openml': len(downloaded.get('openml', [])),
            'data_gov': len(downloaded.get('data_gov', [])),
            'physionet': len(downloaded.get('physionet', [])),
        },
        'datasets': downloaded
    }
    
    catalog_file = output_dir / "dataset_catalog.json"
    with open(catalog_file, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"\n✓ Catalog saved to {catalog_file}")
    return catalog


def main():
    print("=" * 80)
    print("COMPREHENSIVE MEDICAL DATASET DOWNLOADER")
    print("=" * 80)
    
    output_dir = Path("data/downloaded_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = {
        'kaggle': [],
        'uci': [],
        'openml': [],
        'data_gov': [],
        'physionet': []
    }
    
    # 1. Kaggle datasets
    print("\n" + "=" * 80)
    print("KAGGLE DATASETS")
    print("=" * 80)
    
    if setup_kaggle():
        for dataset in KAGGLE_DATASETS:
            if download_kaggle_dataset(dataset, output_dir / "kaggle"):
                downloaded['kaggle'].append(dataset)
            time.sleep(2)  # Rate limiting
    else:
        print("⚠️  Skipping Kaggle (credentials not set up)")
    
    # 2. UCI datasets
    print("\n" + "=" * 80)
    print("UCI ML REPOSITORY DATASETS")
    print("=" * 80)
    
    for name, url in UCI_DATASETS.items():
        if download_uci_dataset(name, url, output_dir / "uci"):
            downloaded['uci'].append(name)
        time.sleep(1)
    
    # 3. OpenML datasets
    print("\n" + "=" * 80)
    print("OPENML DATASETS")
    print("=" * 80)
    
    try:
        for dataset_id in OPENML_DATASETS:
            if download_openml_dataset(dataset_id, output_dir / "openml"):
                downloaded['openml'].append(dataset_id)
            time.sleep(1)
    except ImportError:
        print("⚠️  scikit-learn not available for OpenML")
    
    # 4. Data.gov search
    print("\n" + "=" * 80)
    print("DATA.GOV SEARCH")
    print("=" * 80)
    
    data_gov_results = search_data_gov()
    downloaded['data_gov'] = data_gov_results
    
    # 5. PhysioNet
    print("\n" + "=" * 80)
    print("PHYSIONET DATASETS")
    print("=" * 80)
    
    physionet_results = search_physionet()
    downloaded['physionet'] = physionet_results
    
    # Download MIMIC Demo
    if download_mimic_demo(output_dir / "physionet"):
        downloaded['physionet'].append({'name': 'MIMIC-III Demo', 'downloaded': True})
    
    # Create catalog
    catalog = create_dataset_catalog(output_dir, downloaded)
    
    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    
    print(f"""
📊 Total Datasets: {catalog['total_datasets']}

By Source:
  • Kaggle: {catalog['sources']['kaggle']} datasets
  • UCI ML Repository: {catalog['sources']['uci']} datasets
  • OpenML: {catalog['sources']['openml']} datasets
  • Data.gov: {catalog['sources']['data_gov']} results
  • PhysioNet: {catalog['sources']['physionet']} datasets

📁 Location: {output_dir}

🎯 Next Steps:
  1. Review downloaded datasets
  2. Extract parameters from each
  3. Integrate into digital twin
  4. Run: python integrate_all_downloaded_data.py
""")


if __name__ == "__main__":
    main()
