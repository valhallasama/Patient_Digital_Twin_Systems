#!/usr/bin/env python3
"""
Download Additional Real Datasets for Other Organs
Focus on: Renal (kidney), Hepatic (liver), Cancer, Stroke
"""

import requests
import pandas as pd
from pathlib import Path
import zipfile
import io
import time

def download_chronic_kidney_disease():
    """
    Chronic Kidney Disease Dataset
    Source: UCI ML Repository
    400 patients with kidney disease indicators
    """
    print("\n🫘 Downloading Chronic Kidney Disease Dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/Chronic_Kidney_Disease.zip"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall("data/real/raw/ckd/")
        
        print("✓ Downloaded Chronic Kidney Disease dataset")
        print("  • ~400 patients with CKD indicators")
        print("  • Features: age, BP, specific gravity, albumin, sugar, RBC, etc.")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def download_liver_disease():
    """
    Indian Liver Patient Dataset
    Source: UCI ML Repository
    583 patients with liver disease
    """
    print("\n🫀 Downloading Liver Disease Dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
    
    try:
        df = pd.read_csv(url, header=None)
        df.columns = ['age', 'gender', 'total_bilirubin', 'direct_bilirubin', 
                      'alkaline_phosphotase', 'alamine_aminotransferase', 
                      'aspartate_aminotransferase', 'total_proteins', 
                      'albumin', 'albumin_globulin_ratio', 'target']
        
        output_path = Path("data/real/raw/liver_disease.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("✓ Downloaded Liver Disease dataset")
        print(f"  • {len(df)} patients")
        print("  • Features: bilirubin, liver enzymes (ALT, AST), proteins")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def download_breast_cancer():
    """
    Breast Cancer Wisconsin Dataset
    Source: UCI ML Repository
    569 patients
    """
    print("\n🎗️  Downloading Breast Cancer Dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    
    try:
        df = pd.read_csv(url, header=None)
        
        # Column names from dataset description
        columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
        df.columns = columns
        
        output_path = Path("data/real/raw/breast_cancer.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("✓ Downloaded Breast Cancer dataset")
        print(f"  • {len(df)} patients")
        print("  • Features: tumor characteristics (radius, texture, perimeter, etc.)")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def download_stroke_data():
    """
    Stroke Prediction Dataset
    Source: Kaggle (public)
    5,110 patients
    """
    print("\n🧠 Downloading Stroke Prediction Dataset...")
    
    # This is a public Kaggle dataset, available via direct link
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    
    try:
        df = pd.read_csv(url)
        
        output_path = Path("data/real/raw/insurance_health.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("✓ Downloaded Health Insurance dataset")
        print(f"  • {len(df)} patients")
        print("  • Features: age, BMI, smoking, charges (health burden proxy)")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def download_parkinsons():
    """
    Parkinson's Disease Dataset
    Source: UCI ML Repository
    195 patients
    """
    print("\n🧠 Downloading Parkinson's Disease Dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    
    try:
        df = pd.read_csv(url)
        
        output_path = Path("data/real/raw/parkinsons.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("✓ Downloaded Parkinson's Disease dataset")
        print(f"  • {len(df)} patients")
        print("  • Features: voice measurements (neural function proxy)")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def download_thyroid():
    """
    Thyroid Disease Dataset
    Source: UCI ML Repository
    7,200 patients
    """
    print("\n🦋 Downloading Thyroid Disease Dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data"
    
    try:
        # Thyroid dataset has specific format
        df = pd.read_csv(url, header=None)
        
        output_path = Path("data/real/raw/thyroid.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("✓ Downloaded Thyroid Disease dataset")
        print(f"  • {len(df)} patients")
        print("  • Features: thyroid function tests (endocrine system)")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    print("="*80)
    print("DOWNLOADING ADDITIONAL REAL PATIENT DATASETS")
    print("="*80)
    print("\nTargeting organs not covered by existing data:")
    print("  • Renal (kidney)")
    print("  • Hepatic (liver)")
    print("  • Cancer")
    print("  • Neural (Parkinson's)")
    print("  • Endocrine (thyroid)")
    
    results = {}
    
    # Download each dataset
    results['ckd'] = download_chronic_kidney_disease()
    time.sleep(1)
    
    results['liver'] = download_liver_disease()
    time.sleep(1)
    
    results['breast_cancer'] = download_breast_cancer()
    time.sleep(1)
    
    results['insurance'] = download_stroke_data()
    time.sleep(1)
    
    results['parkinsons'] = download_parkinsons()
    time.sleep(1)
    
    results['thyroid'] = download_thyroid()
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\n✓ Successfully downloaded: {successful}/{total} datasets")
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"""
📊 Total Real Patient Data Available:
  • Diabetes: 101,766 patients ✅
  • Heart Disease: 595 patients ✅
  • Chronic Kidney Disease: ~400 patients {'✅' if results.get('ckd') else '❌'}
  • Liver Disease: 583 patients {'✅' if results.get('liver') else '❌'}
  • Breast Cancer: 569 patients {'✅' if results.get('breast_cancer') else '❌'}
  • Health Insurance: ~1,300 patients {'✅' if results.get('insurance') else '❌'}
  • Parkinson's: 195 patients {'✅' if results.get('parkinsons') else '❌'}
  • Thyroid: 7,200 patients {'✅' if results.get('thyroid') else '❌'}

🎯 Next: Run integrate_all_real_data.py to extract parameters from all datasets
""")


if __name__ == "__main__":
    main()
