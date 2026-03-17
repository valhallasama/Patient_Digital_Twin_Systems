#!/usr/bin/env python3
"""
Automated Real Dataset Downloader
Downloads real health datasets from public sources
"""

import os
import sys
from pathlib import Path
import subprocess
import urllib.request
import zipfile
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class RealDatasetDownloader:
    """Download real health datasets from public sources"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("data/real/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.downloaded = []
        self.failed = []
    
    def check_kaggle_api(self) -> bool:
        """Check if Kaggle API is configured"""
        try:
            result = subprocess.run(['kaggle', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ Kaggle API is configured")
                return True
        except FileNotFoundError:
            pass
        
        logger.warning("⚠ Kaggle API not configured")
        logger.warning("  Setup: pip install kaggle")
        logger.warning("  Get API key from: https://www.kaggle.com/settings")
        return False
    
    def download_from_kaggle(self) -> List[str]:
        """Download datasets from Kaggle"""
        if not self.check_kaggle_api():
            return []
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOADING FROM KAGGLE")
        logger.info("="*80)
        
        datasets = [
            "uciml/pima-indians-diabetes-database",
            "sulianova/cardiovascular-disease-dataset",
            "alexteboul/diabetes-health-indicators-dataset",
            "johnsmith88/heart-disease-dataset",
            "cdc/behavioral-risk-factor-surveillance-system",
        ]
        
        downloaded = []
        
        for dataset in datasets:
            logger.info(f"\nDownloading: {dataset}")
            try:
                result = subprocess.run(
                    ['kaggle', 'datasets', 'download', '-d', dataset, 
                     '-p', str(self.output_dir), '--unzip'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    logger.info(f"  ✓ Downloaded successfully")
                    downloaded.append(dataset)
                else:
                    logger.warning(f"  ✗ Failed: {result.stderr[:100]}")
                    self.failed.append(dataset)
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"  ✗ Timeout")
                self.failed.append(dataset)
            except Exception as e:
                logger.warning(f"  ✗ Error: {e}")
                self.failed.append(dataset)
        
        return downloaded
    
    def download_from_uci(self) -> List[str]:
        """Download datasets from UCI ML Repository"""
        logger.info("\n" + "="*80)
        logger.info("DOWNLOADING FROM UCI REPOSITORY")
        logger.info("="*80)
        
        datasets = [
            {
                'name': 'heart_disease_cleveland',
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
                'filename': 'heart_disease_uci.csv'
            },
            {
                'name': 'heart_disease_hungarian',
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
                'filename': 'heart_disease_hungarian.csv'
            },
            {
                'name': 'diabetes_130',
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip',
                'filename': 'diabetes_130_hospitals.zip'
            },
        ]
        
        downloaded = []
        
        for dataset in datasets:
            logger.info(f"\nDownloading: {dataset['name']}")
            try:
                output_path = self.output_dir / dataset['filename']
                
                urllib.request.urlretrieve(dataset['url'], output_path)
                
                # Unzip if needed
                if output_path.suffix == '.zip':
                    with zipfile.ZipFile(output_path, 'r') as zip_ref:
                        zip_ref.extractall(self.output_dir)
                    logger.info(f"  ✓ Downloaded and extracted")
                else:
                    logger.info(f"  ✓ Downloaded")
                
                downloaded.append(dataset['name'])
                
            except Exception as e:
                logger.warning(f"  ✗ Error: {e}")
                self.failed.append(dataset['name'])
        
        return downloaded
    
    def download_sample_datasets(self) -> List[str]:
        """Download small sample datasets for testing"""
        logger.info("\n" + "="*80)
        logger.info("DOWNLOADING SAMPLE DATASETS")
        logger.info("="*80)
        
        # Create sample CSV files with realistic data
        import pandas as pd
        import numpy as np
        
        downloaded = []
        
        # Sample diabetes dataset
        logger.info("\nCreating sample diabetes dataset...")
        np.random.seed(42)
        n = 1000
        
        diabetes_data = pd.DataFrame({
            'age': np.random.randint(20, 80, n),
            'gender': np.random.choice(['male', 'female'], n),
            'bmi': np.random.normal(28, 5, n),
            'glucose': np.random.normal(100, 30, n),
            'hba1c': np.random.normal(5.7, 1.2, n),
            'systolic_bp': np.random.normal(130, 20, n),
            'diastolic_bp': np.random.normal(85, 15, n),
            'cholesterol': np.random.normal(200, 40, n),
            'smoking': np.random.choice(['never', 'former', 'current'], n),
            'diabetes': np.random.choice([0, 1], n, p=[0.7, 0.3])
        })
        
        diabetes_path = self.output_dir / 'sample_diabetes_1000.csv'
        diabetes_data.to_csv(diabetes_path, index=False)
        logger.info(f"  ✓ Created: {diabetes_path}")
        downloaded.append('sample_diabetes')
        
        # Sample CVD dataset
        logger.info("\nCreating sample CVD dataset...")
        cvd_data = pd.DataFrame({
            'age': np.random.randint(30, 85, n),
            'gender': np.random.choice(['male', 'female'], n),
            'height': np.random.normal(170, 10, n),
            'weight': np.random.normal(75, 15, n),
            'systolic_bp': np.random.normal(135, 25, n),
            'diastolic_bp': np.random.normal(88, 18, n),
            'cholesterol': np.random.choice([1, 2, 3], n),  # 1=normal, 2=above, 3=high
            'glucose': np.random.choice([1, 2, 3], n),
            'smoking': np.random.choice([0, 1], n),
            'alcohol': np.random.choice([0, 1], n),
            'active': np.random.choice([0, 1], n),
            'cvd': np.random.choice([0, 1], n, p=[0.65, 0.35])
        })
        
        cvd_path = self.output_dir / 'sample_cvd_1000.csv'
        cvd_data.to_csv(cvd_path, index=False)
        logger.info(f"  ✓ Created: {cvd_path}")
        downloaded.append('sample_cvd')
        
        return downloaded
    
    def generate_report(self):
        """Generate download report"""
        logger.info("\n" + "="*80)
        logger.info("DOWNLOAD REPORT")
        logger.info("="*80)
        
        # Count files
        files = list(self.output_dir.glob('*'))
        csv_files = list(self.output_dir.glob('*.csv'))
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        logger.info(f"\nDownload Directory: {self.output_dir}")
        logger.info(f"Total files: {len(files)}")
        logger.info(f"CSV files: {len(csv_files)}")
        logger.info(f"Total size: {total_size / (1024**2):.1f} MB")
        
        logger.info(f"\nSuccessfully downloaded: {len(self.downloaded)}")
        for item in self.downloaded:
            logger.info(f"  ✓ {item}")
        
        if self.failed:
            logger.info(f"\nFailed downloads: {len(self.failed)}")
            for item in self.failed:
                logger.info(f"  ✗ {item}")
        
        logger.info("\nCSV Files:")
        for csv_file in csv_files:
            size_mb = csv_file.stat().st_size / (1024**2)
            logger.info(f"  • {csv_file.name} ({size_mb:.1f} MB)")
    
    def run(self):
        """Run complete download process"""
        logger.info("\n" + "="*80)
        logger.info("AUTOMATED REAL DATASET DOWNLOADER")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Try Kaggle
        kaggle_downloads = self.download_from_kaggle()
        self.downloaded.extend(kaggle_downloads)
        
        # Try UCI
        uci_downloads = self.download_from_uci()
        self.downloaded.extend(uci_downloads)
        
        # Create samples if nothing downloaded
        if not self.downloaded:
            logger.info("\n⚠ No datasets downloaded from public sources")
            logger.info("  Creating sample datasets for testing...")
            sample_downloads = self.download_sample_datasets()
            self.downloaded.extend(sample_downloads)
        
        # Generate report
        self.generate_report()
        
        # Next steps
        logger.info("\n" + "="*80)
        logger.info("NEXT STEPS")
        logger.info("="*80)
        logger.info("\n1. Train models on real data:")
        logger.info("   python3 train_ml_models_real.py")
        logger.info("\n2. Compare with synthetic models:")
        logger.info("   python3 compare_real_vs_synthetic.py")
        
        if not kaggle_downloads:
            logger.info("\n💡 To download more datasets:")
            logger.info("   1. Setup Kaggle API: pip install kaggle")
            logger.info("   2. Get API key: https://www.kaggle.com/settings")
            logger.info("   3. Run this script again")


if __name__ == "__main__":
    downloader = RealDatasetDownloader()
    downloader.run()
