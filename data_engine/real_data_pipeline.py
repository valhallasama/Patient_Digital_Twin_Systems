"""
Real Data Acquisition Pipeline
Automatically searches, downloads, validates, and tracks real health datasets
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import hashlib

from data_engine.dataset_discovery import DatasetDiscoveryEngine
from data_engine.dataset_downloader import DatasetDownloader
from data_engine.dataset_validator import DatasetValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLineageTracker:
    """Track provenance and lineage of all downloaded datasets"""
    
    def __init__(self, lineage_path: Path = None):
        self.lineage_path = lineage_path or Path("data/real/lineage.json")
        self.lineage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing lineage
        self.lineage = self._load_lineage()
    
    def _load_lineage(self) -> Dict:
        """Load lineage from disk"""
        if self.lineage_path.exists():
            with open(self.lineage_path, 'r') as f:
                return json.load(f)
        return {
            'datasets': {},
            'sources': {},
            'download_history': []
        }
    
    def _save_lineage(self):
        """Save lineage to disk"""
        with open(self.lineage_path, 'w') as f:
            json.dump(self.lineage, f, indent=2, default=str)
    
    def register_dataset(self, dataset_info: Dict, file_path: Path):
        """Register a downloaded dataset"""
        
        # Calculate file hash for integrity
        file_hash = self._calculate_file_hash(file_path)
        
        dataset_id = f"{dataset_info['source']}_{dataset_info.get('doi', 'unknown')}".replace('/', '_')
        
        record = {
            'dataset_id': dataset_id,
            'source': dataset_info['source'],
            'title': dataset_info.get('title', 'Unknown'),
            'url': dataset_info.get('url', ''),
            'doi': dataset_info.get('doi', ''),
            'downloaded_at': datetime.now().isoformat(),
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'file_hash': file_hash,
            'metadata': dataset_info.get('metadata', {}),
            'validation_status': 'pending',
            'cleaned': False
        }
        
        self.lineage['datasets'][dataset_id] = record
        
        # Track source
        source = dataset_info['source']
        if source not in self.lineage['sources']:
            self.lineage['sources'][source] = {
                'total_datasets': 0,
                'total_size_bytes': 0,
                'first_download': datetime.now().isoformat()
            }
        
        self.lineage['sources'][source]['total_datasets'] += 1
        self.lineage['sources'][source]['total_size_bytes'] += record['file_size']
        self.lineage['sources'][source]['last_download'] = datetime.now().isoformat()
        
        # Add to download history
        self.lineage['download_history'].append({
            'dataset_id': dataset_id,
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'size': record['file_size']
        })
        
        self._save_lineage()
        
        logger.info(f"✓ Registered dataset: {dataset_id}")
        
        return dataset_id
    
    def update_validation_status(self, dataset_id: str, status: str, details: Dict):
        """Update validation status of dataset"""
        if dataset_id in self.lineage['datasets']:
            self.lineage['datasets'][dataset_id]['validation_status'] = status
            self.lineage['datasets'][dataset_id]['validation_details'] = details
            self.lineage['datasets'][dataset_id]['validated_at'] = datetime.now().isoformat()
            self._save_lineage()
    
    def mark_as_cleaned(self, dataset_id: str, cleaned_path: Path):
        """Mark dataset as cleaned"""
        if dataset_id in self.lineage['datasets']:
            self.lineage['datasets'][dataset_id]['cleaned'] = True
            self.lineage['datasets'][dataset_id]['cleaned_path'] = str(cleaned_path)
            self.lineage['datasets'][dataset_id]['cleaned_at'] = datetime.now().isoformat()
            self._save_lineage()
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """Get information about a dataset"""
        return self.lineage['datasets'].get(dataset_id)
    
    def get_all_datasets(self, source: Optional[str] = None, 
                        validated_only: bool = False) -> List[Dict]:
        """Get all datasets, optionally filtered"""
        datasets = list(self.lineage['datasets'].values())
        
        if source:
            datasets = [d for d in datasets if d['source'] == source]
        
        if validated_only:
            datasets = [d for d in datasets if d['validation_status'] == 'valid']
        
        return datasets
    
    def get_statistics(self) -> Dict:
        """Get statistics about downloaded datasets"""
        total_datasets = len(self.lineage['datasets'])
        total_size = sum(d['file_size'] for d in self.lineage['datasets'].values())
        
        validated = sum(1 for d in self.lineage['datasets'].values() 
                       if d['validation_status'] == 'valid')
        cleaned = sum(1 for d in self.lineage['datasets'].values() 
                     if d.get('cleaned', False))
        
        return {
            'total_datasets': total_datasets,
            'total_size_gb': total_size / (1024**3),
            'validated_datasets': validated,
            'cleaned_datasets': cleaned,
            'sources': len(self.lineage['sources']),
            'source_breakdown': self.lineage['sources']
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        if not file_path.exists():
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()


class RealDataPipeline:
    """
    Complete pipeline for real data acquisition
    Searches, downloads, validates, cleans, and tracks all real datasets
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/real")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.discovery = DatasetDiscoveryEngine()
        self.downloader = DatasetDownloader()
        self.validator = DatasetValidator()
        self.lineage = DataLineageTracker()
        
        # Directories
        self.raw_dir = self.data_dir / "raw"
        self.cleaned_dir = self.data_dir / "cleaned"
        self.raw_dir.mkdir(exist_ok=True)
        self.cleaned_dir.mkdir(exist_ok=True)
        
        logger.info("✓ Real Data Pipeline initialized")
        logger.info(f"  Data directory: {self.data_dir}")
    
    def search_all_sources(self, keywords: List[str], limit_per_source: int = 100) -> List[Dict]:
        """
        Search ALL public data sources
        Returns comprehensive list of available datasets
        """
        logger.info("\n" + "="*80)
        logger.info("SEARCHING ALL PUBLIC DATA SOURCES")
        logger.info("="*80)
        logger.info(f"Keywords: {', '.join(keywords)}")
        logger.info(f"Limit per source: {limit_per_source}")
        
        all_datasets = []
        
        # Search Figshare
        logger.info("\n1. Searching Figshare...")
        try:
            figshare_results = self.discovery.search_figshare(keywords, limit_per_source)
            all_datasets.extend(figshare_results)
            logger.info(f"   ✓ Found {len(figshare_results)} datasets")
        except Exception as e:
            logger.error(f"   ✗ Figshare search failed: {e}")
        
        # Search Zenodo
        logger.info("\n2. Searching Zenodo...")
        try:
            zenodo_results = self.discovery.search_zenodo(keywords, limit_per_source)
            all_datasets.extend(zenodo_results)
            logger.info(f"   ✓ Found {len(zenodo_results)} datasets")
        except Exception as e:
            logger.error(f"   ✗ Zenodo search failed: {e}")
        
        # Search Data.gov
        logger.info("\n3. Searching Data.gov...")
        try:
            datagov_results = self.discovery.search_data_gov(keywords, limit_per_source)
            all_datasets.extend(datagov_results)
            logger.info(f"   ✓ Found {len(datagov_results)} datasets")
        except Exception as e:
            logger.error(f"   ✗ Data.gov search failed: {e}")
        
        # Search Kaggle
        logger.info("\n4. Searching Kaggle...")
        try:
            kaggle_results = self.discovery.search_kaggle(keywords, limit_per_source)
            all_datasets.extend(kaggle_results)
            logger.info(f"   ✓ Found {len(kaggle_results)} datasets")
        except Exception as e:
            logger.error(f"   ✗ Kaggle search failed: {e}")
        
        logger.info("\n" + "="*80)
        logger.info(f"TOTAL DATASETS FOUND: {len(all_datasets)}")
        logger.info("="*80)
        
        return all_datasets
    
    def download_dataset(self, dataset_info: Dict) -> Optional[Path]:
        """Download a single dataset"""
        
        logger.info(f"\nDownloading: {dataset_info.get('title', 'Unknown')[:60]}...")
        logger.info(f"  Source: {dataset_info['source']}")
        logger.info(f"  Size: {dataset_info.get('size', 0) / (1024**2):.1f} MB")
        
        try:
            # Download
            file_path = self.downloader.download_dataset(
                dataset_info,
                output_dir=self.raw_dir
            )
            
            if file_path and file_path.exists():
                # Register in lineage
                dataset_id = self.lineage.register_dataset(dataset_info, file_path)
                
                logger.info(f"  ✓ Downloaded to: {file_path}")
                logger.info(f"  ✓ Registered as: {dataset_id}")
                
                return file_path
            else:
                logger.warning(f"  ✗ Download failed")
                return None
                
        except Exception as e:
            logger.error(f"  ✗ Error downloading: {e}")
            return None
    
    def download_all_discovered(self, datasets: List[Dict], 
                               max_size_gb: float = 50.0) -> List[Path]:
        """
        Download all discovered datasets
        Stops if total size exceeds max_size_gb
        """
        logger.info("\n" + "="*80)
        logger.info(f"DOWNLOADING ALL DATASETS (max {max_size_gb} GB)")
        logger.info("="*80)
        
        downloaded_files = []
        total_size = 0
        max_size_bytes = max_size_gb * (1024**3)
        
        for i, dataset in enumerate(datasets, 1):
            dataset_size = dataset.get('size', 0)
            
            # Check size limit
            if total_size + dataset_size > max_size_bytes:
                logger.warning(f"\n⚠ Size limit reached ({max_size_gb} GB)")
                logger.warning(f"  Downloaded {i-1}/{len(datasets)} datasets")
                break
            
            logger.info(f"\n[{i}/{len(datasets)}]")
            
            file_path = self.download_dataset(dataset)
            
            if file_path:
                downloaded_files.append(file_path)
                total_size += dataset_size
        
        logger.info("\n" + "="*80)
        logger.info(f"DOWNLOAD COMPLETE")
        logger.info("="*80)
        logger.info(f"  Downloaded: {len(downloaded_files)} datasets")
        logger.info(f"  Total size: {total_size / (1024**3):.2f} GB")
        
        return downloaded_files
    
    def validate_dataset(self, dataset_id: str) -> bool:
        """Validate a downloaded dataset"""
        
        dataset_info = self.lineage.get_dataset_info(dataset_id)
        if not dataset_info:
            logger.error(f"Dataset {dataset_id} not found in lineage")
            return False
        
        file_path = Path(dataset_info['file_path'])
        
        logger.info(f"\nValidating: {dataset_id}")
        
        try:
            validation_result = self.validator.validate_dataset(file_path)
            
            # Update lineage
            self.lineage.update_validation_status(
                dataset_id,
                'valid' if validation_result['is_valid'] else 'invalid',
                validation_result
            )
            
            if validation_result['is_valid']:
                logger.info(f"  ✓ Valid dataset")
                logger.info(f"    Rows: {validation_result.get('row_count', 'unknown')}")
                logger.info(f"    Columns: {validation_result.get('column_count', 'unknown')}")
            else:
                logger.warning(f"  ✗ Invalid dataset")
                logger.warning(f"    Issues: {validation_result.get('issues', [])}")
            
            return validation_result['is_valid']
            
        except Exception as e:
            logger.error(f"  ✗ Validation error: {e}")
            self.lineage.update_validation_status(dataset_id, 'error', {'error': str(e)})
            return False
    
    def validate_all(self) -> Dict:
        """Validate all downloaded datasets"""
        
        logger.info("\n" + "="*80)
        logger.info("VALIDATING ALL DATASETS")
        logger.info("="*80)
        
        all_datasets = self.lineage.get_all_datasets()
        
        valid_count = 0
        invalid_count = 0
        error_count = 0
        
        for dataset in all_datasets:
            dataset_id = dataset['dataset_id']
            
            if dataset['validation_status'] == 'pending':
                is_valid = self.validate_dataset(dataset_id)
                
                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
            elif dataset['validation_status'] == 'valid':
                valid_count += 1
            elif dataset['validation_status'] == 'invalid':
                invalid_count += 1
            else:
                error_count += 1
        
        logger.info("\n" + "="*80)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*80)
        logger.info(f"  Valid: {valid_count}")
        logger.info(f"  Invalid: {invalid_count}")
        logger.info(f"  Errors: {error_count}")
        
        return {
            'valid': valid_count,
            'invalid': invalid_count,
            'errors': error_count
        }
    
    def clean_dataset(self, dataset_id: str) -> Optional[Path]:
        """Clean a validated dataset"""
        
        dataset_info = self.lineage.get_dataset_info(dataset_id)
        if not dataset_info:
            return None
        
        if dataset_info['validation_status'] != 'valid':
            logger.warning(f"Dataset {dataset_id} not validated, skipping cleaning")
            return None
        
        file_path = Path(dataset_info['file_path'])
        
        logger.info(f"\nCleaning: {dataset_id}")
        
        try:
            # Load data
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                logger.warning(f"  Unsupported file type: {file_path.suffix}")
                return None
            
            # Basic cleaning
            original_rows = len(df)
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Remove rows with all NaN
            df = df.dropna(how='all')
            
            # Remove columns with all NaN
            df = df.dropna(axis=1, how='all')
            
            cleaned_rows = len(df)
            
            # Save cleaned data
            cleaned_path = self.cleaned_dir / f"{dataset_id}_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            
            # Update lineage
            self.lineage.mark_as_cleaned(dataset_id, cleaned_path)
            
            logger.info(f"  ✓ Cleaned dataset saved")
            logger.info(f"    Original rows: {original_rows}")
            logger.info(f"    Cleaned rows: {cleaned_rows}")
            logger.info(f"    Removed: {original_rows - cleaned_rows} rows")
            logger.info(f"    Path: {cleaned_path}")
            
            return cleaned_path
            
        except Exception as e:
            logger.error(f"  ✗ Cleaning error: {e}")
            return None
    
    def clean_all_valid(self) -> List[Path]:
        """Clean all valid datasets"""
        
        logger.info("\n" + "="*80)
        logger.info("CLEANING ALL VALID DATASETS")
        logger.info("="*80)
        
        valid_datasets = self.lineage.get_all_datasets(validated_only=True)
        cleaned_files = []
        
        for dataset in valid_datasets:
            if not dataset.get('cleaned', False):
                cleaned_path = self.clean_dataset(dataset['dataset_id'])
                if cleaned_path:
                    cleaned_files.append(cleaned_path)
        
        logger.info("\n" + "="*80)
        logger.info(f"CLEANING COMPLETE: {len(cleaned_files)} datasets cleaned")
        logger.info("="*80)
        
        return cleaned_files
    
    def run_daily_acquisition(self, keywords: List[str], 
                             max_download_gb: float = 10.0):
        """
        Run complete daily acquisition cycle
        Search → Download → Validate → Clean
        """
        logger.info("\n" + "="*80)
        logger.info("DAILY REAL DATA ACQUISITION CYCLE")
        logger.info("="*80)
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Search
        datasets = self.search_all_sources(keywords, limit_per_source=50)
        
        # Step 2: Download
        downloaded = self.download_all_discovered(datasets, max_size_gb=max_download_gb)
        
        # Step 3: Validate
        validation_results = self.validate_all()
        
        # Step 4: Clean
        cleaned = self.clean_all_valid()
        
        # Step 5: Report
        stats = self.lineage.get_statistics()
        
        logger.info("\n" + "="*80)
        logger.info("DAILY ACQUISITION COMPLETE")
        logger.info("="*80)
        logger.info(f"  Datasets found: {len(datasets)}")
        logger.info(f"  Datasets downloaded: {len(downloaded)}")
        logger.info(f"  Datasets validated: {validation_results['valid']}")
        logger.info(f"  Datasets cleaned: {len(cleaned)}")
        logger.info(f"  Total in repository: {stats['total_datasets']}")
        logger.info(f"  Total size: {stats['total_size_gb']:.2f} GB")
        
        return {
            'found': len(datasets),
            'downloaded': len(downloaded),
            'validated': validation_results['valid'],
            'cleaned': len(cleaned),
            'statistics': stats
        }
    
    def get_report(self) -> Dict:
        """Generate comprehensive report"""
        stats = self.lineage.get_statistics()
        
        return {
            'generated_at': datetime.now().isoformat(),
            'statistics': stats,
            'datasets': self.lineage.get_all_datasets(),
            'valid_datasets': self.lineage.get_all_datasets(validated_only=True),
            'cleaned_datasets': [d for d in self.lineage.get_all_datasets() 
                               if d.get('cleaned', False)]
        }


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("REAL DATA ACQUISITION PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = RealDataPipeline()
    
    # Define search keywords
    keywords = [
        'diabetes',
        'cardiovascular disease',
        'patient records',
        'clinical trial',
        'EHR',
        'electronic health records',
        'hypertension',
        'metabolic syndrome'
    ]
    
    # Run daily acquisition
    result = pipeline.run_daily_acquisition(
        keywords=keywords,
        max_download_gb=5.0  # Limit to 5GB for demo
    )
    
    # Generate report
    report = pipeline.get_report()
    
    print("\n" + "="*80)
    print("ACQUISITION REPORT")
    print("="*80)
    print(f"Total datasets: {report['statistics']['total_datasets']}")
    print(f"Total size: {report['statistics']['total_size_gb']:.2f} GB")
    print(f"Validated: {report['statistics']['validated_datasets']}")
    print(f"Cleaned: {report['statistics']['cleaned_datasets']}")
    print(f"\nSources:")
    for source, info in report['statistics']['source_breakdown'].items():
        print(f"  {source}: {info['total_datasets']} datasets ({info['total_size_bytes']/(1024**2):.1f} MB)")
