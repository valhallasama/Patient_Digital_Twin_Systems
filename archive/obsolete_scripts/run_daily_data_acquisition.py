#!/usr/bin/env python3
"""
Automated Daily Real Data Acquisition
Run this script daily (via cron) to automatically download new health datasets
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_engine.real_data_pipeline import RealDataPipeline
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daily_acquisition.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run daily data acquisition"""
    
    logger.info("\n" + "="*80)
    logger.info("AUTOMATED DAILY DATA ACQUISITION")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize pipeline
    pipeline = RealDataPipeline()
    
    # Comprehensive health-related keywords
    keywords = [
        # Diseases
        'diabetes', 'cardiovascular disease', 'heart disease', 'hypertension',
        'obesity', 'metabolic syndrome', 'stroke', 'coronary artery disease',
        'chronic kidney disease', 'cancer', 'COPD', 'asthma',
        
        # Data types
        'patient records', 'clinical trial', 'EHR', 'electronic health records',
        'medical imaging', 'lab results', 'vital signs', 'wearable data',
        
        # Populations
        'population health', 'epidemiology', 'public health', 'cohort study',
        
        # Specific datasets
        'NHANES', 'UK Biobank', 'MIMIC', 'eICU', 'ADNI', 'TCGA'
    ]
    
    # Run acquisition (download up to 10GB per day)
    try:
        result = pipeline.run_daily_acquisition(
            keywords=keywords,
            max_download_gb=10.0
        )
        
        logger.info("\n" + "="*80)
        logger.info("DAILY ACQUISITION SUMMARY")
        logger.info("="*80)
        logger.info(f"  Datasets found: {result['found']}")
        logger.info(f"  Datasets downloaded: {result['downloaded']}")
        logger.info(f"  Datasets validated: {result['validated']}")
        logger.info(f"  Datasets cleaned: {result['cleaned']}")
        logger.info(f"  Repository total: {result['statistics']['total_datasets']}")
        logger.info(f"  Repository size: {result['statistics']['total_size_gb']:.2f} GB")
        
        # Generate report
        report = pipeline.get_report()
        
        # Save report
        report_path = Path('data/real/daily_report.json')
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n✓ Report saved: {report_path}")
        
        logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Daily acquisition failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
