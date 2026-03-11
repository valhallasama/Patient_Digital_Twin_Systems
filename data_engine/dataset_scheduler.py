import schedule
import time
import logging
from datetime import datetime
from pathlib import Path
import yaml
from dataset_discovery import DatasetDiscoveryEngine
from dataset_downloader import DatasetDownloader
from dataset_validator import DatasetValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetScheduler:
    def __init__(self, config_path: str = "config/system_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.discovery_engine = DatasetDiscoveryEngine()
        self.downloader = DatasetDownloader()
        self.validator = DatasetValidator()
        
        self.auto_discovery = self.config['data_engine']['auto_discovery']
        self.interval_hours = self.config['data_engine']['discovery_interval_hours']
        self.quality_threshold = self.config['data_engine']['data_quality_threshold']
        
    def run_discovery_pipeline(self):
        logger.info(f"[{datetime.now()}] Starting automated dataset discovery pipeline...")
        
        try:
            datasets = self.discovery_engine.discover_all_datasets()
            logger.info(f"Discovered {len(datasets)} datasets")
            
            ranked_datasets = self.discovery_engine.rank_datasets(datasets)
            
            high_quality = self.discovery_engine.filter_high_quality(
                ranked_datasets, 
                threshold=self.quality_threshold
            )
            logger.info(f"Found {len(high_quality)} high-quality datasets")
            
            max_downloads = min(len(high_quality), 10)
            to_download = high_quality[:max_downloads]
            
            logger.info(f"Downloading top {len(to_download)} datasets...")
            download_results = self.downloader.download_multiple_datasets(
                to_download,
                max_workers=self.config['data_engine']['max_concurrent_downloads']
            )
            
            successful_downloads = [r for r in download_results if r['success']]
            logger.info(f"Successfully downloaded {len(successful_downloads)} datasets")
            
            for result in successful_downloads:
                if result['files']:
                    file_paths = [Path(f) for f in result['files']]
                    validation = self.validator.validate_dataset(file_paths)
                    logger.info(f"Dataset '{result['dataset_title']}' quality: {validation['overall_quality']:.2f}")
            
            logger.info("Discovery pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in discovery pipeline: {e}")
    
    def start_scheduler(self):
        if not self.auto_discovery:
            logger.info("Auto-discovery is disabled")
            return
        
        logger.info(f"Starting dataset scheduler (interval: {self.interval_hours} hours)")
        
        schedule.every(self.interval_hours).hours.do(self.run_discovery_pipeline)
        
        self.run_discovery_pipeline()
        
        while True:
            schedule.run_pending()
            time.sleep(60)


if __name__ == "__main__":
    scheduler = DatasetScheduler()
    scheduler.start_scheduler()
