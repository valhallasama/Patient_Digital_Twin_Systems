import requests
import logging
from pathlib import Path
from typing import Dict, Optional
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    def __init__(self, storage_path: str = "data/raw"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.download_log = self.storage_path / "download_log.json"
        self.downloaded_files = self._load_download_log()
        
    def _load_download_log(self) -> Dict:
        if self.download_log.exists():
            with open(self.download_log, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_download_log(self):
        with open(self.download_log, 'w') as f:
            json.dump(self.downloaded_files, f, indent=2)
    
    def _compute_hash(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def download_file(self, url: str, filename: Optional[str] = None, 
                     chunk_size: int = 8192) -> Optional[Path]:
        try:
            if filename is None:
                filename = url.split('/')[-1]
                if not filename or '.' not in filename:
                    filename = f"dataset_{hashlib.md5(url.encode()).hexdigest()}.dat"
            
            file_path = self.storage_path / filename
            
            if url in self.downloaded_files:
                logger.info(f"File already downloaded: {filename}")
                return file_path
            
            logger.info(f"Downloading: {url}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            
            file_hash = self._compute_hash(file_path)
            
            self.downloaded_files[url] = {
                'filename': filename,
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'hash': file_hash
            }
            self._save_download_log()
            
            logger.info(f"Successfully downloaded: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
    
    def download_dataset(self, dataset_info: Dict) -> Dict:
        results = {
            'dataset_title': dataset_info.get('title', 'Unknown'),
            'source': dataset_info.get('source', 'Unknown'),
            'files': [],
            'success': False
        }
        
        files_to_download = []
        
        if 'files' in dataset_info and dataset_info['files']:
            for file_info in dataset_info['files']:
                if isinstance(file_info, dict):
                    download_url = file_info.get('download_url') or file_info.get('url')
                    if download_url:
                        files_to_download.append(download_url)
        
        elif 'resources' in dataset_info and dataset_info['resources']:
            for resource in dataset_info['resources']:
                if isinstance(resource, dict):
                    download_url = resource.get('url')
                    if download_url:
                        files_to_download.append(download_url)
        
        elif 'url' in dataset_info:
            files_to_download.append(dataset_info['url'])
        
        if not files_to_download:
            logger.warning(f"No downloadable files found for: {dataset_info.get('title')}")
            return results
        
        for url in files_to_download:
            file_path = self.download_file(url)
            if file_path:
                results['files'].append(str(file_path))
        
        results['success'] = len(results['files']) > 0
        return results
    
    def download_multiple_datasets(self, datasets: list, max_workers: int = 5) -> list:
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_dataset = {
                executor.submit(self.download_dataset, dataset): dataset 
                for dataset in datasets
            }
            
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset.get('title')}: {e}")
        
        return results


if __name__ == "__main__":
    downloader = DatasetDownloader()
    
    test_dataset = {
        'title': 'Test Health Dataset',
        'source': 'test',
        'url': 'https://example.com/sample_data.csv'
    }
    
    result = downloader.download_dataset(test_dataset)
    logger.info(f"Download result: {result}")
