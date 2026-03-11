import requests
import yaml
import logging
from typing import List, Dict, Optional
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDiscoveryEngine:
    def __init__(self, config_path: str = "config/data_sources.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.discovered_datasets = []
        
    def search_figshare(self, keywords: List[str], limit: int = 100) -> List[Dict]:
        base_url = self.config['research_repositories'][0]['api_url']
        results = []
        
        for keyword in keywords:
            try:
                url = f"{base_url}/articles"
                params = {
                    'search_for': keyword,
                    'page_size': limit
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    articles = response.json()
                    
                    for article in articles:
                        dataset_info = {
                            'source': 'figshare',
                            'title': article.get('title', ''),
                            'url': article.get('url', ''),
                            'doi': article.get('doi', ''),
                            'published_date': article.get('published_date', ''),
                            'size': article.get('size', 0),
                            'files': article.get('files', []),
                            'categories': [keyword],
                            'metadata': article
                        }
                        results.append(dataset_info)
                        
                logger.info(f"Found {len(articles)} datasets for keyword '{keyword}' on Figshare")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching Figshare for '{keyword}': {e}")
                
        return results
    
    def search_zenodo(self, keywords: List[str], limit: int = 100) -> List[Dict]:
        base_url = self.config['research_repositories'][1]['api_url']
        results = []
        
        for keyword in keywords:
            try:
                url = f"{base_url}/records"
                params = {
                    'q': keyword,
                    'size': limit,
                    'type': 'dataset'
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    records = data.get('hits', {}).get('hits', [])
                    
                    for record in records:
                        metadata = record.get('metadata', {})
                        dataset_info = {
                            'source': 'zenodo',
                            'title': metadata.get('title', ''),
                            'url': record.get('links', {}).get('self', ''),
                            'doi': metadata.get('doi', ''),
                            'published_date': metadata.get('publication_date', ''),
                            'size': sum([f.get('size', 0) for f in record.get('files', [])]),
                            'files': record.get('files', []),
                            'categories': [keyword],
                            'metadata': metadata
                        }
                        results.append(dataset_info)
                        
                logger.info(f"Found {len(records)} datasets for keyword '{keyword}' on Zenodo")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching Zenodo for '{keyword}': {e}")
                
        return results
    
    def search_data_gov(self, keywords: List[str], limit: int = 100) -> List[Dict]:
        base_url = self.config['research_repositories'][2]['api_url']
        results = []
        
        for keyword in keywords:
            try:
                url = f"{base_url}/action/package_search"
                params = {
                    'q': keyword,
                    'rows': limit
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    packages = data.get('result', {}).get('results', [])
                    
                    for package in packages:
                        dataset_info = {
                            'source': 'data.gov',
                            'title': package.get('title', ''),
                            'url': package.get('url', ''),
                            'published_date': package.get('metadata_created', ''),
                            'resources': package.get('resources', []),
                            'categories': [keyword],
                            'metadata': package
                        }
                        results.append(dataset_info)
                        
                logger.info(f"Found {len(packages)} datasets for keyword '{keyword}' on Data.gov")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching Data.gov for '{keyword}': {e}")
                
        return results
    
    def discover_all_datasets(self) -> List[Dict]:
        all_datasets = []
        
        figshare_keywords = self.config['research_repositories'][0]['search_keywords']
        zenodo_keywords = self.config['research_repositories'][1]['search_keywords']
        datagov_keywords = self.config['research_repositories'][2]['search_keywords']
        
        logger.info("Starting dataset discovery across all sources...")
        
        all_datasets.extend(self.search_figshare(figshare_keywords))
        all_datasets.extend(self.search_zenodo(zenodo_keywords))
        all_datasets.extend(self.search_data_gov(datagov_keywords))
        
        self.discovered_datasets = all_datasets
        logger.info(f"Total datasets discovered: {len(all_datasets)}")
        
        return all_datasets
    
    def rank_datasets(self, datasets: List[Dict]) -> List[Dict]:
        for dataset in datasets:
            score = 0.0
            
            if dataset.get('size', 0) > 1_000_000:
                score += 0.3
            elif dataset.get('size', 0) > 100_000:
                score += 0.2
            elif dataset.get('size', 0) > 10_000:
                score += 0.1
            
            title = dataset.get('title', '').lower()
            medical_keywords = ['health', 'medical', 'clinical', 'patient', 'disease', 
                              'cardiovascular', 'diabetes', 'cancer']
            relevance = sum([1 for kw in medical_keywords if kw in title])
            score += min(relevance * 0.1, 0.4)
            
            if dataset.get('doi'):
                score += 0.2
            
            if dataset.get('published_date'):
                score += 0.1
            
            dataset['quality_score'] = min(score, 1.0)
        
        ranked = sorted(datasets, key=lambda x: x['quality_score'], reverse=True)
        return ranked
    
    def filter_high_quality(self, datasets: List[Dict], threshold: float = 0.6) -> List[Dict]:
        return [d for d in datasets if d.get('quality_score', 0) >= threshold]


if __name__ == "__main__":
    engine = DatasetDiscoveryEngine()
    
    datasets = engine.discover_all_datasets()
    
    ranked_datasets = engine.rank_datasets(datasets)
    
    high_quality = engine.filter_high_quality(ranked_datasets, threshold=0.6)
    
    logger.info(f"\nHigh-quality datasets found: {len(high_quality)}")
    
    for i, dataset in enumerate(high_quality[:10], 1):
        logger.info(f"\n{i}. {dataset['title']}")
        logger.info(f"   Source: {dataset['source']}")
        logger.info(f"   Quality Score: {dataset['quality_score']:.2f}")
        logger.info(f"   URL: {dataset['url']}")
