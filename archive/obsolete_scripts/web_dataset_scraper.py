#!/usr/bin/env python3
"""
Web Scraper for Medical Datasets
Searches across multiple repositories and websites
"""

import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse
import re


class DatasetScraper:
    """Scrape medical datasets from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.found_datasets = []
    
    def search_kaggle_web(self, query: str):
        """Search Kaggle via web (no API needed)"""
        print(f"\n🔍 Searching Kaggle for: {query}")
        
        try:
            url = f"https://www.kaggle.com/search?q={query.replace(' ', '+')}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find dataset links
                datasets = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if '/datasets/' in href and href.count('/') >= 3:
                        dataset_name = href.split('/datasets/')[-1]
                        if dataset_name and '?' not in dataset_name:
                            datasets.append({
                                'name': dataset_name,
                                'url': f"https://www.kaggle.com{href}",
                                'source': 'Kaggle'
                            })
                
                # Remove duplicates
                unique = {d['name']: d for d in datasets}
                datasets = list(unique.values())
                
                print(f"  ✓ Found {len(datasets)} datasets")
                return datasets
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        return []
    
    def search_uci_repository(self):
        """Scrape UCI ML Repository for health datasets"""
        print("\n🔍 Scraping UCI ML Repository...")
        
        try:
            url = "https://archive.ics.uci.edu/ml/datasets.php"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                datasets = []
                
                # Find all dataset rows
                for row in soup.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        name_cell = cells[0]
                        link = name_cell.find('a')
                        
                        if link:
                            name = link.text.strip()
                            # Filter for health-related
                            health_keywords = [
                                'heart', 'diabetes', 'cancer', 'disease',
                                'health', 'medical', 'patient', 'clinical',
                                'liver', 'kidney', 'thyroid', 'breast',
                                'lung', 'stroke', 'parkinsons', 'alzheimer'
                            ]
                            
                            if any(kw in name.lower() for kw in health_keywords):
                                datasets.append({
                                    'name': name,
                                    'url': urljoin(url, link['href']),
                                    'source': 'UCI'
                                })
                
                print(f"  ✓ Found {len(datasets)} health-related datasets")
                return datasets
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        return []
    
    def search_github_datasets(self):
        """Search GitHub for medical datasets"""
        print("\n🔍 Searching GitHub for medical datasets...")
        
        queries = [
            "medical dataset csv",
            "health dataset",
            "diabetes dataset",
            "heart disease dataset",
            "cancer dataset"
        ]
        
        all_datasets = []
        
        for query in queries:
            try:
                url = f"https://github.com/search?q={query.replace(' ', '+')}&type=repositories"
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for repo in soup.find_all('a', {'class': 'v-align-middle'}):
                        if repo.get('href'):
                            repo_url = f"https://github.com{repo['href']}"
                            all_datasets.append({
                                'name': repo.text.strip(),
                                'url': repo_url,
                                'source': 'GitHub'
                            })
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Error for '{query}': {e}")
        
        # Remove duplicates
        unique = {d['url']: d for d in all_datasets}
        datasets = list(unique.values())
        
        print(f"  ✓ Found {len(datasets)} GitHub repositories")
        return datasets
    
    def search_awesome_lists(self):
        """Search awesome-public-datasets for medical data"""
        print("\n🔍 Searching Awesome Public Datasets...")
        
        try:
            url = "https://raw.githubusercontent.com/awesomedata/awesome-public-datasets/master/README.rst"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                content = response.text
                
                # Find healthcare section
                datasets = []
                in_healthcare = False
                
                for line in content.split('\n'):
                    if 'Healthcare' in line or 'Medical' in line:
                        in_healthcare = True
                    elif line.startswith('##'):
                        in_healthcare = False
                    
                    if in_healthcare and 'http' in line:
                        # Extract URLs
                        urls = re.findall(r'https?://[^\s<>"]+', line)
                        for url in urls:
                            datasets.append({
                                'name': line.split('`')[1] if '`' in line else 'Dataset',
                                'url': url,
                                'source': 'Awesome Public Datasets'
                            })
                
                print(f"  ✓ Found {len(datasets)} datasets")
                return datasets
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        return []
    
    def search_data_world(self):
        """Search data.world for medical datasets"""
        print("\n🔍 Searching data.world...")
        
        queries = ['health', 'medical', 'diabetes', 'heart disease']
        all_datasets = []
        
        for query in queries:
            try:
                url = f"https://data.world/search?q={query.replace(' ', '+')}&type=dataset"
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        if '/dataset/' in link['href']:
                            all_datasets.append({
                                'name': link.text.strip() or 'Dataset',
                                'url': urljoin(url, link['href']),
                                'source': 'data.world'
                            })
                
                time.sleep(2)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        unique = {d['url']: d for d in all_datasets}
        datasets = list(unique.values())
        
        print(f"  ✓ Found {len(datasets)} datasets")
        return datasets
    
    def search_all(self):
        """Search all sources"""
        print("=" * 80)
        print("WEB SCRAPER - SEARCHING FOR MEDICAL DATASETS")
        print("=" * 80)
        
        all_datasets = []
        
        # Kaggle
        for query in ['diabetes', 'heart disease', 'cancer', 'kidney disease', 'liver disease']:
            all_datasets.extend(self.search_kaggle_web(query))
            time.sleep(2)
        
        # UCI
        all_datasets.extend(self.search_uci_repository())
        time.sleep(2)
        
        # GitHub
        all_datasets.extend(self.search_github_datasets())
        time.sleep(2)
        
        # Awesome Lists
        all_datasets.extend(self.search_awesome_lists())
        time.sleep(2)
        
        # data.world
        all_datasets.extend(self.search_data_world())
        
        # Remove duplicates
        unique = {}
        for ds in all_datasets:
            key = ds['url']
            if key not in unique:
                unique[key] = ds
        
        self.found_datasets = list(unique.values())
        
        return self.found_datasets
    
    def save_results(self, output_file: str = "data/found_datasets.json"):
        """Save found datasets to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.found_datasets, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")
        
        # Also create markdown report
        md_file = output_path.with_suffix('.md')
        with open(md_file, 'w') as f:
            f.write("# Found Medical Datasets\n\n")
            
            by_source = {}
            for ds in self.found_datasets:
                source = ds['source']
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(ds)
            
            for source, datasets in sorted(by_source.items()):
                f.write(f"## {source} ({len(datasets)} datasets)\n\n")
                for ds in datasets:
                    f.write(f"- [{ds['name']}]({ds['url']})\n")
                f.write("\n")
        
        print(f"✓ Markdown report: {md_file}")


def main():
    scraper = DatasetScraper()
    datasets = scraper.search_all()
    
    print("\n" + "=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)
    
    by_source = {}
    for ds in datasets:
        source = ds['source']
        by_source[source] = by_source.get(source, 0) + 1
    
    print(f"\n📊 Total datasets found: {len(datasets)}")
    print("\nBy source:")
    for source, count in sorted(by_source.items()):
        print(f"  • {source}: {count}")
    
    scraper.save_results()
    
    print("\n🎯 Next: Review data/found_datasets.json and download relevant ones")


if __name__ == "__main__":
    main()
