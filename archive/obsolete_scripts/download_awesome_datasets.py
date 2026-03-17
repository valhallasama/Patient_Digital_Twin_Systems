#!/usr/bin/env python3
"""
Download Medical Datasets from Awesome Public Datasets List
Filters for relevant medical/patient data and downloads
"""

import json
import requests
from pathlib import Path
import time
from urllib.parse import urlparse
import re
from bs4 import BeautifulSoup

# Keywords for medical/patient-level data
MEDICAL_KEYWORDS = [
    'health', 'medical', 'patient', 'disease', 'clinical',
    'diabetes', 'heart', 'cancer', 'kidney', 'liver',
    'stroke', 'hospital', 'diagnosis', 'treatment',
    'drug', 'medication', 'symptom', 'mortality',
    'epidemiology', 'public health', 'healthcare',
    'mental health', 'obesity', 'hypertension',
    'cardiovascular', 'respiratory', 'neurological'
]

# Keywords to EXCLUDE (not patient-level)
EXCLUDE_KEYWORDS = [
    'genome', 'genomic', 'dna', 'rna', 'gene', 'protein',
    'molecular', 'sequencing', 'microbiome', 'genetic',
    'mutation', 'chromosome', 'nucleotide', 'amino acid'
]


def load_found_datasets():
    """Load the datasets found by web scraper"""
    with open('data/found_datasets.json', 'r') as f:
        return json.load(f)


def is_relevant_medical_dataset(dataset):
    """Check if dataset is relevant patient-level medical data"""
    name = dataset['name'].lower()
    url = dataset['url'].lower()
    
    # Check for medical keywords
    has_medical = any(kw in name or kw in url for kw in MEDICAL_KEYWORDS)
    
    # Check for excluded keywords (genomics, etc.)
    has_excluded = any(kw in name or kw in url for kw in EXCLUDE_KEYWORDS)
    
    return has_medical and not has_excluded


def filter_datasets(datasets):
    """Filter for relevant medical datasets"""
    print("\n🔍 Filtering 1,544 datasets for relevant medical data...")
    
    relevant = []
    
    for ds in datasets:
        if is_relevant_medical_dataset(ds):
            relevant.append(ds)
    
    print(f"✓ Found {len(relevant)} relevant medical datasets (out of {len(datasets)})")
    
    return relevant


def is_downloadable_url(url):
    """Check if URL is directly downloadable"""
    # Direct file extensions
    downloadable_extensions = [
        '.csv', '.xlsx', '.xls', '.json', '.xml',
        '.zip', '.tar', '.gz', '.7z',
        '.txt', '.tsv', '.parquet'
    ]
    
    return any(url.lower().endswith(ext) for ext in downloadable_extensions)


def try_download_dataset(dataset, output_dir):
    """Attempt to download a dataset"""
    name = dataset['name']
    url = dataset['url']
    
    # Skip GitHub YAML files
    if 'github.com/awesomedata' in url and url.endswith('.yml'):
        return False
    
    print(f"\n📥 Attempting: {name[:80]}...")
    print(f"   URL: {url}")
    
    try:
        # Try direct download
        response = requests.get(url, timeout=30, allow_redirects=True)
        
        if response.status_code == 200:
            # Determine filename
            parsed = urlparse(url)
            filename = Path(parsed.path).name
            
            if not filename or len(filename) < 3:
                filename = f"dataset_{hash(url) % 10000}.data"
            
            # Save file
            output_file = output_dir / filename
            
            # Check if it's HTML (landing page, not data)
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type and not is_downloadable_url(url):
                print(f"   ⚠️  Landing page (not direct download)")
                
                # Try to find download links in HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                download_links = []
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if any(ext in href.lower() for ext in ['.csv', '.xlsx', '.zip', '.json']):
                        download_links.append(href)
                
                if download_links:
                    print(f"   📋 Found {len(download_links)} potential download links")
                    # Save links for manual review
                    links_file = output_dir / f"{filename}_links.txt"
                    with open(links_file, 'w') as f:
                        f.write(f"Dataset: {name}\n")
                        f.write(f"URL: {url}\n\n")
                        f.write("Potential download links:\n")
                        for link in download_links[:10]:
                            f.write(f"  - {link}\n")
                    print(f"   ✓ Links saved to {links_file.name}")
                
                return False
            
            # Save actual data file
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content) / 1024  # KB
            print(f"   ✅ Downloaded: {filename} ({file_size:.1f} KB)")
            
            return True
        else:
            print(f"   ❌ HTTP {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout")
        return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection error")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_filtered_datasets(datasets, output_dir, max_downloads=50):
    """Download filtered datasets with limit"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading up to {max_downloads} datasets...")
    print(f"   Output: {output_dir}")
    
    downloaded = []
    failed = []
    landing_pages = []
    
    for i, dataset in enumerate(datasets[:max_downloads]):
        print(f"\n[{i+1}/{min(len(datasets), max_downloads)}]")
        
        success = try_download_dataset(dataset, output_dir)
        
        if success:
            downloaded.append(dataset)
        else:
            failed.append(dataset)
        
        # Rate limiting
        time.sleep(2)
    
    return downloaded, failed


def create_download_report(relevant, downloaded, failed, output_dir):
    """Create report of download results"""
    report = {
        'total_found': 1544,
        'relevant_filtered': len(relevant),
        'attempted': len(downloaded) + len(failed),
        'successful': len(downloaded),
        'failed': len(failed),
        'downloaded_datasets': [
            {'name': ds['name'], 'url': ds['url']} for ds in downloaded
        ],
        'failed_datasets': [
            {'name': ds['name'], 'url': ds['url']} for ds in failed
        ]
    }
    
    report_file = output_dir / 'download_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to {report_file}")
    
    # Also create markdown report
    md_file = output_dir / 'download_report.md'
    with open(md_file, 'w') as f:
        f.write("# Awesome Public Datasets Download Report\n\n")
        f.write(f"**Total datasets found:** 1,544\n\n")
        f.write(f"**Relevant medical datasets:** {len(relevant)}\n\n")
        f.write(f"**Download attempts:** {len(downloaded) + len(failed)}\n\n")
        f.write(f"**Successful:** {len(downloaded)}\n\n")
        f.write(f"**Failed:** {len(failed)}\n\n")
        
        f.write("## Successfully Downloaded\n\n")
        for ds in downloaded:
            f.write(f"- [{ds['name'][:100]}]({ds['url']})\n")
        
        f.write("\n## Failed Downloads\n\n")
        for ds in failed[:20]:  # Limit to first 20
            f.write(f"- [{ds['name'][:100]}]({ds['url']})\n")
    
    print(f"✓ Markdown report: {md_file}")
    
    return report


def main():
    print("=" * 80)
    print("DOWNLOAD AWESOME PUBLIC DATASETS (FILTERED)")
    print("=" * 80)
    
    # Load datasets
    all_datasets = load_found_datasets()
    print(f"\n📊 Total datasets found by scraper: {len(all_datasets)}")
    
    # Filter for relevant medical data
    relevant = filter_datasets(all_datasets)
    
    if not relevant:
        print("\n⚠️  No relevant medical datasets found after filtering")
        return
    
    # Show sample
    print("\n📋 Sample of relevant datasets:")
    for ds in relevant[:10]:
        print(f"  • {ds['name'][:80]}")
    
    # Ask user for confirmation
    print(f"\n🎯 Found {len(relevant)} relevant medical datasets")
    print(f"   (Filtered out {len(all_datasets) - len(relevant)} genomics/research datasets)")
    
    # Download
    output_dir = Path("data/awesome_datasets")
    
    print(f"\n⚠️  Note: Many may be landing pages, not direct downloads")
    print(f"   Will attempt up to 50 downloads (to avoid overwhelming)")
    
    downloaded, failed = download_filtered_datasets(relevant, output_dir, max_downloads=50)
    
    # Create report
    report = create_download_report(relevant, downloaded, failed, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    
    print(f"""
📊 Summary:
  • Total found: 1,544 datasets
  • Relevant medical: {len(relevant)} datasets
  • Download attempts: {len(downloaded) + len(failed)}
  • Successful: {len(downloaded)}
  • Failed/Landing pages: {len(failed)}

📁 Location: {output_dir}

⚠️  Note:
  Many "datasets" are actually landing pages or require authentication.
  Check {output_dir}/*_links.txt for datasets that need manual download.

🎯 Recommendation:
  You already have 108,818 real patients integrated.
  The downloaded files may not add much value.
  Focus on using what you have!
""")


if __name__ == "__main__":
    main()
