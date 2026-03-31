#!/usr/bin/env python3
"""
Download NHANES 2017-2018 Data Files
Uses requests library for reliable downloads
"""

import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library not found")
    print("Install with: pip install requests")
    sys.exit(1)


def download_file(url: str, output_path: Path, description: str):
    """
    Download a file with progress indication
    
    Args:
        url: URL to download from
        output_path: Where to save the file
        description: Description for progress message
    """
    try:
        print(f"[{description}]")
        print(f"  URL: {url}")
        
        # Use requests with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"  ❌ HTTP Error {response.status_code}")
            return False
        
        data = response.content
        
        # Check if we got HTML instead of XPT
        if data.startswith(b'<!DOCTYPE') or data.startswith(b'<html'):
            print(f"  ❌ ERROR: Got HTML page instead of data file")
            print(f"  URL may be incorrect or file not available")
            return False
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(data)
        
        size_kb = len(data) / 1024
        print(f"  ✅ Downloaded: {size_kb:.1f} KB")
        
        # Verify it's an XPT file (SAS XPORT format starts with "HEADER RECORD")
        if not data.startswith(b'HEADER RECORD'):
            print(f"  ⚠️  Warning: File may not be valid SAS XPORT format")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Request Error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Download all NHANES 2017-2018 data files"""
    
    print("=" * 80)
    print("NHANES 2017-2018 Data Download")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path('./data/nhanes/2017-2018')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    # NHANES configuration
    BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"
    cycle = "2017-2018"
    suffix = "J"  # J = 2017-2018, I = 2015-2016, H = 2013-2014
    
    # Core files needed for digital twin (focused list)
    file_prefixes = [
        ("DEMO", "Demographics"),
        ("BMX", "Body Measures"),
        ("BPX", "Blood Pressure"),
        ("GLU", "Glucose"),
        ("GHB", "Glycohemoglobin (HbA1c)"),
        ("HDL", "HDL Cholesterol"),
        ("TCHOL", "Total Cholesterol"),
        ("TRIGLY", "Triglycerides & LDL"),
        ("BIOPRO", "Biochemistry Profile (ALT, AST, Creatinine)"),
        ("CRP", "C-Reactive Protein"),
        ("PAQ", "Physical Activity"),
        ("SMQ", "Smoking"),
        ("ALQ", "Alcohol Use"),
    ]
    
    # Build file list
    files = []
    for i, (prefix, description) in enumerate(file_prefixes, 1):
        filename = f"{prefix}_{suffix}.XPT"
        url = f"{BASE_URL}/{cycle}/{filename}"
        files.append((filename, f"{i}/{len(file_prefixes)} {description}", url))
    
    successful = 0
    failed = 0
    
    for filename, description, url in files:
        output_path = output_dir / filename
        
        # Skip if already exists and is valid
        if output_path.exists():
            size = output_path.stat().st_size
            if size > 1000:  # More than 1KB, likely valid
                print(f"[{description}]")
                print(f"  ⏭️  Already exists ({size/1024:.1f} KB), skipping")
                print()
                successful += 1
                continue
        
        if download_file(url, output_path, description):
            successful += 1
        else:
            failed += 1
        
        print()
    
    # Summary
    print("=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"✅ Successful: {successful}/{len(files)}")
    print(f"❌ Failed: {failed}/{len(files)}")
    print()
    
    if successful > 0:
        print("Downloaded files:")
        for f in sorted(output_dir.glob('*.XPT')):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:20s} {size_kb:8.1f} KB")
        print()
    
    if failed > 0:
        print("⚠️  Some files failed to download.")
        print("This may be due to:")
        print("  1. NHANES website changes")
        print("  2. Network issues")
        print("  3. Files moved to different URLs")
        print()
        print("Alternative: Download manually from:")
        print("  https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017")
        print()
        print("See NHANES_MANUAL_DOWNLOAD.md for detailed instructions.")
    else:
        print("✅ All files downloaded successfully!")
        print()
        print("Next steps:")
        print("  1. Test the loader: python3 examples/test_nhanes_loader.py")
        print("  2. Process data: python3 examples/process_nhanes_data.py")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
