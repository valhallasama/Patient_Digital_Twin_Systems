import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetValidator:
    def __init__(self):
        self.validation_results = {}
        
    def validate_file(self, file_path: Path) -> Dict:
        validation = {
            'file_path': str(file_path),
            'exists': file_path.exists(),
            'size_bytes': 0,
            'readable': False,
            'format': None,
            'quality_score': 0.0,
            'issues': []
        }
        
        if not file_path.exists():
            validation['issues'].append('File does not exist')
            return validation
        
        validation['size_bytes'] = file_path.stat().st_size
        
        if validation['size_bytes'] == 0:
            validation['issues'].append('File is empty')
            return validation
        
        suffix = file_path.suffix.lower()
        validation['format'] = suffix
        
        try:
            if suffix == '.csv':
                df = pd.read_csv(file_path, nrows=1000)
                validation['readable'] = True
                validation.update(self._validate_dataframe(df))
                
            elif suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=1000)
                validation['readable'] = True
                validation.update(self._validate_dataframe(df))
                
            elif suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                validation['readable'] = True
                validation.update(self._validate_json(data))
                
            elif suffix == '.parquet':
                df = pd.read_parquet(file_path)
                validation['readable'] = True
                validation.update(self._validate_dataframe(df))
                
            else:
                validation['issues'].append(f'Unsupported format: {suffix}')
                
        except Exception as e:
            validation['readable'] = False
            validation['issues'].append(f'Error reading file: {str(e)}')
        
        validation['quality_score'] = self._compute_quality_score(validation)
        
        return validation
    
    def _validate_dataframe(self, df: pd.DataFrame) -> Dict:
        stats = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'column_names': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': {},
            'duplicate_rows': 0,
            'numeric_columns': [],
            'categorical_columns': []
        }
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            stats['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct)
            }
        
        stats['duplicate_rows'] = int(df.duplicated().sum())
        
        stats['numeric_columns'] = list(df.select_dtypes(include=[np.number]).columns)
        stats['categorical_columns'] = list(df.select_dtypes(include=['object', 'category']).columns)
        
        return stats
    
    def _validate_json(self, data) -> Dict:
        stats = {
            'data_type': type(data).__name__,
            'num_records': 0
        }
        
        if isinstance(data, list):
            stats['num_records'] = len(data)
            if len(data) > 0 and isinstance(data[0], dict):
                stats['keys'] = list(data[0].keys())
        elif isinstance(data, dict):
            stats['num_records'] = 1
            stats['keys'] = list(data.keys())
        
        return stats
    
    def _compute_quality_score(self, validation: Dict) -> float:
        score = 0.0
        
        if not validation['readable']:
            return 0.0
        
        if validation.get('num_rows', 0) > 1000:
            score += 0.3
        elif validation.get('num_rows', 0) > 100:
            score += 0.2
        elif validation.get('num_rows', 0) > 10:
            score += 0.1
        
        if validation.get('num_columns', 0) > 10:
            score += 0.2
        elif validation.get('num_columns', 0) > 5:
            score += 0.1
        
        if 'missing_values' in validation:
            avg_missing = np.mean([v['percentage'] for v in validation['missing_values'].values()])
            if avg_missing < 10:
                score += 0.3
            elif avg_missing < 30:
                score += 0.2
            elif avg_missing < 50:
                score += 0.1
        
        if validation.get('duplicate_rows', 0) == 0:
            score += 0.1
        
        if len(validation.get('issues', [])) == 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def validate_dataset(self, file_paths: List[Path]) -> Dict:
        dataset_validation = {
            'num_files': len(file_paths),
            'files': [],
            'overall_quality': 0.0,
            'readable_files': 0,
            'total_rows': 0
        }
        
        for file_path in file_paths:
            validation = self.validate_file(file_path)
            dataset_validation['files'].append(validation)
            
            if validation['readable']:
                dataset_validation['readable_files'] += 1
                dataset_validation['total_rows'] += validation.get('num_rows', 0)
        
        if dataset_validation['files']:
            avg_quality = np.mean([f['quality_score'] for f in dataset_validation['files']])
            dataset_validation['overall_quality'] = float(avg_quality)
        
        return dataset_validation


if __name__ == "__main__":
    validator = DatasetValidator()
    
    test_data = pd.DataFrame({
        'patient_id': range(1000),
        'age': np.random.randint(18, 90, 1000),
        'bmi': np.random.uniform(18, 40, 1000),
        'glucose': np.random.uniform(70, 200, 1000)
    })
    
    test_file = Path("data/test_dataset.csv")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_data.to_csv(test_file, index=False)
    
    result = validator.validate_file(test_file)
    logger.info(f"Validation result: {json.dumps(result, indent=2)}")
