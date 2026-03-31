"""
NHANES CSV Data Loader
Loads harmonized NHANES data from CSV files (1988-2018)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from .nhanes_variable_mapping import NHANESVariableMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHANESCSVLoader:
    """
    Load NHANES data from harmonized CSV files
    This handles the cleaned/harmonized NHANES dataset (1988-2018)
    """
    
    def __init__(self, data_path: str = "./data/nhanes/raw_csv"):
        """
        Initialize NHANES CSV loader
        
        Args:
            data_path: Path to directory containing NHANES CSV files
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            logger.warning(f"NHANES path not found: {self.data_path}")
            logger.info("Extract NHANES data to this directory")
        
        # Cache for loaded data
        self._demographics = None
        self._questionnaire = None
        self._chemicals = None
        self._medications = None
        
        # Variable mapper
        self.mapper = NHANESVariableMapper()
    
    def load_demographics(self) -> pd.DataFrame:
        """Load demographics data"""
        if self._demographics is None:
            demo_file = self.data_path / "demographics_clean.csv"
            if demo_file.exists():
                logger.info(f"Loading demographics from {demo_file}")
                self._demographics = pd.read_csv(demo_file)
                logger.info(f"Loaded {len(self._demographics)} patients")
            else:
                logger.warning(f"Demographics file not found: {demo_file}")
                self._demographics = pd.DataFrame()
        
        return self._demographics
    
    def load_questionnaire(self) -> pd.DataFrame:
        """Load questionnaire data (health conditions, lifestyle)"""
        if self._questionnaire is None:
            quest_file = self.data_path / "questionnaire_clean.csv"
            if quest_file.exists():
                logger.info(f"Loading questionnaire from {quest_file}")
                self._questionnaire = pd.read_csv(quest_file)
                logger.info(f"Loaded {len(self._questionnaire)} patient questionnaires")
            else:
                logger.warning(f"Questionnaire file not found: {quest_file}")
                self._questionnaire = pd.DataFrame()
        
        return self._questionnaire
    
    def load_chemicals(self) -> pd.DataFrame:
        """Load laboratory/chemical data"""
        if self._chemicals is None:
            chem_file = self.data_path / "chemicals_clean.csv"
            if chem_file.exists():
                logger.info(f"Loading chemicals from {chem_file}")
                # This file is large, load in chunks if needed
                self._chemicals = pd.read_csv(chem_file)
                logger.info(f"Loaded {len(self._chemicals)} chemical measurements")
            else:
                logger.warning(f"Chemicals file not found: {chem_file}")
                self._chemicals = pd.DataFrame()
        
        return self._chemicals
    
    def load_medications(self) -> pd.DataFrame:
        """Load medication data"""
        if self._medications is None:
            med_file = self.data_path / "medications_clean.csv"
            if med_file.exists():
                logger.info(f"Loading medications from {med_file}")
                self._medications = pd.read_csv(med_file)
                logger.info(f"Loaded {len(self._medications)} medication records")
            else:
                logger.warning(f"Medications file not found: {med_file}")
                self._medications = pd.DataFrame()
        
        return self._medications
    
    def get_patient_data(self, seqn: int) -> Dict:
        """
        Get all data for a single patient by SEQN (sequence number)
        
        Args:
            seqn: Patient sequence number
            
        Returns:
            Dictionary with patient data
        """
        patient_data = {}
        
        # Demographics
        demo = self.load_demographics()
        if not demo.empty and 'SEQN' in demo.columns:
            patient_demo = demo[demo['SEQN'] == seqn]
            if not patient_demo.empty:
                patient_data['demographics'] = patient_demo.iloc[0].to_dict()
        
        # Questionnaire
        quest = self.load_questionnaire()
        if not quest.empty and 'SEQN' in quest.columns:
            patient_quest = quest[quest['SEQN'] == seqn]
            if not patient_quest.empty:
                patient_data['questionnaire'] = patient_quest.iloc[0].to_dict()
        
        # Chemicals (may have multiple rows per patient)
        chem = self.load_chemicals()
        if not chem.empty and 'SEQN' in chem.columns:
            patient_chem = chem[chem['SEQN'] == seqn]
            if not patient_chem.empty:
                patient_data['chemicals'] = patient_chem.to_dict('records')
        
        # Medications (may have multiple rows per patient)
        meds = self.load_medications()
        if not meds.empty and 'SEQN' in meds.columns:
            patient_meds = meds[meds['SEQN'] == seqn]
            if not patient_meds.empty:
                patient_data['medications'] = patient_meds.to_dict('records')
        
        return patient_data
    
    def extract_patient_features(self, seqn: int) -> Dict:
        """
        Extract standardized features for a patient using variable mapper
        
        Args:
            seqn: Patient sequence number
            
        Returns:
            Dictionary with standardized features
        """
        patient_data = self.get_patient_data(seqn)
        
        if not patient_data:
            return {}
        
        demo_series = pd.Series(patient_data.get('demographics', {}))
        quest_series = pd.Series(patient_data.get('questionnaire', {}))
        
        chem_dict = {}
        if 'chemicals' in patient_data and patient_data['chemicals']:
            chem_dict = patient_data['chemicals'][0]
        chem_series = pd.Series(chem_dict)
        
        mapped_data = self.mapper.map_patient_data(demo_series, quest_series, chem_series)
        
        standardized = self.mapper.standardize_values(mapped_data)
        
        standardized['patient_id'] = str(seqn)
        standardized['source'] = 'NHANES'
        
        return standardized
    
    def get_patient_cohort(self, max_patients: Optional[int] = None, 
                          min_age: int = 18, max_age: int = 90) -> List[Dict]:
        """
        Get a cohort of patients with complete data
        
        Args:
            max_patients: Maximum number of patients to return
            min_age: Minimum age filter
            max_age: Maximum age filter
            
        Returns:
            List of patient feature dictionaries
        """
        demo = self.load_demographics()
        
        if demo.empty:
            logger.error("No demographics data available")
            return []
        
        # Filter by age
        if 'RIDAGEYR' in demo.columns:
            demo = demo[(demo['RIDAGEYR'] >= min_age) & (demo['RIDAGEYR'] <= max_age)]
        
        # Get unique patient IDs
        if 'SEQN' not in demo.columns:
            logger.error("SEQN column not found in demographics")
            return []
        
        patient_ids = demo['SEQN'].unique()
        
        if max_patients:
            patient_ids = patient_ids[:max_patients]
        
        logger.info(f"Processing {len(patient_ids)} patients...")
        
        cohort = []
        for i, seqn in enumerate(patient_ids):
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(patient_ids)} patients")
            
            features = self.extract_patient_features(seqn)
            
            # Only include patients with minimum required data
            if features and 'age' in features and 'sex' in features:
                cohort.append(features)
        
        logger.info(f"Extracted {len(cohort)} patients with complete data")
        
        return cohort
