"""
LLM-Based Medical Report Parser
Extracts structured data from unstructured medical documents
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Demographics:
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    ethnicity: Optional[str] = None


@dataclass
class PhysicalMeasurements:
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    waist_circumference_cm: Optional[float] = None


@dataclass
class VitalSigns:
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    heart_rate: Optional[int] = None
    temperature_c: Optional[float] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None


@dataclass
class LabResults:
    glucose_mmol_l: Optional[float] = None
    hba1c_percent: Optional[float] = None
    total_cholesterol_mmol_l: Optional[float] = None
    ldl_cholesterol_mmol_l: Optional[float] = None
    hdl_cholesterol_mmol_l: Optional[float] = None
    triglycerides_mmol_l: Optional[float] = None
    creatinine_umol_l: Optional[float] = None
    egfr: Optional[float] = None
    alt: Optional[float] = None
    ast: Optional[float] = None


@dataclass
class LifestyleProfile:
    smoking_status: Optional[str] = None  # never, former, current
    smoking_pack_years: Optional[float] = None
    alcohol_units_per_week: Optional[float] = None
    exercise_hours_per_week: Optional[float] = None
    sleep_hours_per_night: Optional[float] = None
    diet_quality: Optional[str] = None  # poor, fair, good, excellent
    stress_level: Optional[str] = None  # low, moderate, high


@dataclass
class FamilyHistory:
    father_cvd: Optional[bool] = None
    father_cvd_age: Optional[int] = None
    mother_cvd: Optional[bool] = None
    mother_cvd_age: Optional[int] = None
    father_diabetes: Optional[bool] = None
    mother_diabetes: Optional[bool] = None
    father_cancer: Optional[bool] = None
    mother_cancer: Optional[bool] = None
    siblings_conditions: Optional[List[str]] = None


@dataclass
class MedicalHistory:
    current_conditions: Optional[List[str]] = None
    past_conditions: Optional[List[str]] = None
    surgeries: Optional[List[str]] = None
    allergies: Optional[List[str]] = None


@dataclass
class Medications:
    current_medications: Optional[List[Dict[str, str]]] = None


@dataclass
class ImagingFindings:
    ct_findings: Optional[str] = None
    mri_findings: Optional[str] = None
    xray_findings: Optional[str] = None
    ultrasound_findings: Optional[str] = None
    echocardiogram: Optional[str] = None


@dataclass
class StructuredPatientData:
    demographics: Demographics
    physical: PhysicalMeasurements
    vitals: VitalSigns
    labs: LabResults
    lifestyle: LifestyleProfile
    family_history: FamilyHistory
    medical_history: MedicalHistory
    medications: Medications
    imaging: ImagingFindings
    doctor_notes: Optional[str] = None
    raw_text: Optional[str] = None


class LLMMedicalParser:
    """
    Parse unstructured medical reports using LLM
    Supports: OpenAI GPT-4, Anthropic Claude, or local models
    """
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4", api_key: Optional[str] = None):
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize LLM client
        if model_provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                self.use_llm = True
            except ImportError:
                logger.warning("OpenAI not installed. Using rule-based parser.")
                self.use_llm = False
        elif model_provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                self.use_llm = True
            except ImportError:
                logger.warning("Anthropic not installed. Using rule-based parser.")
                self.use_llm = False
        else:
            logger.warning("Unknown provider. Using rule-based parser.")
            self.use_llm = False
    
    def create_extraction_prompt(self, medical_text: str) -> str:
        """Create prompt for LLM to extract structured data"""
        
        prompt = f"""You are a medical data extraction AI. Extract structured information from the following medical report.

Medical Report:
{medical_text}

Extract the following information in JSON format:

{{
  "demographics": {{
    "age": <int or null>,
    "gender": "<male/female or null>",
    "occupation": "<string or null>",
    "ethnicity": "<string or null>"
  }},
  "physical": {{
    "height_cm": <float or null>,
    "weight_kg": <float or null>,
    "bmi": <float or null>,
    "waist_circumference_cm": <float or null>
  }},
  "vitals": {{
    "systolic_bp": <float or null>,
    "diastolic_bp": <float or null>,
    "heart_rate": <int or null>,
    "temperature_c": <float or null>,
    "respiratory_rate": <int or null>,
    "oxygen_saturation": <float or null>
  }},
  "labs": {{
    "glucose_mmol_l": <float or null>,
    "hba1c_percent": <float or null>,
    "total_cholesterol_mmol_l": <float or null>,
    "ldl_cholesterol_mmol_l": <float or null>,
    "hdl_cholesterol_mmol_l": <float or null>,
    "triglycerides_mmol_l": <float or null>,
    "creatinine_umol_l": <float or null>,
    "egfr": <float or null>,
    "alt": <float or null>,
    "ast": <float or null>
  }},
  "lifestyle": {{
    "smoking_status": "<never/former/current or null>",
    "smoking_pack_years": <float or null>,
    "alcohol_units_per_week": <float or null>,
    "exercise_hours_per_week": <float or null>,
    "sleep_hours_per_night": <float or null>,
    "diet_quality": "<poor/fair/good/excellent or null>",
    "stress_level": "<low/moderate/high or null>"
  }},
  "family_history": {{
    "father_cvd": <bool or null>,
    "father_cvd_age": <int or null>,
    "mother_cvd": <bool or null>,
    "father_diabetes": <bool or null>,
    "mother_diabetes": <bool or null>
  }},
  "medical_history": {{
    "current_conditions": [<list of strings or null>],
    "past_conditions": [<list of strings or null>],
    "surgeries": [<list of strings or null>],
    "allergies": [<list of strings or null>]
  }},
  "medications": {{
    "current_medications": [
      {{"name": "<string>", "dose": "<string>", "frequency": "<string>"}}
    ]
  }},
  "imaging": {{
    "ct_findings": "<string or null>",
    "mri_findings": "<string or null>",
    "echocardiogram": "<string or null>"
  }}
}}

Important:
- Convert all units to standard format (mmol/L for glucose, cholesterol)
- Extract numeric values only
- Use null for missing information
- Be precise with medical terminology

Return ONLY the JSON, no additional text.
"""
        return prompt
    
    def parse_with_llm(self, medical_text: str) -> Dict[str, Any]:
        """Use LLM to parse medical text"""
        
        prompt = self.create_extraction_prompt(medical_text)
        
        try:
            if self.model_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a medical data extraction expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                result = response.choices[0].message.content
            elif self.model_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text
            else:
                return self.parse_with_rules(medical_text)
            
            # Parse JSON response
            extracted_data = json.loads(result)
            return extracted_data
            
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            logger.info("Falling back to rule-based parser")
            return self.parse_with_rules(medical_text)
    
    def parse_with_rules(self, medical_text: str) -> Dict[str, Any]:
        """Fallback rule-based parser (regex patterns)"""
        
        text = medical_text.lower()
        
        # Extract age
        age_match = re.search(r'(\d+)[\s-]?(?:year|yr|yo|y\.o\.)', text)
        age = int(age_match.group(1)) if age_match else None
        
        # Extract gender
        gender = None
        if 'male' in text and 'female' not in text:
            gender = 'male'
        elif 'female' in text:
            gender = 'female'
        
        # Extract BMI
        bmi_match = re.search(r'bmi[:\s]+(\d+\.?\d*)', text)
        bmi = float(bmi_match.group(1)) if bmi_match else None
        
        # Extract blood pressure
        bp_match = re.search(r'(\d{2,3})/(\d{2,3})', text)
        systolic = int(bp_match.group(1)) if bp_match else None
        diastolic = int(bp_match.group(2)) if bp_match else None
        
        # Extract smoking
        smoking_status = None
        if 'smoker' in text or 'smoking' in text:
            if 'non' in text or 'never' in text:
                smoking_status = 'never'
            elif 'former' in text or 'ex' in text:
                smoking_status = 'former'
            else:
                smoking_status = 'current'
        
        # Extract pack-years
        pack_years_match = re.search(r'(\d+)\s*pack[\s-]?years?', text)
        pack_years = float(pack_years_match.group(1)) if pack_years_match else None
        
        # Extract LDL
        ldl_match = re.search(r'ldl[:\s]+(\d+\.?\d*)', text)
        ldl = float(ldl_match.group(1)) if ldl_match else None
        
        # Extract HbA1c
        hba1c_match = re.search(r'hba1c[:\s]+(\d+\.?\d*)', text)
        hba1c = float(hba1c_match.group(1)) if hba1c_match else None
        
        return {
            "demographics": {"age": age, "gender": gender},
            "physical": {"bmi": bmi},
            "vitals": {"systolic_bp": systolic, "diastolic_bp": diastolic},
            "labs": {"ldl_cholesterol_mmol_l": ldl, "hba1c_percent": hba1c},
            "lifestyle": {"smoking_status": smoking_status, "smoking_pack_years": pack_years},
            "family_history": {},
            "medical_history": {},
            "medications": {},
            "imaging": {}
        }
    
    def parse(self, medical_text: str) -> StructuredPatientData:
        """Main parsing method"""
        
        logger.info("Parsing medical report...")
        
        if self.use_llm:
            extracted = self.parse_with_llm(medical_text)
        else:
            extracted = self.parse_with_rules(medical_text)
        
        # Convert to structured dataclasses
        structured_data = StructuredPatientData(
            demographics=Demographics(**extracted.get("demographics", {})),
            physical=PhysicalMeasurements(**extracted.get("physical", {})),
            vitals=VitalSigns(**extracted.get("vitals", {})),
            labs=LabResults(**extracted.get("labs", {})),
            lifestyle=LifestyleProfile(**extracted.get("lifestyle", {})),
            family_history=FamilyHistory(**extracted.get("family_history", {})),
            medical_history=MedicalHistory(**extracted.get("medical_history", {})),
            medications=Medications(**extracted.get("medications", {})),
            imaging=ImagingFindings(**extracted.get("imaging", {})),
            raw_text=medical_text
        )
        
        logger.info("✓ Parsing complete")
        return structured_data
    
    def to_dict(self, structured_data: StructuredPatientData) -> Dict:
        """Convert structured data to dictionary"""
        return asdict(structured_data)


# Example usage
if __name__ == "__main__":
    # Example medical report
    report = """
    Patient: 52-year-old male truck driver
    
    Physical Exam:
    - BMI: 31
    - BP: 145/92 mmHg
    - Heart rate: 78 bpm
    
    Labs:
    - LDL cholesterol: 4.1 mmol/L
    - HbA1c: 5.9%
    - Glucose: 6.2 mmol/L
    
    Social History:
    - Current smoker, 20 pack-years
    - Alcohol: 15 units/week
    - Exercise: minimal, sedentary lifestyle
    - Sleep: 5 hours per night
    
    Family History:
    - Father: MI at age 58
    
    Imaging:
    - CT coronary: mild coronary artery calcification
    
    Assessment:
    - Metabolic syndrome
    - High cardiovascular risk
    """
    
    # Initialize parser (will use rule-based if no API key)
    parser = LLMMedicalParser(model_provider="openai", api_key=None)
    
    # Parse report
    structured_data = parser.parse(report)
    
    # Display results
    print("\n" + "="*80)
    print("EXTRACTED STRUCTURED DATA")
    print("="*80)
    print(json.dumps(parser.to_dict(structured_data), indent=2, default=str))
