#!/usr/bin/env python3
"""
LLM-Powered Medical Report Parser
Replaces regex-based extraction with intelligent LLM parsing
"""

from typing import Dict, List, Optional
import json
import re


class LLMMedicalParser:
    """
    Advanced medical report parser using LLM
    Extracts structured patient data from unstructured medical reports
    """
    
    def __init__(self, llm_provider: str = "openai", model: str = "gpt-4"):
        """
        Initialize LLM medical parser
        
        Args:
            llm_provider: LLM provider (openai, anthropic, local)
            model: Model name
        """
        self.provider = llm_provider
        self.model = model
        self.llm_client = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        # TODO: Initialize actual LLM client when API keys available
        return None
    
    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Call LLM with prompt"""
        if self.llm_client is None:
            # Fallback to regex-based extraction
            return self._fallback_extraction(prompt)
        
        # TODO: Implement actual LLM API call
        # Example for OpenAI:
        # response = self.llm_client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=temperature
        # )
        # return response.choices[0].message.content
        
        return self._fallback_extraction(prompt)
    
    def _fallback_extraction(self, prompt: str) -> str:
        """Fallback regex-based extraction when LLM unavailable"""
        # Extract report text from prompt
        if "Medical Report:" in prompt:
            report_text = prompt.split("Medical Report:")[1].split("Extract")[0].strip()
        else:
            report_text = prompt
        
        # Use regex patterns
        extracted = {}
        
        # Age
        age_match = re.search(r'(\d+)\s*(?:years old|yo|y/o)', report_text, re.IGNORECASE)
        if age_match:
            extracted['age'] = int(age_match.group(1))
        
        # Sex
        if re.search(r'\bmale\b', report_text, re.IGNORECASE):
            extracted['sex'] = 'M'
        elif re.search(r'\bfemale\b', report_text, re.IGNORECASE):
            extracted['sex'] = 'F'
        
        # BMI
        bmi_match = re.search(r'BMI[:\s]+(\d+\.?\d*)', report_text, re.IGNORECASE)
        if bmi_match:
            extracted['bmi'] = float(bmi_match.group(1))
        
        # Blood pressure
        bp_match = re.search(r'BP[:\s]+(\d+)/(\d+)', report_text, re.IGNORECASE)
        if bp_match:
            extracted['blood_pressure'] = {
                'systolic': int(bp_match.group(1)),
                'diastolic': int(bp_match.group(2))
            }
        
        # Glucose
        glucose_match = re.search(r'glucose[:\s]+(\d+\.?\d*)', report_text, re.IGNORECASE)
        if glucose_match:
            extracted['fasting_glucose'] = float(glucose_match.group(1))
        
        # HbA1c
        hba1c_match = re.search(r'HbA1c[:\s]+(\d+\.?\d*)%?', report_text, re.IGNORECASE)
        if hba1c_match:
            extracted['hba1c'] = float(hba1c_match.group(1))
        
        # Cholesterol
        ldl_match = re.search(r'LDL[:\s]+(\d+\.?\d*)', report_text, re.IGNORECASE)
        if ldl_match:
            extracted['ldl_cholesterol'] = float(ldl_match.group(1))
        
        hdl_match = re.search(r'HDL[:\s]+(\d+\.?\d*)', report_text, re.IGNORECASE)
        if hdl_match:
            extracted['hdl_cholesterol'] = float(hdl_match.group(1))
        
        return json.dumps(extracted, indent=2)
    
    def parse_medical_report(self, report_text: str) -> Dict:
        """
        Parse medical report and extract structured patient data
        
        Args:
            report_text: Raw medical report text
            
        Returns:
            Structured patient data dictionary
        """
        prompt = f"""
You are a medical AI assistant extracting structured data from medical reports.

Medical Report:
{report_text}

Extract the following information and return as JSON:

{{
  "demographics": {{
    "age": <integer>,
    "sex": "M" or "F",
    "ethnicity": <string or null>
  }},
  "physiology": {{
    "bmi": <float>,
    "weight": <float in kg>,
    "height": <float in cm>,
    "fasting_glucose": <float in mg/dL>,
    "hba1c": <float as percentage>,
    "blood_pressure": {{
      "systolic": <integer>,
      "diastolic": <integer>
    }},
    "heart_rate": <integer>,
    "total_cholesterol": <float in mg/dL>,
    "ldl_cholesterol": <float in mg/dL>,
    "hdl_cholesterol": <float in mg/dL>,
    "triglycerides": <float in mg/dL>,
    "alt": <float in U/L>,
    "ast": <float in U/L>,
    "creatinine": <float in mg/dL>,
    "egfr": <float in mL/min>
  }},
  "lifestyle": {{
    "physical_activity": "sedentary" | "light" | "moderate" | "vigorous",
    "diet_quality": "poor" | "fair" | "good" | "excellent",
    "sleep_duration": <float in hours>,
    "stress_level": "low" | "moderate" | "high",
    "smoking_status": "never" | "former" | "current",
    "alcohol_consumption": "none" | "light" | "moderate" | "heavy"
  }},
  "medical_history": {{
    "diagnoses": [<list of conditions>],
    "medications": [<list of medications>],
    "family_history": {{
      "diabetes": <boolean>,
      "heart_disease": <boolean>,
      "cancer": <boolean>
    }}
  }}
}}

Rules:
- Only include fields that are explicitly mentioned in the report
- Use null for missing values
- Ensure all numeric values are in the specified units
- Infer lifestyle factors from context when mentioned
- Extract family history from social/family history sections

Return ONLY the JSON, no additional text.
"""
        
        response = self._call_llm(prompt, temperature=0.3)
        
        try:
            # Parse JSON response
            extracted_data = json.loads(response)
            
            # Flatten structure for compatibility
            patient_data = {}
            
            # Demographics
            if 'demographics' in extracted_data:
                patient_data.update(extracted_data['demographics'])
            
            # Physiology
            if 'physiology' in extracted_data:
                patient_data.update(extracted_data['physiology'])
            
            # Lifestyle
            if 'lifestyle' in extracted_data:
                patient_data['lifestyle'] = extracted_data['lifestyle']
            
            # Medical history
            if 'medical_history' in extracted_data:
                patient_data['medical_history'] = extracted_data['medical_history']
                if 'family_history' in extracted_data['medical_history']:
                    patient_data['family_history'] = extracted_data['medical_history']['family_history']
            
            return patient_data
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract from response
            return self._extract_from_text(response)
    
    def _extract_from_text(self, text: str) -> Dict:
        """Extract data from non-JSON text response"""
        # Fallback extraction
        patient_data = {}
        
        # Try to find JSON in text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        return patient_data
    
    def assess_data_quality(self, patient_data: Dict) -> Dict:
        """
        Assess completeness and quality of extracted data
        
        Args:
            patient_data: Extracted patient data
            
        Returns:
            Quality assessment with completeness score and missing fields
        """
        required_fields = {
            'critical': ['age', 'sex'],
            'important': ['bmi', 'blood_pressure', 'fasting_glucose', 'hba1c'],
            'useful': ['ldl_cholesterol', 'hdl_cholesterol', 'lifestyle']
        }
        
        present = {'critical': 0, 'important': 0, 'useful': 0}
        missing = {'critical': [], 'important': [], 'useful': []}
        
        for category, fields in required_fields.items():
            for field in fields:
                if field in patient_data and patient_data[field] is not None:
                    present[category] += 1
                else:
                    missing[category].append(field)
        
        # Calculate completeness score
        total_fields = sum(len(fields) for fields in required_fields.values())
        present_fields = sum(present.values())
        completeness = (present_fields / total_fields) * 100
        
        return {
            'completeness_score': round(completeness, 1),
            'present_fields': present,
            'missing_fields': missing,
            'quality_level': 'high' if completeness >= 80 else 'medium' if completeness >= 50 else 'low',
            'ready_for_simulation': present['critical'] == len(required_fields['critical'])
        }
    
    def enrich_with_context(self, patient_data: Dict, report_text: str) -> Dict:
        """
        Use LLM to add contextual information and inferences
        
        Args:
            patient_data: Extracted patient data
            report_text: Original report text
            
        Returns:
            Enriched patient data with inferences
        """
        prompt = f"""
You are a medical AI analyzing a patient report.

Extracted Data:
{json.dumps(patient_data, indent=2)}

Original Report:
{report_text}

Based on the report context, infer and add:

1. **Lifestyle factors** (if not explicitly stated):
   - Physical activity level (from occupation, hobbies, complaints)
   - Diet quality (from weight trends, conditions, recommendations)
   - Stress level (from mental health notes, work situation)

2. **Risk context**:
   - What risk factors are present?
   - What patterns suggest future health issues?

3. **Clinical context**:
   - What is the overall health status?
   - What should be monitored?

Return JSON with:
{{
  "inferred_lifestyle": {{...}},
  "risk_context": {{...}},
  "clinical_notes": "..."
}}
"""
        
        response = self._call_llm(prompt, temperature=0.5)
        
        try:
            enrichment = json.loads(response)
            
            # Merge inferred lifestyle
            if 'inferred_lifestyle' in enrichment:
                if 'lifestyle' not in patient_data:
                    patient_data['lifestyle'] = {}
                patient_data['lifestyle'].update(enrichment['inferred_lifestyle'])
            
            # Add context
            patient_data['_context'] = {
                'risk_context': enrichment.get('risk_context', {}),
                'clinical_notes': enrichment.get('clinical_notes', '')
            }
            
        except:
            pass
        
        return patient_data


# Convenience function
def parse_report(report_text: str) -> Dict:
    """Quick parse function"""
    parser = LLMMedicalParser()
    return parser.parse_medical_report(report_text)
