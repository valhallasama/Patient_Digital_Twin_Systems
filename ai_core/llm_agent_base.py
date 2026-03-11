"""
LLM-Powered Medical Agent Base
True AI reasoning agents, not just rule-based calculators
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMAgentBase(ABC):
    """
    Base class for LLM-powered medical specialist agents
    Each agent uses LLM for reasoning, not just formulas
    """
    
    def __init__(
        self,
        name: str,
        specialty: str,
        model_provider: str = "openai",
        model_name: str = "gpt-4",
        api_key: Optional[str] = None
    ):
        self.name = name
        self.specialty = specialty
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize LLM client
        self.use_llm = False
        if model_provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                self.use_llm = True
                logger.info(f"✓ {name} initialized with {model_name}")
            except ImportError:
                logger.warning(f"{name}: OpenAI not installed, using fallback")
        elif model_provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                self.use_llm = True
                logger.info(f"✓ {name} initialized with {model_name}")
            except ImportError:
                logger.warning(f"{name}: Anthropic not installed, using fallback")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Define the agent's role and expertise"""
        pass
    
    @abstractmethod
    def create_analysis_prompt(self, patient_data: Dict) -> str:
        """Create prompt for analyzing patient"""
        pass
    
    @abstractmethod
    def fallback_analysis(self, patient_data: Dict) -> Dict:
        """Fallback rule-based analysis if LLM unavailable"""
        pass
    
    def analyze_patient(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Analyze patient using LLM reasoning
        Returns structured assessment with explanations
        """
        
        if self.use_llm:
            return self.llm_analysis(patient_data)
        else:
            return self.fallback_analysis(patient_data)
    
    def llm_analysis(self, patient_data: Dict) -> Dict[str, Any]:
        """Use LLM for deep medical reasoning"""
        
        system_prompt = self.get_system_prompt()
        analysis_prompt = self.create_analysis_prompt(patient_data)
        
        try:
            if self.model_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                result = response.choices[0].message.content
            elif self.model_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": analysis_prompt}]
                )
                result = response.content[0].text
            else:
                return self.fallback_analysis(patient_data)
            
            # Parse JSON response
            analysis = json.loads(result)
            analysis['agent'] = self.name
            analysis['reasoning_method'] = 'llm'
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.name} LLM analysis failed: {e}")
            logger.info(f"{self.name} using fallback analysis")
            return self.fallback_analysis(patient_data)
    
    def communicate(self, message: str, other_agents: List['LLMAgentBase']) -> List[Dict]:
        """
        Inter-agent communication
        Agents can discuss and reason together
        """
        
        if not self.use_llm:
            return []
        
        responses = []
        for agent in other_agents:
            if agent.name != self.name:
                prompt = f"""
                You are {self.name}, a {self.specialty} specialist.
                
                Another specialist ({agent.name}) has shared this observation:
                {message}
                
                Respond with your perspective as a {self.specialty} specialist.
                How does this relate to your domain?
                Do you agree or have additional insights?
                
                Provide a brief, focused response (2-3 sentences).
                """
                
                try:
                    if self.model_provider == "openai":
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": self.get_system_prompt()},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.5,
                            max_tokens=200
                        )
                        reply = response.choices[0].message.content
                        responses.append({
                            'from': self.name,
                            'to': agent.name,
                            'message': reply
                        })
                except Exception as e:
                    logger.error(f"Communication failed: {e}")
        
        return responses


class CardiologyLLMAgent(LLMAgentBase):
    """LLM-powered Cardiology specialist"""
    
    def __init__(self, model_provider="openai", model_name="gpt-4", api_key=None):
        super().__init__(
            name="Cardiology AI Agent",
            specialty="Cardiology",
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert Cardiology AI agent with deep knowledge of:
- Cardiovascular disease pathophysiology
- Risk assessment (Framingham, ASCVD, QRISK)
- Atherosclerosis and plaque formation
- Heart failure mechanisms
- Arrhythmia patterns
- Coronary artery disease
- Hypertension management
- Lipid disorders

Your role is to:
1. Assess cardiovascular risk comprehensively
2. Identify concerning patterns in patient data
3. Explain mechanisms and pathways
4. Recommend evidence-based interventions
5. Predict disease progression
6. Consider interactions with other conditions

Provide structured, evidence-based analysis with clear reasoning."""
    
    def create_analysis_prompt(self, patient_data: Dict) -> str:
        return f"""Analyze this patient's cardiovascular risk:

Patient Data:
{json.dumps(patient_data, indent=2)}

Provide comprehensive cardiovascular assessment in JSON format:

{{
  "risk_assessment": {{
    "overall_cvd_risk": "<low/moderate/high/very_high>",
    "10_year_risk_percentage": <float>,
    "risk_category_explanation": "<string>"
  }},
  "risk_factors": {{
    "modifiable": [
      {{"factor": "<string>", "severity": "<low/moderate/high>", "impact": "<string>"}}
    ],
    "non_modifiable": [
      {{"factor": "<string>", "impact": "<string>"}}
    ]
  }},
  "pathophysiology": {{
    "mechanisms": ["<list of active disease mechanisms>"],
    "progression_pathway": "<explanation of how disease develops>",
    "timeline": "<expected progression timeline>"
  }},
  "concerning_findings": [
    {{"finding": "<string>", "significance": "<string>", "action_needed": "<string>"}}
  ],
  "protective_factors": [
    "<list of factors reducing risk>"
  ],
  "recommendations": [
    {{
      "intervention": "<string>",
      "rationale": "<string>",
      "expected_benefit": "<string>",
      "priority": "<high/medium/low>"
    }}
  ],
  "monitoring": {{
    "parameters": ["<parameters to monitor>"],
    "frequency": "<monitoring frequency>",
    "targets": {{"<parameter>": "<target value>"}}
  }},
  "prognosis": {{
    "short_term": "<1-2 year outlook>",
    "long_term": "<10 year outlook>",
    "key_determinants": ["<factors affecting prognosis>"]
  }},
  "reasoning": "<detailed explanation of your analysis>"
}}

Be thorough, evidence-based, and explain your reasoning clearly."""
    
    def fallback_analysis(self, patient_data: Dict) -> Dict:
        """Simple rule-based fallback"""
        age = patient_data.get('demographics', {}).get('age', 50)
        systolic_bp = patient_data.get('vitals', {}).get('systolic_bp', 120)
        ldl = patient_data.get('labs', {}).get('ldl_cholesterol_mmol_l', 3.0)
        smoking = patient_data.get('lifestyle', {}).get('smoking_status', 'never')
        
        # Simple risk calculation
        risk_score = 0
        if age > 60: risk_score += 3
        elif age > 45: risk_score += 2
        if systolic_bp > 140: risk_score += 2
        if ldl > 4.0: risk_score += 2
        if smoking == 'current': risk_score += 3
        
        risk_level = 'low' if risk_score < 3 else 'moderate' if risk_score < 6 else 'high'
        
        return {
            'agent': self.name,
            'reasoning_method': 'rule_based_fallback',
            'risk_assessment': {
                'overall_cvd_risk': risk_level,
                '10_year_risk_percentage': min(risk_score * 5, 40),
                'risk_category_explanation': f'Risk score: {risk_score}/10'
            },
            'recommendations': [
                {'intervention': 'Lifestyle modification', 'priority': 'high'},
                {'intervention': 'Regular monitoring', 'priority': 'medium'}
            ],
            'reasoning': 'Fallback rule-based analysis (LLM unavailable)'
        }


class EndocrinologyLLMAgent(LLMAgentBase):
    """LLM-powered Endocrinology specialist"""
    
    def __init__(self, model_provider="openai", model_name="gpt-4", api_key=None):
        super().__init__(
            name="Endocrinology AI Agent",
            specialty="Endocrinology",
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert Endocrinology AI agent specializing in:
- Diabetes mellitus (Type 1, Type 2, prediabetes)
- Metabolic syndrome
- Insulin resistance and beta-cell dysfunction
- Thyroid disorders
- Lipid metabolism
- Hormonal imbalances
- Obesity and weight management

Your expertise includes:
1. Glucose homeostasis and dysregulation
2. HbA1c interpretation and trends
3. Insulin sensitivity assessment
4. Metabolic risk stratification
5. Diabetes complications (micro and macrovascular)
6. Pharmacological interventions
7. Lifestyle medicine for metabolic health

Provide evidence-based endocrine assessment with mechanistic explanations."""
    
    def create_analysis_prompt(self, patient_data: Dict) -> str:
        return f"""Analyze this patient's metabolic and endocrine status:

Patient Data:
{json.dumps(patient_data, indent=2)}

Provide comprehensive endocrine assessment in JSON format:

{{
  "metabolic_status": {{
    "diabetes_status": "<none/prediabetes/type2_diabetes>",
    "metabolic_syndrome": <bool>,
    "insulin_resistance_estimate": "<low/moderate/high>",
    "beta_cell_function_estimate": "<normal/impaired/severely_impaired>"
  }},
  "glucose_homeostasis": {{
    "fasting_glucose_interpretation": "<string>",
    "hba1c_interpretation": "<string>",
    "glycemic_control": "<excellent/good/fair/poor>",
    "trajectory": "<improving/stable/worsening>"
  }},
  "risk_assessment": {{
    "diabetes_risk": "<low/moderate/high/very_high>",
    "progression_risk": "<string>",
    "complications_risk": {{
      "retinopathy": "<low/moderate/high>",
      "nephropathy": "<low/moderate/high>",
      "neuropathy": "<low/moderate/high>",
      "cardiovascular": "<low/moderate/high>"
    }}
  }},
  "pathophysiology": {{
    "primary_mechanisms": ["<list mechanisms>"],
    "contributing_factors": ["<list factors>"],
    "disease_stage": "<string>"
  }},
  "recommendations": [
    {{
      "intervention": "<string>",
      "rationale": "<string>",
      "expected_benefit": "<string>",
      "priority": "<high/medium/low>"
    }}
  ]],
  "monitoring": {{
    "parameters": ["<parameters to monitor>"],
    "frequency": "<string>",
    "targets": {{"<parameter>": "<target>"}}
  }},
  "reasoning": "<detailed explanation>"
}}"""
    
    def fallback_analysis(self, patient_data: Dict) -> Dict:
        glucose = patient_data.get('labs', {}).get('glucose_mmol_l', 5.0)
        hba1c = patient_data.get('labs', {}).get('hba1c_percent', 5.5)
        bmi = patient_data.get('physical', {}).get('bmi', 25)
        
        diabetes_status = 'none'
        if hba1c >= 6.5 or glucose >= 7.0:
            diabetes_status = 'type2_diabetes'
        elif hba1c >= 5.7 or glucose >= 5.6:
            diabetes_status = 'prediabetes'
        
        return {
            'agent': self.name,
            'reasoning_method': 'rule_based_fallback',
            'metabolic_status': {
                'diabetes_status': diabetes_status,
                'metabolic_syndrome': bmi > 30 and hba1c > 5.7
            },
            'recommendations': [
                {'intervention': 'Glucose monitoring', 'priority': 'high'},
                {'intervention': 'Dietary modification', 'priority': 'high'}
            ],
            'reasoning': 'Fallback rule-based analysis (LLM unavailable)'
        }


class MultiAgentLLMSystem:
    """
    Multi-agent system with LLM-powered reasoning
    Agents collaborate and discuss patient cases
    """
    
    def __init__(self, model_provider="openai", model_name="gpt-4", api_key=None):
        self.agents = []
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        self.communication_board = []
    
    def register_agent(self, agent: LLMAgentBase):
        """Add specialist agent to the system"""
        self.agents.append(agent)
        logger.info(f"✓ Registered: {agent.name}")
    
    def analyze_patient(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Multi-agent collaborative analysis
        Each agent analyzes, then they discuss findings
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"MULTI-AGENT ANALYSIS: {len(self.agents)} specialists")
        logger.info(f"{'='*80}")
        
        # Step 1: Individual agent analyses
        agent_analyses = {}
        for agent in self.agents:
            logger.info(f"\n{agent.name} analyzing...")
            analysis = agent.analyze_patient(patient_data)
            agent_analyses[agent.name] = analysis
            logger.info(f"✓ {agent.name} complete")
        
        # Step 2: Inter-agent communication (if LLM available)
        discussions = []
        if any(agent.use_llm for agent in self.agents):
            logger.info("\nInter-agent discussion...")
            for agent in self.agents:
                if agent.use_llm:
                    # Agent shares key finding
                    key_finding = agent_analyses[agent.name].get('reasoning', '')[:200]
                    responses = agent.communicate(key_finding, self.agents)
                    discussions.extend(responses)
        
        # Step 3: Consensus building
        consensus = self.build_consensus(agent_analyses)
        
        return {
            'patient_data': patient_data,
            'agent_analyses': agent_analyses,
            'inter_agent_discussions': discussions,
            'consensus': consensus,
            'system_recommendation': self.generate_system_recommendation(agent_analyses, consensus)
        }
    
    def build_consensus(self, agent_analyses: Dict) -> Dict:
        """Build consensus from multiple agent opinions"""
        
        # Aggregate risk assessments
        risk_scores = []
        for analysis in agent_analyses.values():
            risk_assessment = analysis.get('risk_assessment', {})
            risk_level = risk_assessment.get('overall_cvd_risk') or risk_assessment.get('diabetes_risk', 'moderate')
            
            risk_map = {'low': 1, 'moderate': 2, 'high': 3, 'very_high': 4}
            risk_scores.append(risk_map.get(risk_level, 2))
        
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 2
        consensus_risk = 'low' if avg_risk < 1.5 else 'moderate' if avg_risk < 2.5 else 'high' if avg_risk < 3.5 else 'very_high'
        
        # Collect all recommendations
        all_recommendations = []
        for analysis in agent_analyses.values():
            recs = analysis.get('recommendations', [])
            all_recommendations.extend(recs)
        
        return {
            'consensus_risk_level': consensus_risk,
            'average_risk_score': avg_risk,
            'agreement_level': 'high' if max(risk_scores) - min(risk_scores) <= 1 else 'moderate',
            'combined_recommendations': all_recommendations[:10]  # Top 10
        }
    
    def generate_system_recommendation(self, agent_analyses: Dict, consensus: Dict) -> Dict:
        """Generate final system recommendation"""
        
        return {
            'overall_assessment': f"Multi-agent consensus: {consensus['consensus_risk_level']} risk",
            'confidence': consensus['agreement_level'],
            'priority_actions': [rec for rec in consensus['combined_recommendations'] if rec.get('priority') == 'high'][:5],
            'specialist_count': len(self.agents),
            'reasoning_method': 'multi_agent_llm_collaboration'
        }


# Example usage
if __name__ == "__main__":
    # Sample patient data
    patient = {
        'demographics': {'age': 52, 'gender': 'male'},
        'physical': {'bmi': 31},
        'vitals': {'systolic_bp': 145, 'diastolic_bp': 92},
        'labs': {
            'glucose_mmol_l': 6.2,
            'hba1c_percent': 5.9,
            'ldl_cholesterol_mmol_l': 4.1
        },
        'lifestyle': {
            'smoking_status': 'current',
            'smoking_pack_years': 20,
            'exercise_hours_per_week': 1,
            'sleep_hours_per_night': 5
        },
        'family_history': {'father_cvd_age': 58}
    }
    
    # Initialize multi-agent system
    system = MultiAgentLLMSystem(model_provider="openai", api_key=None)
    
    # Register specialist agents
    system.register_agent(CardiologyLLMAgent(model_provider="openai", api_key=None))
    system.register_agent(EndocrinologyLLMAgent(model_provider="openai", api_key=None))
    
    # Analyze patient
    result = system.analyze_patient(patient)
    
    print("\n" + "="*80)
    print("MULTI-AGENT ANALYSIS RESULT")
    print("="*80)
    print(json.dumps(result['consensus'], indent=2))
    print("\nSystem Recommendation:")
    print(json.dumps(result['system_recommendation'], indent=2))
