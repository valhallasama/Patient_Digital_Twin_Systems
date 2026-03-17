#!/usr/bin/env python3
"""
LLM Integration Layer for Digital Twin
Provides interpretation, explanation, and recommendations
"""

from typing import Dict, List, Optional
import json
from pathlib import Path


class LLMInterpreter:
    """
    LLM-powered interpretation and communication layer
    Uses LLM for non-numerical tasks: explanation, recommendations, guidelines
    """
    
    def __init__(self, llm_provider: str = "openai", model: str = "gpt-4"):
        """
        Initialize LLM interpreter
        
        Args:
            llm_provider: "openai", "anthropic", "local", etc.
            model: Model name
        """
        self.provider = llm_provider
        self.model = model
        self.llm_client = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client (placeholder for actual implementation)"""
        # TODO: Initialize actual LLM client
        # For now, return None and use template-based responses
        return None
    
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Call LLM with prompt
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            
        Returns:
            LLM response text
        """
        if self.llm_client is None:
            # Fallback: return template-based response
            return self._template_response(prompt)
        
        # TODO: Implement actual LLM API call
        # Example for OpenAI:
        # response = self.llm_client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=temperature
        # )
        # return response.choices[0].message.content
        
        return self._template_response(prompt)
    
    def _template_response(self, prompt: str) -> str:
        """Template-based fallback when LLM not available"""
        if "interpret patient data" in prompt.lower():
            return "Patient profile analyzed. Key risk factors identified."
        elif "explain results" in prompt.lower():
            return "Simulation shows parameter progression over time based on lifestyle factors."
        else:
            return "Analysis complete."
    
    # ========================================================================
    # 1. PATIENT DATA INTERPRETATION
    # ========================================================================
    
    def interpret_patient_data(self, patient_data: Dict) -> Dict:
        """
        Analyze patient data and provide context-aware interpretation
        
        Args:
            patient_data: Raw patient data
            
        Returns:
            Interpretation with risk context, data quality, recommendations
        """
        prompt = f"""
You are a medical AI assistant analyzing patient data for a digital twin simulation.

Patient Data:
{json.dumps(patient_data, indent=2)}

Provide a structured analysis:

1. **Data Completeness** (0-100%):
   - What data is present?
   - What critical data is missing?
   - How will missing data affect simulation accuracy?

2. **Risk Profile Summary**:
   - What are the key risk factors?
   - What is the overall risk level (low/moderate/high)?
   - What are the most concerning findings?

3. **Context & Patterns**:
   - Are there concerning patterns in the data?
   - What lifestyle factors are most problematic?
   - What protective factors are present?

4. **Simulation Expectations**:
   - What parameters are likely to change most?
   - What diseases are most likely to develop?
   - What timeline should we expect?

Format as JSON with keys: data_completeness, risk_profile, context, expectations
"""
        
        response = self._call_llm(prompt, temperature=0.3)
        
        try:
            return json.loads(response)
        except:
            return {
                "data_completeness": "Unable to parse",
                "risk_profile": "Analysis pending",
                "context": response,
                "expectations": "See full analysis"
            }
    
    # ========================================================================
    # 2. RESULTS EXPLANATION
    # ========================================================================
    
    def explain_simulation_results(self, 
                                   patient_data: Dict,
                                   trajectory: List[Dict],
                                   predictions: List[Dict]) -> Dict:
        """
        Generate comprehensive explanation of simulation results
        
        Args:
            patient_data: Original patient data
            trajectory: Parameter evolution over time
            predictions: Disease predictions
            
        Returns:
            Structured explanation with key findings, causes, implications
        """
        # Extract key changes
        initial = trajectory[0] if trajectory else {}
        final = trajectory[-1] if trajectory else {}
        
        prompt = f"""
You are a medical AI explaining digital twin simulation results to a healthcare provider.

Patient: {patient_data.get('age', 'Unknown')} year old {patient_data.get('sex', 'Unknown')}
Lifestyle: {patient_data.get('lifestyle', {})}

Simulation Duration: 2 years

Key Parameter Changes:
{self._format_parameter_changes(initial, final)}

Disease Predictions:
{json.dumps(predictions[:3], indent=2)}

Provide a clear, evidence-based explanation:

1. **What Happened** (2-3 sentences):
   - Summarize the key findings
   - What parameters changed most significantly?

2. **Why It Happened** (3-4 bullet points):
   - What lifestyle factors drove these changes?
   - What physiological mechanisms are involved?
   - How did different organ systems interact?

3. **Clinical Significance**:
   - What do these changes mean for patient health?
   - What diseases are most likely and when?
   - What is the urgency level?

4. **Key Insights**:
   - What was surprising or noteworthy?
   - What patterns emerged?
   - What should clinicians focus on?

Format as JSON with keys: summary, causes, significance, insights
"""
        
        response = self._call_llm(prompt, temperature=0.5)
        
        try:
            return json.loads(response)
        except:
            return {
                "summary": "Simulation completed successfully",
                "causes": ["Lifestyle factors", "Age-related changes", "Metabolic interactions"],
                "significance": response,
                "insights": "See detailed analysis"
            }
    
    def _format_parameter_changes(self, initial: Dict, final: Dict) -> str:
        """Format parameter changes for LLM prompt"""
        changes = []
        
        if 'agents' in initial and 'agents' in final:
            # Metabolic
            if 'metabolic' in initial['agents'] and 'metabolic' in final['agents']:
                init_m = initial['agents']['metabolic']
                final_m = final['agents']['metabolic']
                changes.append(f"- HbA1c: {init_m.get('hba1c', 0):.2f}% → {final_m.get('hba1c', 0):.2f}%")
                changes.append(f"- Glucose: {init_m.get('glucose', 0):.0f} → {final_m.get('glucose', 0):.0f} mg/dL")
            
            # Cardiovascular
            if 'cardiovascular' in initial['agents'] and 'cardiovascular' in final['agents']:
                init_c = initial['agents']['cardiovascular']
                final_c = final['agents']['cardiovascular']
                changes.append(f"- Blood Pressure: {init_c.get('systolic_bp', 0):.0f}/{init_c.get('diastolic_bp', 0):.0f} → {final_c.get('systolic_bp', 0):.0f}/{final_c.get('diastolic_bp', 0):.0f} mmHg")
                changes.append(f"- LDL: {init_c.get('ldl', 0):.0f} → {final_c.get('ldl', 0):.0f} mg/dL")
        
        return "\n".join(changes) if changes else "No significant changes"
    
    # ========================================================================
    # 3. PERSONALIZED RECOMMENDATIONS
    # ========================================================================
    
    def generate_recommendations(self,
                                patient_data: Dict,
                                predictions: List[Dict],
                                patient_context: Optional[Dict] = None) -> Dict:
        """
        Generate personalized, actionable recommendations
        
        Args:
            patient_data: Patient profile
            predictions: Disease predictions
            patient_context: Additional context (barriers, preferences, etc.)
            
        Returns:
            Structured recommendations with priorities and timelines
        """
        context_str = json.dumps(patient_context, indent=2) if patient_context else "Not provided"
        
        prompt = f"""
You are a medical AI creating personalized health recommendations.

Patient Profile:
- Age: {patient_data.get('age', 'Unknown')}
- Lifestyle: {patient_data.get('lifestyle', {})}
- Predicted Risks: {[p['disease'] for p in predictions[:3]]}

Patient Context:
{context_str}

Create a realistic, evidence-based action plan:

1. **Immediate Actions** (Next 2 weeks):
   - 3 specific, achievable actions
   - Why each is important
   - How to implement

2. **Short-term Goals** (1-3 months):
   - 3-4 progressive goals
   - Expected impact on each risk
   - Milestones to track

3. **Long-term Strategy** (6-12 months):
   - Sustainable lifestyle changes
   - Maintenance approach
   - Monitoring plan

4. **Prioritization**:
   - Rank recommendations by:
     * Impact (risk reduction)
     * Feasibility (ease of implementation)
     * Cost (financial burden)

5. **Personalization Notes**:
   - How to adapt to patient's specific situation
   - Potential barriers and solutions
   - Motivation strategies

Format as JSON with keys: immediate, short_term, long_term, priorities, personalization
"""
        
        response = self._call_llm(prompt, temperature=0.6)
        
        try:
            return json.loads(response)
        except:
            return {
                "immediate": ["Improve diet quality", "Increase physical activity", "Reduce stress"],
                "short_term": ["Establish exercise routine", "Dietary changes", "Regular monitoring"],
                "long_term": ["Maintain healthy lifestyle", "Prevent disease progression"],
                "priorities": response,
                "personalization": "Adapt to individual circumstances"
            }
    
    # ========================================================================
    # 4. CLINICAL GUIDELINE INTEGRATION
    # ========================================================================
    
    def get_clinical_guidelines(self,
                               patient_data: Dict,
                               predictions: List[Dict]) -> Dict:
        """
        Retrieve relevant clinical guidelines and recommendations
        
        Args:
            patient_data: Patient profile
            predictions: Disease predictions
            
        Returns:
            Clinical guidelines with treatment targets, monitoring, referrals
        """
        top_risks = [p['disease'] for p in predictions[:3]]
        
        prompt = f"""
You are a medical AI providing evidence-based clinical guidelines.

Patient: {patient_data.get('age', 'Unknown')}yo, {patient_data.get('sex', 'Unknown')}
Top Predicted Risks: {', '.join(top_risks)}

Current Parameters:
- HbA1c: {patient_data.get('hba1c', 'Unknown')}%
- BP: {patient_data.get('blood_pressure', {}).get('systolic', 'Unknown')}/{patient_data.get('blood_pressure', {}).get('diastolic', 'Unknown')} mmHg
- LDL: {patient_data.get('ldl_cholesterol', 'Unknown')} mg/dL

Based on current ADA, AHA, and other major guidelines (2024-2026):

1. **Treatment Targets**:
   - What are the recommended target values?
   - What thresholds trigger intervention?
   - What are acceptable ranges?

2. **Recommended Interventions**:
   - Lifestyle modifications (specific)
   - Medications (if indicated)
   - Other therapies

3. **Monitoring Plan**:
   - What parameters to monitor?
   - How frequently?
   - What tests are needed?

4. **Referral Criteria**:
   - When to refer to specialists?
   - What specialists are needed?
   - Urgency level?

5. **Prevention Strategies**:
   - Primary prevention (if no disease)
   - Secondary prevention (if disease present)
   - Evidence level for each recommendation

Format as JSON with keys: targets, interventions, monitoring, referrals, prevention
"""
        
        response = self._call_llm(prompt, temperature=0.3)
        
        try:
            return json.loads(response)
        except:
            return {
                "targets": "Standard clinical targets apply",
                "interventions": ["Lifestyle modification", "Consider pharmacotherapy"],
                "monitoring": "Regular follow-up recommended",
                "referrals": response,
                "prevention": "Evidence-based prevention strategies"
            }
    
    # ========================================================================
    # 5. PATIENT-FRIENDLY COMMUNICATION
    # ========================================================================
    
    def generate_patient_report(self,
                               patient_data: Dict,
                               predictions: List[Dict],
                               recommendations: Dict,
                               literacy_level: str = "general") -> str:
        """
        Generate patient-friendly report
        
        Args:
            patient_data: Patient profile
            predictions: Disease predictions
            recommendations: Personalized recommendations
            literacy_level: "simple", "general", or "advanced"
            
        Returns:
            Patient-friendly report text
        """
        literacy_instructions = {
            "simple": "Use 6th grade reading level. Short sentences. No medical jargon.",
            "general": "Use clear language. Explain medical terms when used.",
            "advanced": "Can use medical terminology with explanations."
        }
        
        prompt = f"""
You are creating a health report for a patient.

Writing Level: {literacy_instructions.get(literacy_level, literacy_instructions['general'])}

Patient: {patient_data.get('age', 'Unknown')} years old
Simulation Results: {len(predictions)} health risks identified
Top Concerns: {[p['disease'] for p in predictions[:2]]}

Create a compassionate, empowering report:

1. **Your Health Picture** (2-3 paragraphs):
   - What the simulation showed
   - What this means for you
   - Why this matters

2. **What You Can Do** (Clear action steps):
   - Top 3 actions you can start today
   - Why each will help
   - How to get started

3. **Your Timeline**:
   - What to expect in the next few months
   - When you might see improvements
   - Important milestones

4. **Questions to Ask Your Doctor**:
   - 3-4 specific questions based on your results
   - What tests or monitoring you might need

5. **Staying Motivated**:
   - Why these changes are worth it
   - How to track your progress
   - Where to get support

Tone: Encouraging, honest, actionable. Focus on what patient CAN control.
"""
        
        response = self._call_llm(prompt, temperature=0.7)
        return response
    
    # ========================================================================
    # 6. RISK EXPLANATION
    # ========================================================================
    
    def explain_specific_risk(self,
                             disease: str,
                             prediction: Dict,
                             patient_data: Dict) -> str:
        """
        Explain a specific disease risk in detail
        
        Args:
            disease: Disease name
            prediction: Prediction details
            patient_data: Patient profile
            
        Returns:
            Detailed explanation
        """
        prompt = f"""
Explain this health risk to a patient in clear, understandable terms:

Disease: {disease}
Risk Level: {prediction.get('probability', 0) * 100:.0f}%
Time Frame: {prediction.get('time_to_onset_days', 'Unknown')} days
Current Status: {prediction.get('status', 'Unknown')}

Patient Profile:
- Age: {patient_data.get('age', 'Unknown')}
- Lifestyle: {patient_data.get('lifestyle', {})}

Explain:
1. What is {disease}? (1-2 sentences, simple language)
2. Why are you at risk? (Specific to this patient)
3. What does {prediction.get('probability', 0) * 100:.0f}% risk mean?
4. What happens if you don't make changes?
5. What happens if you DO make changes?
6. What are the most important actions to take?

Use analogies and examples. Be encouraging but honest.
"""
        
        response = self._call_llm(prompt, temperature=0.6)
        return response


# Convenience function for quick integration
def create_llm_interpreter(provider: str = "openai", model: str = "gpt-4") -> LLMInterpreter:
    """Create LLM interpreter instance"""
    return LLMInterpreter(provider, model)
