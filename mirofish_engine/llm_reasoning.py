"""
LLM-Powered Agent Reasoning
Integrates GPT-4/Claude for intelligent agent decision-making
"""

import os
from typing import Dict, List, Any, Optional
import json


class LLMReasoningEngine:
    """
    Uses LLM (GPT-4/Claude) for agent reasoning
    Replaces simple rule-based logic with intelligent decision-making
    """
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.use_llm = self.api_key is not None
        
        if not self.use_llm:
            print("⚠️  No LLM API key found. Using rule-based reasoning.")
            print("   Set OPENAI_API_KEY environment variable to enable GPT-4 reasoning.")
    
    def agent_decide(
        self,
        agent_name: str,
        current_state: Dict[str, Any],
        perceptions: Dict[str, Any],
        memory: List[Any],
        personality: Dict[str, Any],
        medical_knowledge: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to make intelligent decisions for agent
        
        Args:
            agent_name: Name of the agent (e.g., "Metabolic")
            current_state: Current physiological state
            perceptions: What the agent perceives from environment
            memory: Recent health history
            personality: Agent personality traits
            medical_knowledge: Relevant medical knowledge from graph
        
        Returns:
            Decision with actions and signals to send
        """
        
        if not self.use_llm:
            return self._fallback_reasoning(agent_name, current_state, perceptions)
        
        try:
            import openai
            openai.api_key = self.api_key
            
            # Construct prompt for LLM
            prompt = self._build_reasoning_prompt(
                agent_name, current_state, perceptions, memory, personality, medical_knowledge
            )
            
            # Call LLM
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(agent_name)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Parse LLM response
            decision_text = response.choices[0].message.content
            decision = self._parse_llm_decision(decision_text)
            
            return decision
            
        except Exception as e:
            print(f"⚠️  LLM reasoning failed: {e}. Using fallback.")
            return self._fallback_reasoning(agent_name, current_state, perceptions)
    
    def _get_system_prompt(self, agent_name: str) -> str:
        """Get system prompt for specific agent type"""
        
        prompts = {
            "Metabolic": """You are the Metabolic System Agent (pancreas/liver) in a patient's body.
Your role is to regulate glucose, insulin, and energy metabolism.
You must decide how to respond to current conditions and what signals to send to other organs.
Consider: glucose levels, insulin sensitivity, beta cell function, stress hormones.
Output your decision as JSON with: action, intensity, signals_to_send, reasoning.""",
            
            "Cardiovascular": """You are the Cardiovascular System Agent (heart/vessels) in a patient's body.
Your role is to pump blood, regulate pressure, and maintain vessel health.
You must decide how to respond to current conditions and what signals to send to other organs.
Consider: blood pressure, glucose damage, inflammation, lipid levels.
Output your decision as JSON with: action, intensity, signals_to_send, reasoning.""",
            
            "Immune": """You are the Immune System Agent in a patient's body.
Your role is to detect damage, manage inflammation, and protect against threats.
You must decide how to respond to current conditions and what signals to send to other organs.
Consider: vessel damage, metabolic stress, chronic inflammation.
Output your decision as JSON with: action, intensity, signals_to_send, reasoning.""",
            
            "Endocrine": """You are the Endocrine System Agent (hormonal regulation) in a patient's body.
Your role is to release hormones in response to stress and metabolic needs.
You must decide how to respond to current conditions and what signals to send to other organs.
Consider: stress levels, sleep quality, metabolic demands.
Output your decision as JSON with: action, intensity, signals_to_send, reasoning.""",
            
            "Neural": """You are the Neural System Agent (brain/nervous system) in a patient's body.
Your role is to process stress, manage sleep, and coordinate body responses.
You must decide how to respond to current conditions and what signals to send to other organs.
Consider: lifestyle stress, sleep quality, cognitive load.
Output your decision as JSON with: action, intensity, signals_to_send, reasoning.""",
            
            "Renal": """You are the Renal System Agent (kidneys) in a patient's body.
Your role is to filter blood, regulate electrolytes, and maintain kidney function.
You must decide how to respond to current conditions and what signals to send to other organs.
Consider: blood pressure, glucose levels, filtration capacity.
Output your decision as JSON with: action, intensity, signals_to_send, reasoning.""",
            
            "Hepatic": """You are the Hepatic System Agent (liver) in a patient's body.
Your role is to metabolize nutrients, produce proteins, and manage lipids.
You must decide how to respond to current conditions and what signals to send to other organs.
Consider: fat intake, inflammation, lipid levels.
Output your decision as JSON with: action, intensity, signals_to_send, reasoning."""
        }
        
        return prompts.get(agent_name, "You are a body system agent. Make intelligent decisions.")
    
    def _build_reasoning_prompt(
        self,
        agent_name: str,
        current_state: Dict[str, Any],
        perceptions: Dict[str, Any],
        memory: List[Any],
        personality: Dict[str, Any],
        medical_knowledge: Optional[Dict[str, Any]]
    ) -> str:
        """Build detailed prompt for LLM reasoning"""
        
        prompt = f"""Current Situation for {agent_name} Agent:

MY CURRENT STATE:
{json.dumps(current_state, indent=2)}

WHAT I PERCEIVE:
Environment signals: {json.dumps(perceptions.get('signals_for_me', {}), indent=2)}
Other agents status: {perceptions.get('other_agents_state', {})}

MY PERSONALITY:
{json.dumps(personality, indent=2)}

RECENT MEMORY (last 3 events):
"""
        
        for mem in memory[-3:] if memory else []:
            if hasattr(mem, 'event'):
                prompt += f"- {mem.event} (impact: {mem.impact:+.2f})\n"
        
        if medical_knowledge:
            prompt += f"\nRELEVANT MEDICAL KNOWLEDGE:\n{json.dumps(medical_knowledge, indent=2)}\n"
        
        prompt += """
Based on this information, decide what action to take and what signals to send to other organs.

Consider:
1. Is my current state healthy or stressed?
2. What signals from other organs require my response?
3. What actions should I take to maintain homeostasis?
4. What signals should I send to other organs?
5. How does my personality affect my response?

Output ONLY valid JSON in this format:
{
  "action": "maintain|compensate|emergency_response",
  "intensity": 1.0,
  "signals_to_send": {
    "signal_name": value,
    ...
  },
  "reasoning": "Brief explanation of decision"
}
"""
        
        return prompt
    
    def _parse_llm_decision(self, decision_text: str) -> Dict[str, Any]:
        """Parse LLM response into decision dict"""
        try:
            # Extract JSON from response
            start = decision_text.find('{')
            end = decision_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = decision_text[start:end]
                decision = json.loads(json_str)
                return decision
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"⚠️  Failed to parse LLM decision: {e}")
            return {
                'action': 'maintain',
                'intensity': 1.0,
                'signals_to_send': {},
                'reasoning': 'Parse error - using default'
            }
    
    def _fallback_reasoning(
        self,
        agent_name: str,
        current_state: Dict[str, Any],
        perceptions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback to rule-based reasoning when LLM unavailable"""
        
        signals = perceptions.get('signals_for_me', {})
        decision = {
            'action': 'maintain',
            'intensity': 1.0,
            'signals_to_send': {},
            'reasoning': 'Rule-based fallback'
        }
        
        # Simple rule-based logic per agent
        if agent_name == "Metabolic":
            glucose = signals.get('glucose', current_state.get('glucose', 5.0))
            if glucose > 7.0:
                decision['action'] = 'emergency_response'
                decision['signals_to_send']['high_glucose_alert'] = True
            elif glucose > 6.0:
                decision['action'] = 'compensate'
        
        elif agent_name == "Cardiovascular":
            bp = current_state.get('systolic_bp', 120)
            if bp > 140:
                decision['action'] = 'compensate'
                decision['signals_to_send']['elevated_bp'] = bp
        
        return decision
    
    def explain_disease_emergence(
        self,
        disease_name: str,
        causative_agents: List[str],
        timeline_events: List[Dict[str, Any]],
        final_states: Dict[str, Any]
    ) -> str:
        """
        Use LLM to explain how disease emerged from agent interactions
        This provides human-readable causal narrative
        """
        
        if not self.use_llm:
            return f"{disease_name} emerged from interactions between {', '.join(causative_agents)}."
        
        try:
            import openai
            openai.api_key = self.api_key
            
            prompt = f"""Explain how {disease_name} emerged in this patient's digital twin simulation.

CAUSATIVE AGENTS: {', '.join(causative_agents)}

KEY EVENTS IN TIMELINE:
"""
            for event in timeline_events[-10:]:
                prompt += f"Day {event.get('day', 0)}: {event.get('event', 'Unknown event')}\n"
            
            prompt += f"\nFINAL AGENT STATES:\n{json.dumps(final_states, indent=2)}\n"
            prompt += "\nProvide a clear, medical explanation of the causal pathway that led to this disease."
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical AI explaining disease pathophysiology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"LLM explanation failed: {e}"


# Global instance
_llm_engine = None

def get_llm_engine() -> LLMReasoningEngine:
    """Get or create global LLM reasoning engine"""
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMReasoningEngine()
    return _llm_engine
