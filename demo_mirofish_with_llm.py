#!/usr/bin/env python3
"""
MiroFish-Inspired Patient Digital Twin with LLM Reasoning
Demonstrates GPT-4 powered agent decision-making
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mirofish_engine.parallel_digital_patient import ParallelDigitalPatient
import json


def extract_seed_with_worse_prognosis(report_text: str) -> dict:
    """
    Extract seed information - this time WITHOUT interventions
    to show realistic disease emergence
    """
    seed_information = {
        'patient_id': 'DT-SIM-0426',
        'initial_composition': {
            'glucose': 5.8,
            'cortisol': 1.3,  # Elevated stress
            'ldl': 3.6,
            'hdl': 1.1,
            'inflammation_markers': 0.18
        },
        'agent_seeds': {
            'cardiovascular': {
                'initial_state': {
                    'systolic_bp': 132,
                    'diastolic_bp': 86,
                    'heart_rate': 72,
                    'cardiac_output': 5.0,
                    'vessel_elasticity': 0.88,  # Slightly worse
                    'atherosclerosis_level': 0.12  # Starting damage
                },
                'resilience': 0.5,  # Lower resilience
                'reactivity': 0.6,
                'adaptability': 0.4,
                'cooperation': 0.7
            },
            'metabolic': {
                'initial_state': {
                    'glucose': 5.8,
                    'insulin': 14.0,  # Higher baseline
                    'hba1c': 5.7,
                    'insulin_sensitivity': 0.70,  # Worse
                    'beta_cell_function': 0.85,  # Already declining
                    'insulin_resistance': 0.30  # Higher
                },
                'resilience': 0.4,  # Low resilience
                'reactivity': 0.7,
                'adaptability': 0.3,
                'cooperation': 0.6
            },
            'renal': {
                'initial_state': {
                    'egfr': 95,  # Slightly reduced
                    'creatinine': 92,
                    'filtration_capacity': 0.95,
                    'damage_level': 0.05
                },
                'resilience': 0.6,
                'reactivity': 0.5,
                'adaptability': 0.5,
                'cooperation': 0.7
            },
            'hepatic': {
                'initial_state': {
                    'alt': 42,
                    'ast': 30,
                    'ldl': 3.6,
                    'hdl': 1.1,
                    'fat_content': 0.08,  # More fatty liver
                    'detox_capacity': 0.92
                },
                'resilience': 0.5,
                'reactivity': 0.6,
                'adaptability': 0.4,
                'cooperation': 0.6
            },
            'immune': {
                'initial_state': {
                    'wbc': 6.8,
                    'crp': 2.2,  # Higher inflammation
                    'inflammation': 0.22,
                    'immune_activation': 0.25
                },
                'resilience': 0.5,
                'reactivity': 0.8,  # Very reactive
                'adaptability': 0.4,
                'cooperation': 0.5
            },
            'endocrine': {
                'initial_state': {
                    'cortisol': 1.3,  # Elevated
                    'thyroid': 1.0,
                    'stress_response': 0.4
                },
                'resilience': 0.4,
                'reactivity': 0.7,
                'adaptability': 0.4,
                'cooperation': 0.6
            },
            'neural': {
                'initial_state': {
                    'stress_level': 0.5,  # Higher stress
                    'sleep_quality': 0.60,  # Worse sleep
                    'cognitive_function': 0.95
                },
                'resilience': 0.5,
                'reactivity': 0.6,
                'adaptability': 0.5,
                'cooperation': 0.7
            }
        }
    }
    
    return seed_information


def main():
    print("\n" + "="*80)
    print("MIROFISH + LLM: INTELLIGENT PATIENT DIGITAL TWIN")
    print("GPT-4 Powered Agent Reasoning for Disease Prediction")
    print("="*80)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("\n✅ OpenAI API key found - LLM reasoning ENABLED")
        print("   Agents will use GPT-4 for intelligent decision-making")
    else:
        print("\n⚠️  No OpenAI API key found - using rule-based reasoning")
        print("   Set OPENAI_API_KEY environment variable to enable GPT-4")
        print("   Example: export OPENAI_API_KEY='sk-...'")
    
    # Step 1: Extract seed information
    print("\n📊 STEP 1: Extracting Seed Information")
    print("-" * 80)
    
    medical_report = """
    Patient ID: DT-SIM-0426
    Age: 38 years, Male, BMI: 26.5
    
    Risk Factors:
    - Borderline prediabetic (HbA1c 5.7%, Glucose 5.8)
    - Elevated LDL (3.6 mmol/L)
    - Borderline hypertension (132/86)
    - Sedentary lifestyle (1-2 sessions/week)
    - Poor sleep (6.5h average)
    - Moderate chronic stress
    - Mild inflammation (CRP 1.8)
    - Early fatty liver (ALT 42)
    
    Prognosis: WITHOUT lifestyle changes, high risk for:
    - Type 2 Diabetes (2-3 years)
    - Cardiovascular Disease (3-5 years)
    """
    
    print(medical_report)
    
    seed_info = extract_seed_with_worse_prognosis(medical_report)
    print(f"\n✓ Extracted seed for {len(seed_info['agent_seeds'])} agents")
    print("✓ Configured for REALISTIC disease emergence (no interventions)")
    
    # Step 2: Create digital patient
    print("\n🌐 STEP 2: Constructing Parallel Digital Patient")
    print("-" * 80)
    
    digital_patient = ParallelDigitalPatient(
        patient_id='DT-SIM-0426',
        seed_information=seed_info
    )
    
    print(f"✓ Initialized {len(digital_patient.agents)} autonomous agents")
    for agent_name, agent in digital_patient.agents.items():
        print(f"  - {agent_name.capitalize()}: {agent.health_status.value} "
              f"(resilience: {agent.personality.resilience:.1f})")
    
    # Step 3: Simulate WITHOUT interventions
    print("\n⏱️  STEP 3: Simulating 5-Year Natural Progression")
    print("-" * 80)
    print("NO interventions - showing natural disease emergence")
    print("Agents will interact and diseases will emerge from swarm intelligence\n")
    
    # Run simulation WITHOUT interventions
    timeline = digital_patient.simulate_future(
        days=1825,  # 5 years
        interventions=None  # NO interventions!
    )
    
    # Step 4: Analyze results
    print("\n📋 STEP 4: Disease Prediction Report")
    print("-" * 80)
    
    report = digital_patient.generate_report()
    print("\n" + report['summary'])
    
    # Step 5: Show LLM reasoning (if available)
    if api_key and digital_patient.diseases_emerged:
        print("\n🤖 STEP 5: LLM Analysis of Disease Emergence")
        print("-" * 80)
        
        from mirofish_engine.llm_reasoning import get_llm_engine
        llm_engine = get_llm_engine()
        
        for disease in digital_patient.diseases_emerged[:2]:  # First 2 diseases
            pathway = digital_patient.trace_disease_pathway(disease.name)
            
            print(f"\n📊 {disease.name}:")
            explanation = llm_engine.explain_disease_emergence(
                disease_name=disease.name,
                causative_agents=disease.causative_agents,
                timeline_events=pathway.get('key_events', []),
                final_states={
                    name: agent.state
                    for name, agent in digital_patient.agents.items()
                    if name in disease.causative_agents
                }
            )
            print(explanation)
    
    # Step 6: Agent conversations
    print("\n💬 STEP 6: Deep Interaction with Agents")
    print("-" * 80)
    
    if digital_patient.diseases_emerged:
        # Find most stressed agent
        most_stressed = max(
            digital_patient.agents.items(),
            key=lambda x: x[1].stress_level
        )
        agent_name = most_stressed[0]
        
        print(f"\nTalking to most stressed agent: {agent_name.capitalize()}")
        print(f"User: What happened to you over the past 5 years?")
        response = digital_patient.chat_with_agent(agent_name, 'what happened to you?')
        print(f"{response}\n")
    
    # Step 7: Save results
    print("\n💾 STEP 7: Saving Results")
    print("-" * 80)
    
    output_file = digital_patient.save_results()
    
    # Final summary
    print("\n" + "="*80)
    print("SIMULATION ANALYSIS")
    print("="*80)
    
    print(f"\n📈 Diseases Emerged: {len(digital_patient.diseases_emerged)}")
    if digital_patient.diseases_emerged:
        for disease in digital_patient.diseases_emerged:
            years = disease.day_emerged / 365
            print(f"  • {disease.name}: {disease.probability:.0%} at {years:.1f} years")
    else:
        print("  ⚠️  NO DISEASES EMERGED")
        print("\n  Possible reasons:")
        print("  1. Simulation parameters too conservative")
        print("  2. Agent resilience too high")
        print("  3. Homeostasis mechanisms too strong")
        print("  4. Need to increase stress/damage accumulation rates")
    
    print(f"\n💾 Full results: {output_file}")
    
    print("\n" + "="*80)
    print("WHY USE LLM REASONING?")
    print("="*80)
    print("""
Current System: Rule-based logic
  - Simple if/then rules
  - Fixed thresholds
  - No context understanding
  - No medical knowledge

With GPT-4 Integration:
  ✅ Intelligent decision-making based on medical knowledge
  ✅ Context-aware responses to complex situations
  ✅ Natural language explanations of agent reasoning
  ✅ Adaptive behavior based on personality and memory
  ✅ Medical knowledge graph integration (GraphRAG)
  ✅ Human-readable disease emergence narratives

To enable: export OPENAI_API_KEY='your-key-here'
""")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
