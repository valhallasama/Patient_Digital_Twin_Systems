#!/usr/bin/env python3
"""
MiroFish-Inspired Patient Digital Twin Demo
Demonstrates swarm intelligence for disease prediction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mirofish_engine.parallel_digital_patient import ParallelDigitalPatient
from mirofish_engine.organ_agents import create_agent_from_seed
import json


def extract_seed_from_medical_report(report_text: str) -> dict:
    """
    Extract seed information from medical report
    This is the "seed from real world" that MiroFish uses
    """
    # For demo, using the patient report from earlier
    seed_information = {
        'patient_id': 'DT-SIM-0426',
        'initial_composition': {
            'glucose': 5.8,
            'cortisol': 1.2,  # Slightly elevated due to stress
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
                    'vessel_elasticity': 0.9,  # Slightly reduced
                    'atherosclerosis_level': 0.1
                },
                'resilience': 0.6,
                'reactivity': 0.5,
                'adaptability': 0.5,
                'cooperation': 0.7
            },
            'metabolic': {
                'initial_state': {
                    'glucose': 5.8,
                    'insulin': 12.0,
                    'hba1c': 5.7,
                    'insulin_sensitivity': 0.75,  # Reduced
                    'beta_cell_function': 0.9,
                    'insulin_resistance': 0.25
                },
                'resilience': 0.5,
                'reactivity': 0.6,
                'adaptability': 0.4,
                'cooperation': 0.6
            },
            'renal': {
                'initial_state': {
                    'egfr': 98,
                    'creatinine': 90,
                    'filtration_capacity': 0.98,
                    'damage_level': 0.02
                },
                'resilience': 0.7,
                'reactivity': 0.4,
                'adaptability': 0.6,
                'cooperation': 0.7
            },
            'hepatic': {
                'initial_state': {
                    'alt': 42,
                    'ast': 30,
                    'ldl': 3.6,
                    'hdl': 1.1,
                    'fat_content': 0.05,  # Slight fatty liver
                    'detox_capacity': 0.95
                },
                'resilience': 0.6,
                'reactivity': 0.5,
                'adaptability': 0.5,
                'cooperation': 0.6
            },
            'immune': {
                'initial_state': {
                    'wbc': 6.5,
                    'crp': 1.8,
                    'inflammation': 0.18,
                    'immune_activation': 0.2
                },
                'resilience': 0.6,
                'reactivity': 0.7,
                'adaptability': 0.5,
                'cooperation': 0.5
            },
            'endocrine': {
                'initial_state': {
                    'cortisol': 1.2,
                    'thyroid': 1.0,
                    'stress_response': 0.3
                },
                'resilience': 0.5,
                'reactivity': 0.6,
                'adaptability': 0.5,
                'cooperation': 0.6
            },
            'neural': {
                'initial_state': {
                    'stress_level': 0.4,  # Moderate stress
                    'sleep_quality': 0.65,  # 6.5h average
                    'cognitive_function': 1.0
                },
                'resilience': 0.6,
                'reactivity': 0.5,
                'adaptability': 0.6,
                'cooperation': 0.7
            }
        }
    }
    
    return seed_information


def main():
    print("\n" + "="*80)
    print("MIROFISH-INSPIRED PATIENT DIGITAL TWIN")
    print("Swarm Intelligence for Disease Prediction")
    print("="*80)
    
    # Step 1: Extract seed information (like MiroFish extracts from news/events)
    print("\n📊 STEP 1: Extracting Seed Information from Medical Report")
    print("-" * 80)
    
    medical_report = """
    Patient ID: DT-SIM-0426
    Age: 38 years, Male
    BMI: 26.5 (Overweight)
    
    Vital Signs:
    - BP: 132/86 mmHg (borderline elevated)
    - Heart Rate: 72 bpm
    
    Lab Results:
    - Fasting Glucose: 5.8 mmol/L (borderline prediabetic)
    - HbA1c: 5.7% (upper limit normal)
    - LDL: 3.6 mmol/L (elevated)
    - HDL: 1.1 mmol/L
    - CRP: 1.8 mg/L (mild inflammation)
    - ALT: 42 U/L (slightly elevated)
    
    Lifestyle:
    - Exercise: 1-2 sessions/week (sedentary)
    - Sleep: 6.5h average (insufficient)
    - Stress: Moderate (office work)
    """
    
    print(medical_report)
    
    seed_info = extract_seed_from_medical_report(medical_report)
    print(f"\n✓ Extracted seed information for {len(seed_info['agent_seeds'])} body system agents")
    
    # Step 2: Create parallel digital patient (like MiroFish creates parallel world)
    print("\n🌐 STEP 2: Constructing Parallel Digital Patient")
    print("-" * 80)
    
    digital_patient = ParallelDigitalPatient(
        patient_id='DT-SIM-0426',
        seed_information=seed_info
    )
    
    print(f"✓ Initialized {len(digital_patient.agents)} autonomous agents:")
    for agent_name, agent in digital_patient.agents.items():
        print(f"  - {agent_name.capitalize()} Agent: {agent.health_status.value}")
    
    # Step 3: Simulate future (like MiroFish simulates society evolution)
    print("\n⏱️  STEP 3: Simulating 5-Year Future Evolution")
    print("-" * 80)
    print("Each agent will interact, make decisions, and evolve...")
    print("Disease emergence will be detected from swarm intelligence patterns\n")
    
    # Define interventions to test
    interventions = [
        # No intervention for first 2 years to see natural progression
        # Then try lifestyle changes
        {
            'day': 730,  # After 2 years
            'type': 'lifestyle',
            'change': 'exercise',
            'intensity': 0.7  # Moderate exercise
        },
        {
            'day': 730,
            'type': 'lifestyle',
            'change': 'diet',
            'diet_type': 'mediterranean'
        }
    ]
    
    # Run simulation
    timeline = digital_patient.simulate_future(
        days=1825,  # 5 years
        interventions=interventions
    )
    
    # Step 4: Analyze results (like MiroFish generates prediction report)
    print("\n📋 STEP 4: Generating Prediction Report")
    print("-" * 80)
    
    report = digital_patient.generate_report()
    
    print("\n" + report['summary'])
    
    # Step 5: Deep interaction with agents (like MiroFish's agent chat)
    print("\n💬 STEP 5: Deep Interaction with Agents")
    print("-" * 80)
    
    # Chat with metabolic agent
    print("\nUser: Hey Metabolic Agent, why did diabetes emerge?")
    response = digital_patient.chat_with_agent('metabolic', 'why did diabetes emerge?')
    print(f"{response}\n")
    
    # Chat with cardiovascular agent
    print("User: Cardiovascular Agent, what's your current state?")
    response = digital_patient.chat_with_agent('cardiovascular', 'what is your state?')
    print(f"{response}\n")
    
    # Step 6: Trace disease pathways (swarm intelligence analysis)
    print("\n🔍 STEP 6: Tracing Disease Emergence Pathways")
    print("-" * 80)
    
    if digital_patient.diseases_emerged:
        for disease in digital_patient.diseases_emerged:
            print(f"\nDisease: {disease.name}")
            print(f"Probability: {disease.probability:.1%}")
            print(f"Emerged on Day: {disease.day_emerged} (~{disease.day_emerged/365:.1f} years)")
            print(f"Causative Agents: {', '.join(disease.causative_agents)}")
            print(f"Mechanism: {disease.mechanism}")
            
            # Get detailed pathway
            pathway = digital_patient.trace_disease_pathway(disease.name)
            if 'key_events' in pathway and pathway['key_events']:
                print("\nKey Events Leading to Emergence:")
                for event in pathway['key_events'][-5:]:
                    print(f"  Day {event['day']}: {event['event']} (stress: {event['stress_level']:.1%})")
    
    # Step 7: Save results
    print("\n💾 STEP 7: Saving Simulation Results")
    print("-" * 80)
    
    output_file = digital_patient.save_results()
    print(f"Complete simulation data saved to: {output_file}")
    
    # Final summary
    print("\n" + "="*80)
    print("MIROFISH-STYLE PREDICTION COMPLETE")
    print("="*80)
    print("\n✨ Key Insights:")
    print("  1. Disease emerges from SWARM INTELLIGENCE of interacting agents")
    print("  2. Each organ/system is an autonomous agent with memory & personality")
    print("  3. Predictions show WHEN and HOW diseases will emerge")
    print("  4. Interventions can be tested in the digital twin before real patient")
    print("  5. You can chat with any agent to understand their state and decisions")
    print("\nThis is a living, evolving parallel digital patient! 🚀")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
