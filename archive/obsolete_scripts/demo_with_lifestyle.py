#!/usr/bin/env python3
"""
Patient Digital Twin with Realistic Lifestyle Simulation
Shows disease emergence from chronic stress, poor diet, and sedentary lifestyle
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mirofish_engine.parallel_digital_patient import ParallelDigitalPatient
from mirofish_engine.lifestyle_simulator import PatientLifestyleProfile, LifestyleSimulator
from mirofish_engine.internal_milieu import InternalMilieu
import json


def main():
    print("\n" + "="*80)
    print("PATIENT DIGITAL TWIN WITH REALISTIC LIFESTYLE SIMULATION")
    print("Disease Emergence from Chronic Stress + Poor Diet + Sedentary Lifestyle")
    print("="*80)
    
    # Create patient lifestyle profile
    print("\n📋 Patient Lifestyle Profile")
    print("-" * 80)
    
    lifestyle_profile = PatientLifestyleProfile(
        occupation="office_worker",
        exercise_frequency="low",  # 1-2 sessions/week
        diet_quality="poor",  # High carb, high fat
        sleep_pattern="insufficient",  # 6.5h average
        stress_level="moderate"  # Office work stress
    )
    
    print(f"""
Patient: DT-SIM-0426 (Age 38, Male, BMI 26.5)

Lifestyle Characteristics:
  • Occupation: Office worker (sedentary)
  • Exercise: Low (1-2 sessions/week)
  • Diet: Poor quality (high carb, high fat)
  • Sleep: Insufficient (6.5h average)
  • Stress: Moderate (chronic work stress)

This profile will generate realistic daily inputs:
  - Weekday stress: 0.5-0.7
  - Weekend stress: 0.3-0.4
  - Daily glucose spikes: 5-7 mmol/L
  - Daily fat intake: 80-120g
  - Exercise: Mostly weekends only
""")
    
    # Create lifestyle simulator
    lifestyle_sim = LifestyleSimulator(lifestyle_profile)
    
    # Show sample week
    print("\n📅 Sample Week of Lifestyle Inputs")
    print("-" * 80)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day_name in enumerate(days):
        inputs = lifestyle_sim.get_daily_inputs(i)
        print(f"{day_name}: Stress={inputs['lifestyle_stress']:.2f}, "
              f"Sleep={inputs['sleep_quality']:.2f}, "
              f"Exercise={inputs['exercise']:.2f}, "
              f"Glucose={inputs['food_glucose']:.1f}mmol/L, "
              f"Fat={inputs['dietary_fat']:.0f}g")
    
    # Create seed information with worse starting conditions
    print("\n🌱 Creating Parallel Digital Patient")
    print("-" * 80)
    
    seed_info = {
        'patient_id': 'DT-SIM-0426',
        'initial_composition': {
            'glucose': 5.8,
            'cortisol': 1.3,
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
                    'vessel_elasticity': 0.88,
                    'atherosclerosis_level': 0.12
                },
                'resilience': 0.5,
                'reactivity': 0.6,
                'adaptability': 0.4,
                'cooperation': 0.7
            },
            'metabolic': {
                'initial_state': {
                    'glucose': 5.8,
                    'insulin': 14.0,
                    'hba1c': 5.7,
                    'insulin_sensitivity': 0.70,
                    'beta_cell_function': 0.85,
                    'insulin_resistance': 0.30
                },
                'resilience': 0.4,
                'reactivity': 0.7,
                'adaptability': 0.3,
                'cooperation': 0.6
            },
            'renal': {
                'initial_state': {
                    'egfr': 95,
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
                    'fat_content': 0.08,
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
                    'crp': 2.2,
                    'inflammation': 0.22,
                    'immune_activation': 0.25
                },
                'resilience': 0.5,
                'reactivity': 0.8,
                'adaptability': 0.4,
                'cooperation': 0.5
            },
            'endocrine': {
                'initial_state': {
                    'cortisol': 1.3,
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
                    'stress_level': 0.5,
                    'sleep_quality': 0.60,
                    'cognitive_function': 0.95
                },
                'resilience': 0.5,
                'reactivity': 0.6,
                'adaptability': 0.5,
                'cooperation': 0.7
            }
        }
    }
    
    # Create digital patient
    digital_patient = ParallelDigitalPatient('DT-SIM-0426', seed_info)
    
    print(f"✓ Initialized {len(digital_patient.agents)} autonomous agents")
    
    # Simulate with lifestyle inputs
    print("\n⏱️  Running 5-Year Simulation with Lifestyle Inputs")
    print("-" * 80)
    print("Each day, agents receive realistic lifestyle inputs:")
    print("  • Chronic stress from work")
    print("  • Poor sleep quality")
    print("  • Sedentary lifestyle")
    print("  • High-carb, high-fat diet")
    print("\nDisease will emerge from cumulative effects...\n")
    
    # Custom simulation loop with lifestyle inputs
    timeline = []
    for day in range(1825):  # 5 years
        # Get daily lifestyle inputs
        daily_inputs = lifestyle_sim.get_daily_inputs(day)
        
        # Update environment with lifestyle inputs
        digital_patient.environment.external_inputs.update(daily_inputs)
        
        # Simulate one day
        day_state = digital_patient._simulate_one_day()
        timeline.append(day_state)
        
        # Check for disease emergence
        emerged_diseases = digital_patient._detect_disease_emergence()
        if emerged_diseases:
            for disease in emerged_diseases:
                if disease.name not in [d.name for d in digital_patient.diseases_emerged]:
                    digital_patient.diseases_emerged.append(disease)
                    print(f"⚠️  Day {day}: {disease.name} emerged (probability: {disease.probability:.1%})")
        
        # Progress indicator
        if day % 365 == 0 and day > 0:
            print(f"✓ Year {day//365} complete")
    
    digital_patient.timeline = timeline
    digital_patient.current_day = 1825
    
    # Generate report
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    
    report = digital_patient.generate_report()
    print("\n" + report['summary'])
    
    # Show agent final states
    print("\n📊 Final Agent States (After 5 Years)")
    print("-" * 80)
    for name, agent in digital_patient.agents.items():
        print(f"{name.capitalize():15} | Status: {agent.health_status.value:12} | "
              f"Stress: {agent.stress_level:5.1%} | {agent._summarize_state()}")
    
    # Save results
    print("\n💾 Saving Results")
    print("-" * 80)
    output_file = digital_patient.save_results()
    print(f"Complete data saved to: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    if digital_patient.diseases_emerged:
        print(f"\n✅ SUCCESS: {len(digital_patient.diseases_emerged)} diseases emerged from lifestyle simulation!")
        print("\nDiseases Detected:")
        for disease in digital_patient.diseases_emerged:
            years = disease.day_emerged / 365
            print(f"\n  {disease.name}")
            print(f"    • Probability: {disease.probability:.1%}")
            print(f"    • Time to onset: {years:.1f} years (Day {disease.day_emerged})")
            print(f"    • Causative agents: {', '.join(disease.causative_agents)}")
            print(f"    • Mechanism: {disease.mechanism}")
    else:
        print("\n⚠️  No diseases emerged - parameters may need further adjustment")
    
    print("\n" + "="*80)
    print("HOW IT WORKS")
    print("="*80)
    print("""
This simulation demonstrates MiroFish-style swarm intelligence:

1. LIFESTYLE INPUTS (New!)
   - Realistic daily patterns: stress, diet, sleep, exercise
   - Weekday vs weekend variations
   - Chronic exposure to risk factors

2. AGENT INTERACTIONS
   - Neural Agent: Detects chronic stress from poor sleep
   - Endocrine Agent: Releases cortisol in response
   - Metabolic Agent: Insulin resistance from cortisol + glucose spikes
   - Cardiovascular Agent: BP rises from stress + glucose damage
   - Immune Agent: Inflammation from vessel damage
   - Disease emerges from cascade!

3. SWARM INTELLIGENCE
   - No single agent causes disease
   - Disease emerges from INTERACTIONS
   - Temporal evolution over years
   - Realistic disease progression

This is how MiroFish works - agents interact based on inputs,
and complex outcomes (diseases) emerge from simple rules!
""")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
