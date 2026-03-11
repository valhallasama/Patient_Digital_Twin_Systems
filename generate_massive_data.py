#!/usr/bin/env python3
"""
Massive-scale synthetic data generation
Generates 10M+ patients using rule-based algorithms (NO LLM)

DATA GENERATION METHOD:
- Uses statistical distributions (normal, uniform, beta, etc.)
- Rule-based disease risk calculations
- Deterministic algorithms based on medical literature
- NO AI/LLM involved - pure mathematical/statistical generation
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))

from synthetic_data_generator.patient_population_generator import PatientPopulationGenerator
from synthetic_data_generator.disease_progression_generator import DiseaseProgressionGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_storage_info():
    """Get current storage information"""
    stat = shutil.disk_usage('/home/tc115/Yue/Patient_Digital_Twin_Systems')
    total_gb = stat.total / (1024**3)
    used_gb = stat.used / (1024**3)
    free_gb = stat.free / (1024**3)
    return total_gb, used_gb, free_gb


def calculate_max_patients_by_storage(target_storage_gb=50):
    """Calculate how many patients we can generate with target storage"""
    # Approximately 6KB per patient (conservative estimate)
    bytes_per_patient = 6 * 1024
    target_bytes = target_storage_gb * (1024**3)
    max_patients = int(target_bytes / bytes_per_patient)
    return max_patients


def generate_massive_dataset(
    target_patients=10_000_000,
    batch_size=100_000,
    output_dir="data/synthetic_massive",
    include_trajectories=False,
    trajectory_sample_rate=0.01
):
    """
    Generate massive synthetic patient dataset
    
    Args:
        target_patients: Total number of patients to generate
        batch_size: Patients per batch (larger = faster but more memory)
        output_dir: Output directory
        include_trajectories: Whether to generate disease trajectories
        trajectory_sample_rate: Fraction of patients to generate trajectories for
    """
    
    logger.info("="*80)
    logger.info("MASSIVE-SCALE SYNTHETIC DATA GENERATION")
    logger.info("="*80)
    logger.info("\nDATA GENERATION METHOD:")
    logger.info("  ✓ Rule-based statistical algorithms")
    logger.info("  ✓ Medical literature-based risk models")
    logger.info("  ✓ Deterministic mathematical functions")
    logger.info("  ✗ NO LLM/AI involved in generation")
    logger.info("="*80)
    
    # Check storage
    total_gb, used_gb, free_gb = get_storage_info()
    logger.info(f"\nStorage Status:")
    logger.info(f"  Total: {total_gb:.1f} GB")
    logger.info(f"  Used: {used_gb:.1f} GB")
    logger.info(f"  Free: {free_gb:.1f} GB")
    
    # Estimate storage needed
    estimated_storage_gb = (target_patients * 6 * 1024) / (1024**3)
    logger.info(f"\nEstimated storage needed: {estimated_storage_gb:.2f} GB")
    
    if estimated_storage_gb > free_gb * 0.8:
        logger.warning(f"⚠️  May not have enough storage!")
        logger.warning(f"   Recommended: Reduce target to {int(free_gb * 0.8 * 1024**3 / (6*1024)):,} patients")
        logger.warning(f"   Or free up more disk space")
        
        response = input("\nContinue anyway? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Generation cancelled")
            return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate batches
    num_batches = (target_patients + batch_size - 1) // batch_size
    
    logger.info(f"\nGeneration Plan:")
    logger.info(f"  Target patients: {target_patients:,}")
    logger.info(f"  Batch size: {batch_size:,}")
    logger.info(f"  Number of batches: {num_batches}")
    logger.info(f"  Include trajectories: {include_trajectories}")
    if include_trajectories:
        logger.info(f"  Trajectory sample rate: {trajectory_sample_rate:.1%}")
    logger.info("="*80)
    
    generator = PatientPopulationGenerator()
    prog_gen = DiseaseProgressionGenerator() if include_trajectories else None
    
    total_generated = 0
    all_batch_files = []
    
    for batch_num in range(num_batches):
        start_id = batch_num * batch_size
        current_batch_size = min(batch_size, target_patients - total_generated)
        
        if current_batch_size <= 0:
            break
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH {batch_num + 1}/{num_batches}")
        logger.info(f"{'='*80}")
        logger.info(f"Generating patients {start_id:,} to {start_id + current_batch_size - 1:,}")
        
        # Check storage before each batch
        _, _, free_gb = get_storage_info()
        logger.info(f"Free storage: {free_gb:.2f} GB")
        
        if free_gb < 10:
            logger.warning("⚠️  Low storage (<10GB)! Stopping generation.")
            break
        
        try:
            # Generate batch
            data = generator.generate_complete_population(n=current_batch_size, output_dir=output_dir)
            
            # Save batch file
            batch_file = Path(output_dir) / f"batch_{batch_num:04d}.csv"
            data['complete'].to_csv(batch_file, index=False)
            all_batch_files.append(batch_file)
            
            total_generated += len(data['complete'])
            
            logger.info(f"✓ Batch saved: {batch_file.name}")
            logger.info(f"  Total generated: {total_generated:,} patients")
            
            # Generate trajectories for sample
            if include_trajectories and prog_gen:
                sample_size = max(1, int(len(data['complete']) * trajectory_sample_rate))
                sample_patients = data['complete'].sample(n=sample_size)
                
                logger.info(f"  Generating trajectories for {sample_size:,} sample patients...")
                
                traj_dir = Path(output_dir) / f"trajectories_batch_{batch_num:04d}"
                trajectories = prog_gen.generate_population_trajectories(
                    sample_patients,
                    years=10,
                    output_dir=str(traj_dir)
                )
                logger.info(f"  ✓ Trajectories saved to {traj_dir.name}")
            
            # Progress update
            progress = (batch_num + 1) / num_batches * 100
            logger.info(f"\nOverall Progress: {progress:.1f}% ({total_generated:,}/{target_patients:,})")
            
        except Exception as e:
            logger.error(f"❌ Error in batch {batch_num}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nTotal patients generated: {total_generated:,}")
    logger.info(f"Batch files created: {len(all_batch_files)}")
    logger.info(f"Data saved to: {output_dir}")
    
    # Storage summary
    _, used_gb_after, free_gb_after = get_storage_info()
    storage_used = used_gb_after - used_gb
    logger.info(f"\nStorage used: {storage_used:.2f} GB")
    logger.info(f"Remaining free: {free_gb_after:.2f} GB")
    
    # Generate summary statistics
    logger.info("\n" + "="*80)
    logger.info("DATASET STATISTICS")
    logger.info("="*80)
    
    # Load first batch for quick stats
    if all_batch_files:
        sample_df = pd.read_csv(all_batch_files[0])
        
        logger.info(f"\nSample Statistics (Batch 0):")
        logger.info(f"  Age: {sample_df['age'].mean():.1f} ± {sample_df['age'].std():.1f} years")
        logger.info(f"  BMI: {sample_df['bmi'].mean():.1f} ± {sample_df['bmi'].std():.1f}")
        logger.info(f"  Gender: {(sample_df['gender']=='male').mean():.1%} male")
        logger.info(f"\nDisease Prevalence:")
        logger.info(f"  Hypertension: {sample_df['hypertension'].mean():.1%}")
        logger.info(f"  Diabetes: {sample_df['diabetes'].mean():.1%}")
        logger.info(f"  Heart Disease: {sample_df['heart_disease'].mean():.1%}")
        logger.info(f"  Cancer: {sample_df['cancer'].mean():.1%}")
    
    # Save metadata
    metadata = {
        'total_patients': total_generated,
        'num_batches': len(all_batch_files),
        'batch_size': batch_size,
        'storage_used_gb': storage_used,
        'generation_method': 'rule-based statistical (NO LLM)',
        'output_directory': output_dir
    }
    
    import json
    with open(Path(output_dir) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n✅ Massive data generation complete!")
    logger.info(f"Metadata saved to: {output_dir}/metadata.json")
    
    return total_generated, all_batch_files


def main():
    logger.info("\n" + "="*80)
    logger.info("MASSIVE-SCALE DATA GENERATOR")
    logger.info("="*80)
    
    # Configuration options
    logger.info("\nConfiguration Options:")
    logger.info("1. Generate 10 million patients (recommended)")
    logger.info("2. Generate 20 million patients")
    logger.info("3. Generate 50 million patients (requires ~300GB)")
    logger.info("4. Custom amount")
    logger.info("5. Maximum based on available storage")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        target = 10_000_000
    elif choice == '2':
        target = 20_000_000
    elif choice == '3':
        target = 50_000_000
    elif choice == '4':
        target = int(input("Enter number of patients: "))
    elif choice == '5':
        _, _, free_gb = get_storage_info()
        target = calculate_max_patients_by_storage(free_gb * 0.8)
        logger.info(f"Maximum patients based on storage: {target:,}")
    else:
        logger.info("Invalid choice, using default: 10 million")
        target = 10_000_000
    
    # Ask about trajectories
    include_traj = input("\nGenerate disease trajectories? (yes/no): ").strip().lower() == 'yes'
    
    # Start generation
    generate_massive_dataset(
        target_patients=target,
        batch_size=100_000,
        output_dir="data/synthetic_massive",
        include_trajectories=include_traj,
        trajectory_sample_rate=0.01  # 1% sample for trajectories
    )


if __name__ == "__main__":
    main()
