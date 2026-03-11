#!/usr/bin/env python3
"""
Automatic data generation with storage monitoring
Generates maximum amount of synthetic patient data based on available storage
"""

import os
import sys
import shutil
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from synthetic_data_generator.patient_population_generator import PatientPopulationGenerator
from synthetic_data_generator.disease_progression_generator import DiseaseProgressionGenerator
from synthetic_data_generator.lifestyle_generator import LifestyleGenerator
from synthetic_data_generator.environment_generator import EnvironmentGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_available_storage_gb():
    """Get available storage in GB"""
    stat = shutil.disk_usage('/home/tc115/Yue/Patient_Digital_Twin_Systems')
    available_gb = stat.free / (1024**3)
    return available_gb


def estimate_data_size_per_patient():
    """Estimate storage per patient in bytes (conservative estimate)"""
    # Complete patient record: ~2KB per patient in CSV
    # Disease trajectories (10 years): ~1KB per patient
    # Lifestyle data (30 days): ~3KB per patient
    # Total: ~6KB per patient
    return 6 * 1024  # 6KB per patient


def calculate_max_patients(available_gb, safety_margin=0.8):
    """Calculate maximum patients we can generate"""
    available_bytes = available_gb * (1024**3) * safety_margin
    bytes_per_patient = estimate_data_size_per_patient()
    max_patients = int(available_bytes / bytes_per_patient)
    return max_patients


def generate_batch(generator, start_id, batch_size, output_dir):
    """Generate a batch of patients"""
    logger.info(f"Generating batch: {start_id} to {start_id + batch_size - 1}")
    
    # Generate patient population
    data = generator.generate_complete_population(n=batch_size, output_dir=output_dir)
    
    # Save with batch identifier
    batch_file = Path(output_dir) / f"batch_{start_id}_{start_id + batch_size - 1}.csv"
    data['complete'].to_csv(batch_file, index=False)
    
    return data['complete']


def main():
    logger.info("="*80)
    logger.info("AUTOMATIC DATA GENERATION WITH STORAGE MONITORING")
    logger.info("="*80)
    
    # Check available storage
    available_gb = get_available_storage_gb()
    logger.info(f"\nAvailable storage: {available_gb:.2f} GB")
    
    # Calculate maximum patients
    max_patients = calculate_max_patients(available_gb)
    logger.info(f"Estimated maximum patients: {max_patients:,}")
    
    # Ask user for confirmation or use default
    target_patients = min(max_patients, 5_000_000)  # Cap at 5 million for safety
    logger.info(f"\nTarget generation: {target_patients:,} patients")
    logger.info(f"Estimated storage needed: {(target_patients * estimate_data_size_per_patient()) / (1024**3):.2f} GB")
    
    output_dir = "data/synthetic"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate in batches to avoid memory issues
    batch_size = 50_000  # 50K patients per batch
    num_batches = (target_patients + batch_size - 1) // batch_size
    
    logger.info(f"\nGenerating in {num_batches} batches of {batch_size:,} patients each")
    logger.info("="*80)
    
    generator = PatientPopulationGenerator()
    prog_gen = DiseaseProgressionGenerator()
    
    total_generated = 0
    
    for batch_num in range(num_batches):
        start_id = batch_num * batch_size
        current_batch_size = min(batch_size, target_patients - total_generated)
        
        if current_batch_size <= 0:
            break
        
        logger.info(f"\n[BATCH {batch_num + 1}/{num_batches}]")
        
        # Check storage before each batch
        current_available = get_available_storage_gb()
        logger.info(f"Available storage: {current_available:.2f} GB")
        
        if current_available < 5:  # Stop if less than 5GB available
            logger.warning("Low storage detected! Stopping generation.")
            break
        
        # Generate batch
        try:
            patients = generate_batch(generator, start_id, current_batch_size, output_dir)
            total_generated += len(patients)
            
            logger.info(f"✓ Batch complete. Total generated: {total_generated:,} patients")
            
            # Generate disease trajectories for a sample (to save space)
            sample_size = min(1000, len(patients))
            sample_patients = patients.sample(n=sample_size)
            
            logger.info(f"  Generating disease trajectories for {sample_size} sample patients...")
            trajectories = prog_gen.generate_population_trajectories(
                sample_patients, 
                years=10,
                output_dir=f"{output_dir}/trajectories_batch_{batch_num}"
            )
            
        except Exception as e:
            logger.error(f"Error in batch {batch_num}: {e}")
            break
    
    logger.info("\n" + "="*80)
    logger.info("DATA GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total patients generated: {total_generated:,}")
    logger.info(f"Data saved to: {output_dir}")
    
    final_available = get_available_storage_gb()
    storage_used = available_gb - final_available
    logger.info(f"Storage used: {storage_used:.2f} GB")
    logger.info(f"Remaining storage: {final_available:.2f} GB")
    
    # Generate summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    import pandas as pd
    
    # Load first batch for statistics
    first_batch = pd.read_csv(f"{output_dir}/complete_patient_data.csv")
    
    logger.info(f"\nAge: {first_batch['age'].mean():.1f} ± {first_batch['age'].std():.1f}")
    logger.info(f"BMI: {first_batch['bmi'].mean():.1f} ± {first_batch['bmi'].std():.1f}")
    logger.info(f"Diabetes prevalence: {first_batch['diabetes'].mean():.1%}")
    logger.info(f"Hypertension prevalence: {first_batch['hypertension'].mean():.1%}")
    logger.info(f"Heart disease prevalence: {first_batch['heart_disease'].mean():.1%}")
    
    logger.info("\n✅ Automatic data generation completed successfully!")


if __name__ == "__main__":
    main()
