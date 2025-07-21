#!/usr/bin/env python3
"""
Complete Pipeline Execution Script

This script runs the complete pipeline in order:
1. Data Cleanup
2. Data Ingestion
3. Data Transformation  
4. Model Training
5. Prediction Pipeline Test
"""

import os
import sys
import time
import subprocess
import shutil

def cleanup_data_files():
    """Clean up processed data files and artifacts before starting pipeline."""
    print(f"\n{'='*60}")
    print("CLEANUP: Removing existing processed data and artifacts")
    print(f"{'='*60}")
    
    # Files and directories to clean (but preserve directory structure)
    files_to_clean = [
        "data/processed/wiki",
        "data/processed/combined", 
        "data/processed/tabular",
        "data/processed/sequential",
        "artifacts/preprocessor.pkl",
        "artifacts/combined_phase1",
        "saved_models/combined_phase1"
    ]
    
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"+ Removed file: {file_path}")
                else:
                    shutil.rmtree(file_path)
                    print(f"+ Removed directory: {file_path}")
            except Exception as e:
                print(f"! Could not remove {file_path}: {e}")
        else:
            print(f"i Not found: {file_path}")
    
    # Ensure required directories exist
    required_dirs = [
        "data/processed/wiki",
        "data/processed/combined", 
        "data/processed/tabular",
        "data/processed/sequential",
        "artifacts",
        "saved_models"
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("+ Ensured required directories exist")

def run_command(command, description):
    """Run a command and log the results."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"+ SUCCESS ({elapsed_time:.2f}s)")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
        else:
            print(f"X FAILED ({elapsed_time:.2f}s)")
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"X EXCEPTION: {e}")
        return False

def main():
    """Run the complete pipeline."""
    print(">> Starting Complete Wiki Fraud Detection Pipeline")
    
    # Step 0: Cleanup existing data
    # cleanup_data_files()
    
    # Pipeline commands
    commands = [
        # Step 1: Data Ingestion
        {
            'command': 'python -m src.components.data_ingestion',
            'description': "Data Ingestion - Loading and preprocessing raw wiki data"
        },
        
        # Step 2: Data Transformation
        {
            'command': 'python -m src.components.data_transformation',
            'description': "Data Transformation - Feature engineering and preprocessing"
        },
        {
            'command': 'python tests\\verify_combined_data.py',
            'description': "Data Verification - Verifying transformed data integrity"
        },
        
        # Step 3: Model Training
        {
            'command': 'python -m scripts.train_phase1',
            'description': "Model Training - Training the fraud detection model"
        },
        {
            'command': 'python -m scripts.evaluate',
            'description': "Model Training - Training the fraud detection model (Phase 2)"
        },
        
        # Step 4: Prediction Pipeline Test
        {
            'command': 'python -m scripts.predict_su',
            'description': "Prediction Pipeline - Testing with simple fraud detection model"
        }
    ]
    
    total_start_time = time.time()
    success_count = 0
    
    # Run each command
    for i, cmd_info in enumerate(commands, 1):
        print(f"\n\n>> EXECUTING STEP {i}/{len(commands)}")
        success = run_command(cmd_info['command'], cmd_info['description'])
        if success:
            success_count += 1
        else:
            print(f"! Step {i} failed. Continuing to next step...")
    
    # Summary
    total_time = time.time() - total_start_time
    print(f"\n\n{'='*60}")
    print(">> PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Steps Completed: {success_count}/{len(commands)}")
    print(f"Success Rate: {success_count/len(commands):.1%}")
    print(f"Total Time: {total_time:.2f} seconds")
    
    if success_count == len(commands):
        print("* COMPLETE PIPELINE SUCCESS!")
    elif success_count > 0:
        print("! PARTIAL SUCCESS - Some steps completed")
    else:
        print("X PIPELINE FAILED - No steps completed successfully")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()