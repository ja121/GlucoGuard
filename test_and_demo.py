import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path
import shutil

from src.data_processing.cgm_dataset import AdvancedCGMDataset, CGMConfig
from src.models.predictor import HierarchicalGlucosePredictor
from src.training.lightning_module import GlucoseLightningModule

def generate_synthetic_cgm_data(num_subjects=5, num_days=7, sampling_rate_min=5, include_wearable_data=True):
    """Generates a synthetic CGM dataset with optional wearable data."""
    print("Generating synthetic CGM and wearable data...")
    
    dfs = []
    for subject_id in range(1, num_subjects + 1):
        start_date = pd.to_datetime('2025-01-01') + pd.to_timedelta(subject_id * 10, unit='D')
        end_date = start_date + pd.to_timedelta(num_days, unit='D')
        timestamps = pd.date_range(start=start_date, end=end_date, freq=f'{sampling_rate_min}T')
        
        # Create a realistic glucose pattern
        base_glucose = 120
        daily_cycle = 20 * np.sin(2 * np.pi * timestamps.hour / 24)
        meal_spikes = 30 * np.sin(2 * np.pi * timestamps.hour / 6) ** 2
        noise = np.random.normal(0, 5, len(timestamps))
        glucose = np.clip(base_glucose + daily_cycle + meal_spikes + noise, 40, 400)
        
        data = {
            'timestamp': timestamps,
            'glucose': glucose,
            'subject_id': subject_id
        }
        
        if include_wearable_data:
            # Heart Rate
            base_hr = 70
            hr_daily_cycle = 10 * np.sin(2 * np.pi * timestamps.hour / 24)
            hr_activity_spikes = 20 * np.sin(2 * np.pi * timestamps.hour / 4) ** 4
            hr_noise = np.random.normal(0, 3, len(timestamps))
            data['hr'] = np.clip(base_hr + hr_daily_cycle + hr_activity_spikes + hr_noise, 50, 180)
            
            # Heart Rate Variability
            data['hrv'] = np.clip(50 - 10 * np.sin(2 * np.pi * timestamps.hour / 24) + np.random.normal(0, 5, len(timestamps)), 20, 100)
            
            # Respiratory Rate
            data['resp_rate'] = np.clip(16 + 2 * np.sin(2 * np.pi * timestamps.hour / 24) + np.random.normal(0, 1, len(timestamps)), 12, 25)
            
            # Skin Temperature
            data['skin_temp'] = np.clip(34 + 0.5 * np.sin(2 * np.pi * timestamps.hour / 24) + np.random.normal(0, 0.2, len(timestamps)), 32, 36)
            
            # SpO2
            data['spo2'] = np.clip(98 - np.random.gamma(0.5, 0.5, len(timestamps)), 92, 100)
            
            # Accelerometer
            data['accel_x'] = np.sin(2 * np.pi * timestamps.hour / 2) + np.random.normal(0, 0.1, len(timestamps))
            data['accel_y'] = np.cos(2 * np.pi * timestamps.hour / 2) + np.random.normal(0, 0.1, len(timestamps))
            data['accel_z'] = -0.9 + np.random.normal(0, 0.05, len(timestamps))
            
            # Sleep Stages (0: Awake, 1: Light, 2: Deep, 3: REM)
            is_night = (timestamps.hour >= 22) | (timestamps.hour <= 6)
            data['sleep_stage'] = np.where(is_night, np.random.randint(1, 4, len(timestamps)), 0)

        df = pd.DataFrame(data)
        dfs.append(df)
        
    synthetic_data = pd.concat(dfs, ignore_index=True)
    print(f"Generated {len(synthetic_data)} data points for {num_subjects} subjects.")
    return synthetic_data

def run_test_pipeline(include_wearable_data=True):
    """Runs a test of the full data and model pipeline using synthetic data."""
    
    # 1. Generate synthetic data
    synthetic_df = generate_synthetic_cgm_data(include_wearable_data=include_wearable_data)

    # 2. Set up config and dataloader
    print(f"\nInitializing data pipeline {'with' if include_wearable_data else 'without'} wearable data...")
    config = CGMConfig()
    
    try:
        test_dataset = AdvancedCGMDataset(config=config, dataframe=synthetic_df, mode='train')
        
        # The dataset is very small, so we use a small batch size
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)
        print("Data pipeline initialized successfully.")
        
    except Exception as e:
        print(f"Error initializing data pipeline: {e}")
        return

    # 3. Initialize model and trainer
    print("\nInitializing model and trainer...")
    try:
        # We need to determine the number of features from the dataset
        sample_cgm_features, sample_wearable_features, _, _ = test_dataset[0]
        n_cgm_features = sample_cgm_features.shape[1]
        n_wearable_features = sample_wearable_features.shape[1] if sample_wearable_features is not None else 0
        print(f"Detected {n_cgm_features} CGM features and {n_wearable_features} wearable features.")

        model = HierarchicalGlucosePredictor(config=config, n_cgm_features=n_cgm_features, n_wearable_features=n_wearable_features)
        lightning_module = GlucoseLightningModule(model=model, config=config)
        
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2, # Run only on 2 batches for a quick test
            limit_val_batches=0,
            accelerator='cpu', # Force CPU for this test
            logger=False, # Disable logging for this test
            enable_checkpointing=False
        )
        print("Model and trainer initialized successfully.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # 4. Run training for a few steps
    print("\nRunning a few training steps (smoke test)...")
    try:
        trainer.fit(lightning_module, test_loader)
        print("\nSmoke test PASSED: Training loop ran without errors.")
    except Exception as e:
        print(f"\nSmoke test FAILED: Error during training loop: {e}")


if __name__ == '__main__':
    print("--- Running test with wearable data ---")
    run_test_pipeline(include_wearable_data=True)
    
    print("\n\n--- Running test without wearable data ---")
    run_test_pipeline(include_wearable_data=False)
