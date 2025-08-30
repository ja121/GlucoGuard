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

def generate_synthetic_cgm_data(num_subjects=5, num_days=7, sampling_rate_min=5):
    """Generates a synthetic CGM dataset."""
    print("Generating synthetic CGM data...")

    dfs = []
    for subject_id in range(1, num_subjects + 1):
        start_date = pd.to_datetime('2025-01-01') + pd.to_timedelta(subject_id * 10, unit='D')
        end_date = start_date + pd.to_timedelta(num_days, unit='D')
        timestamps = pd.date_range(start=start_date, end=end_date, freq=f'{sampling_rate_min}T')

        # Create a realistic glucose pattern with daily cycles and noise
        base_glucose = 120
        daily_cycle = 20 * np.sin(2 * np.pi * timestamps.hour / 24)
        meal_spikes = 30 * np.sin(2 * np.pi * timestamps.hour / 6) ** 2 # Spikes around meal times
        noise = np.random.normal(0, 5, len(timestamps))

        glucose = base_glucose + daily_cycle + meal_spikes + noise
        glucose = np.clip(glucose, 40, 400)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'glucose': glucose,
            'subject_id': subject_id
        })
        dfs.append(df)

    synthetic_data = pd.concat(dfs, ignore_index=True)
    print(f"Generated {len(synthetic_data)} data points for {num_subjects} subjects.")
    return synthetic_data

def run_test_pipeline():
    """Runs a test of the full data and model pipeline using synthetic data."""

    # 1. Generate synthetic data
    synthetic_df = generate_synthetic_cgm_data()

    # 2. Set up config and dataloader
    print("\nInitializing data pipeline with synthetic data...")
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
        sample_features, _, _ = test_dataset[0]
        num_features = sample_features.shape[1]
        print(f"Detected {num_features} features.")

        model = HierarchicalGlucosePredictor(config=config, n_features=num_features)
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
    synthetic_df = generate_synthetic_cgm_data()
    print("\n--- Synthetic Dataframe Info ---")
    print("Columns:", synthetic_df.columns)
    print("\nHead:\n", synthetic_df.head())
    print("\n---------------------------------")
