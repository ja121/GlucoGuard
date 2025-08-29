"""
Adapter to make Claude Opus 4.1's glucose prediction code work with Awesome-CGM datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import os
from typing import Dict, List, Optional

class AwesomeCGMAdapter:
    """Adapter to load and preprocess Awesome-CGM datasets for the glucose prediction model"""

    def __init__(self, data_dir: str = "./cgm_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Dataset information from Awesome-CGM
        self.datasets = {
            'Brown2019': {
                'description': '168 T1D subjects, 6 months',
                'subjects': 168,
                'duration_months': 6,
                'url': 'https://github.com/IrinaStatsLab/Awesome-CGM'  # Will need actual URL
            },
            'Lynch2022': {
                'description': '440 T1D subjects, 13 weeks',
                'subjects': 440,
                'duration_weeks': 13,
                'url': 'https://github.com/IrinaStatsLab/Awesome-CGM'
            },
            'Shah2019': {
                'description': '168 healthy subjects',
                'subjects': 168,
                'type': 'healthy',
                'url': 'https://github.com/IrinaStatsLab/Awesome-CGM'
            },
            'Hall2018': {
                'description': '57 mixed subjects',
                'subjects': 57,
                'type': 'mixed',
                'url': 'https://github.com/IrinaStatsLab/Awesome-CGM'
            }
        }

    def download_dataset(self, dataset_name: str) -> bool:
        """Download dataset if not already present"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(self.datasets.keys())}")

        dataset_path = self.data_dir / dataset_name
        if dataset_path.exists():
            print(f"Dataset {dataset_name} already exists")
            return True

        print(f"Please manually download {dataset_name} from:")
        print(f"https://github.com/IrinaStatsLab/Awesome-CGM")
        print(f"And place it in: {dataset_path}")
        return False

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load and standardize dataset format for the glucose prediction model"""
        dataset_path = self.data_dir / dataset_name

        if not dataset_path.exists():
            print(f"Dataset {dataset_name} not found. Please download first.")
            return None

        # Try to find CSV files in the dataset directory
        csv_files = list(dataset_path.glob("*.csv"))
        if not csv_files:
            csv_files = list(dataset_path.rglob("*.csv"))

        if not csv_files:
            print(f"No CSV files found in {dataset_path}")
            return None

        # Load and combine CSV files
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        if not dfs:
            return None

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Standardize column names to match the expected format
        standardized_df = self._standardize_columns(combined_df)

        return standardized_df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match the expected format"""
        # Common column name mappings
        column_mappings = {
            # Glucose values
            'glucose': 'glucose',
            'cgm': 'glucose',
            'bg': 'glucose',
            'blood_glucose': 'glucose',
            'sensor_glucose': 'glucose',

            # Subject ID
            'subject_id': 'subject_id',
            'subject': 'subject_id',
            'patient_id': 'subject_id',
            'id': 'subject_id',

            # Timestamp
            'timestamp': 'timestamp',
            'time': 'timestamp',
            'datetime': 'timestamp',
            'date_time': 'timestamp',

            # Additional possible columns
            'meal': 'meal',
            'insulin': 'insulin',
            'activity': 'activity',
            'sleep': 'sleep'
        }

        # Rename columns
        df_renamed = df.rename(columns={
            col: column_mappings.get(col.lower(), col)
            for col in df.columns
        })

        # Ensure required columns exist
        required_columns = ['glucose', 'subject_id', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df_renamed.columns]

        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df_renamed.columns)}")

        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df_renamed.columns:
            df_renamed['timestamp'] = pd.to_datetime(df_renamed['timestamp'])

        # Sort by subject and time
        if 'subject_id' in df_renamed.columns and 'timestamp' in df_renamed.columns:
            df_renamed = df_renamed.sort_values(['subject_id', 'timestamp'])

        return df_renamed

    def prepare_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataset for the AdvancedCGMDataset class"""
        # Remove rows with missing glucose values
        df_clean = df.dropna(subset=['glucose'])

        # Filter valid glucose range (typically 40-400 mg/dL)
        df_clean = df_clean[(df_clean['glucose'] >= 40) & (df_clean['glucose'] <= 400)]

        # Ensure 5-minute sampling (resample if needed)
        df_resampled = self._resample_to_5min(df_clean)

        return df_resampled

    def _resample_to_5min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to 5-minute intervals"""
        resampled_dfs = []

        if 'subject_id' not in df.columns:
            # Handle case where there is no subject_id
            df = df.set_index('timestamp')
            resampled = df.resample('5T').mean()
            resampled['glucose'] = resampled['glucose'].fillna(method='ffill', limit=3)
            return resampled.reset_index()

        for subject_id in df['subject_id'].unique():
            subject_df = df[df['subject_id'] == subject_id].copy()
            subject_df = subject_df.set_index('timestamp')

            # Resample to 5-minute intervals
            resampled = subject_df.resample('5T').mean()
            resampled['subject_id'] = subject_id

            # Forward fill small gaps (up to 15 minutes)
            resampled['glucose'] = resampled['glucose'].fillna(method='ffill', limit=3)

            resampled_dfs.append(resampled.reset_index())

        return pd.concat(resampled_dfs, ignore_index=True)
