import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import warnings

from src.data_processing.adapter import AwesomeCGMAdapter

warnings.filterwarnings('ignore')

@dataclass
class CGMConfig:
    """Configuration for CGM processing"""
    sequence_length: int = 36 # 3 hours at 5-min intervals
    prediction_horizon: int = 6 # 30 minutes ahead
    sampling_rate: int = 5 # minutes

    # Feature engineering
    use_derivatives: bool = True
    use_fft: bool = True
    use_wavelets: bool = True

    # Thresholds
    hypo_threshold: float = 70.0
    hyper_threshold: float = 180.0
    severe_hypo: float = 54.0
    severe_hyper: float = 250.0

class AdvancedCGMDataset(Dataset):
    def __init__(self, dataset_name: str, config: CGMConfig, mode='train'):
        self.adapter = AwesomeCGMAdapter()
        self.config = config
        self.mode = mode

        # Load dataset using adapter
        raw_data = self.adapter.load_dataset(dataset_name)
        if raw_data is None:
            raise ValueError(f"Could not load dataset {dataset_name}")

        # Prepare for model
        self.data = self.adapter.prepare_for_model(raw_data)

        # Continue with original processing...
        self.data = self._engineer_features(self.data)
        self.sequences = self._create_sequences()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract 50+ features from CGM signal"""

        # Temporal features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])

        # Circadian encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Statistical features (multiple windows)
        for window in [6, 12, 24, 48]:  # 30min, 1h, 2h, 4h
            df[f'mean_{window}'] = df.groupby('subject_id')['glucose'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'std_{window}'] = df.groupby('subject_id')['glucose'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f'cv_{window}'] = df[f'std_{window}'] / (df[f'mean_{window}'] + 1e-8)

        # Derivatives (rate of change)
        df['roc_5min'] = df.groupby('subject_id')['glucose'].diff()
        df['roc_15min'] = df.groupby('subject_id')['glucose'].diff(3)
        df['roc_30min'] = df.groupby('subject_id')['glucose'].diff(6)

        # Acceleration (2nd derivative)
        df['acceleration'] = df.groupby('subject_id')['roc_5min'].diff()

        # Glucose variability metrics
        df['mage'] = self._calculate_mage(df)  # Mean Amplitude of Glycemic Excursions
        df['modd'] = self._calculate_modd(df)  # Mean of Daily Differences
        df['conga'] = self._calculate_conga(df)  # Continuous Overall Net Glycemic Action

        # Risk indices
        df['lbgi'] = self._calculate_lbgi(df['glucose'])  # Low Blood Glucose Index
        df['hbgi'] = self._calculate_hbgi(df['glucose'])  # High Blood Glucose Index

        # Frequency domain features (FFT)
        if self.config.use_fft:
            df = self._add_fft_features(df)

        # Wavelet features
        if self.config.use_wavelets:
            df = self._add_wavelet_features(df)

        return df

    def _create_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor, dict]]:
        """Create multi-scale sequences with metadata"""
        sequences = []

        # This part needs to be adapted to handle the dataframe from the adapter
        # The original code assumes a single dataframe, but we have multiple datasets
        # For now, let's assume self.data is a single dataframe as the original code did

        # Get all feature columns
        feature_cols = [col for col in self.data.columns if col not in ['subject_id', 'timestamp', 'glucose']]

        for subject_id in self.data['subject_id'].unique():
            subject_data = self.data[self.data['subject_id'] == subject_id].copy()

            # Skip if too short
            if len(subject_data) < self.config.sequence_length + self.config.prediction_horizon:
                continue

            # Create overlapping windows with stride
            stride = 3 if self.mode == 'train' else 6  # More augmentation for training

            # Scale features
            scaler = RobustScaler()
            subject_data[feature_cols] = scaler.fit_transform(subject_data[feature_cols])

            for i in range(0, len(subject_data) - self.config.sequence_length - self.config.prediction_horizon, stride):
                # Input sequence
                seq_data = subject_data.iloc[i:i + self.config.sequence_length]

                # Target values (multiple horizons)
                targets = {}
                # 30min, 1hr, 1.5hr predictions
                target_horizons = [self.config.prediction_horizon, self.config.prediction_horizon * 2, self.config.prediction_horizon * 3]

                glucose_targets = []
                hypo_targets = []
                hyper_targets = []

                for h in target_horizons:
                    if i + self.config.sequence_length + h < len(subject_data):
                        target_glucose = subject_data.iloc[i + self.config.sequence_length + h]['glucose']
                        glucose_targets.append(target_glucose)
                        hypo_targets.append(int(target_glucose < self.config.hypo_threshold))
                        hyper_targets.append(int(target_glucose > self.config.hyper_threshold))
                    else:
                        # Handle cases where the horizon goes beyond the data
                        glucose_targets.append(np.nan)
                        hypo_targets.append(np.nan)
                        hyper_targets.append(np.nan)

                targets['glucose'] = torch.tensor(glucose_targets, dtype=torch.float32)
                targets['risk'] = torch.tensor(hypo_targets + hyper_targets, dtype=torch.float32)

                # Extract features
                features = torch.tensor(seq_data[feature_cols].values, dtype=torch.float32)

                # Metadata
                metadata = {
                    'subject_id': subject_id,
                    'timestamp': seq_data.iloc[-1]['timestamp'],
                    'hour': seq_data.iloc[-1]['hour'],
                    'day_of_week': seq_data.iloc[-1]['day_of_week']
                }

                sequences.append((features, targets, metadata))

        return sequences

    def _calculate_lbgi(self, glucose: pd.Series) -> pd.Series:
        """Low Blood Glucose Index - risk metric"""
        f_glucose = 1.509 * (np.log(glucose)**1.084 - 5.381)
        rl = np.where(f_glucose < 0, 10 * f_glucose**2, 0)
        return pd.Series(rl, index=glucose.index)

    def _calculate_hbgi(self, glucose: pd.Series) -> pd.Series:
        """High Blood Glucose Index"""
        f_glucose = 1.509 * (np.log(glucose)**1.084 - 5.381)
        rh = np.where(f_glucose > 0, 10 * f_glucose**2, 0)
        return pd.Series(rh, index=glucose.index)

    def _calculate_mage(self, df: pd.DataFrame) -> pd.Series:
        """Placeholder for MAGE calculation"""
        return pd.Series(0, index=df.index)

    def _calculate_modd(self, df: pd.DataFrame) -> pd.Series:
        """Placeholder for MODD calculation"""
        return pd.Series(0, index=df.index)

    def _calculate_conga(self, df: pd.DataFrame) -> pd.Series:
        """Placeholder for CONGA calculation"""
        return pd.Series(0, index=df.index)

    def _add_fft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder for FFT feature calculation"""
        return df

    def _add_wavelet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder for wavelet feature calculation"""
        return df
