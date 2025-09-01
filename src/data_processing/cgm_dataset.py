import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import warnings

from src.data_processing.adapter import AwesomeCGMAdapter
from src.features import glycemic_variability as gv
from src.features import trend_arrows as ta

warnings.filterwarnings('ignore')

@dataclass
class CGMConfig:
    """Configuration for CGM processing and model hyperparameters"""
    # Data processing
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
    
    # Model Architecture
    d_model_cgm: int = 128
    d_model_wearable: int = 64
    d_model_fusion: int = 128
    n_attention_heads: int = 8
    n_attention_layers: int = 3
    
    # Regularization
    dropout_rate: float = 0.1

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

class AdvancedCGMDataset(Dataset):
    def __init__(self, config: CGMConfig, mode='train', dataset_name: str = None, dataframe: Optional[pd.DataFrame] = None):
        self.config = config
        self.mode = mode
        
        if dataframe is not None:
            raw_data = dataframe
        elif dataset_name is not None:
            self.adapter = AwesomeCGMAdapter()
            raw_data = self.adapter.load_dataset(dataset_name)
            if raw_data is None:
                raise ValueError(f"Could not load dataset {dataset_name}")
            raw_data = self.adapter.prepare_for_model(raw_data)
        else:
            raise ValueError("Either dataset_name or dataframe must be provided.")
        
        # Continue with original processing...
        self.data = self._engineer_features(raw_data)
        self.sequences = self._create_sequences()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract 50+ features from CGM signal"""
        print("Columns at start of _engineer_features:", df.columns)
        print("Glucose head:", df['glucose'].head())
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
        df['mage'] = df.groupby('subject_id')['glucose'].transform(lambda x: gv.calculate_mage(x.tolist()))
        
        modd_series = df.groupby('subject_id').apply(lambda x: gv.calculate_modd(x['glucose'].tolist(), x['timestamp'].tolist()))
        df = df.merge(modd_series.rename('modd'), left_on='subject_id', right_index=True)

        df['conga'] = df.groupby('subject_id')['glucose'].transform(lambda x: gv.calculate_conga(x.tolist()))
        
        # Calculate LBGI and HBGI per-reading
        risk_df = gv.calculate_lbgi_hbgi(df['glucose'])
        df['lbgi'] = risk_df['lbgi']
        df['hbgi'] = risk_df['hbgi']
        
        # Calculate summary risk metrics per subject
        df['adrr'] = df.groupby('subject_id')['glucose'].transform(lambda x: gv.calculate_adrr(x))
        df['j_index'] = df.groupby('subject_id')['glucose'].transform(lambda x: gv.calculate_j_index(x.tolist()))
        
        # Frequency domain features (FFT)
        if self.config.use_fft:
            df = self._add_fft_features(df)

        # Wavelet features
        if self.config.use_wavelets:
            df = self._add_wavelet_features(df)
            
        # Trend arrows
        df = ta.process_cgm_data_with_trends(df)
        
        # Wearable data features (optional)
        wearable_cols = ['hr', 'hrv', 'resp_rate', 'skin_temp', 'spo2', 'accel_x', 'accel_y', 'accel_z']
        for col in wearable_cols:
            if col in df.columns:
                # Fill missing values
                df[col] = df.groupby('subject_id')[col].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
                
                # Calculate rolling statistics
                for window in [6, 12, 24, 48]:
                    df[f'{col}_mean_{window}'] = df.groupby('subject_id')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    df[f'{col}_std_{window}'] = df.groupby('subject_id')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
        
        return df

    def _create_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor, dict]]:
        """Create multi-scale sequences with metadata"""
        sequences = []
        
        # Define categorical and non-feature columns
        non_numeric_features = ['trend_arrow', 'trend_description', 'predicted_30min_change']
        id_and_target_cols = ['subject_id', 'timestamp', 'glucose']
        
        # Get only the numeric feature columns for scaling
        feature_cols = [
            col for col in self.data.columns 
            if col not in id_and_target_cols + non_numeric_features
        ]

        for subject_id in self.data['subject_id'].unique():
            subject_data = self.data[self.data['subject_id'] == subject_id].copy()
            
            # Skip if too short
            if len(subject_data) < self.config.sequence_length + self.config.prediction_horizon:
                continue
                
            # Create overlapping windows with stride
            stride = 3 if self.mode == 'train' else 6  # More augmentation for training
            
            # Scale features
            scaler = RobustScaler()
            # Handle potential NaN values in feature columns before scaling
            subject_data[feature_cols] = subject_data[feature_cols].fillna(0)
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

                # Separate CGM and wearable features
                wearable_base_cols = ['hr', 'hrv', 'resp_rate', 'skin_temp', 'spo2', 'accel_x', 'accel_y', 'accel_z', 'sleep_stage']
                wearable_feature_cols = [col for col in feature_cols if any(base in col for base in wearable_base_cols)]
                cgm_feature_cols = [col for col in feature_cols if col not in wearable_feature_cols]

                cgm_features = torch.tensor(seq_data[cgm_feature_cols].values, dtype=torch.float32)
                
                # Create a placeholder tensor for wearable features if they don't exist.
                # This is crucial for the default collate function of the DataLoader.
                wearable_features = torch.empty(cgm_features.shape[0], 0)
                if wearable_feature_cols:
                    wearable_features = torch.tensor(seq_data[wearable_feature_cols].values, dtype=torch.float32)

                # Metadata - ensure all values are collate-able
                metadata = {
                    'subject_id': subject_id,
                    'timestamp': str(seq_data.iloc[-1]['timestamp']), # Convert timestamp to string
                    'hour': seq_data.iloc[-1]['hour'],
                    'day_of_week': seq_data.iloc[-1]['day_of_week']
                }
                
                sequences.append((cgm_features, wearable_features, targets, metadata))

        return sequences

    def _add_fft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add FFT features"""
        
        def get_fft_features(x):
            from scipy.fft import fft
            
            # Perform FFT
            fft_vals = fft(x.values)
            
            # Get dominant frequencies and their magnitudes
            fft_freq = np.fft.fftfreq(len(x), d=self.config.sampling_rate)
            
            # Get the absolute values of the FFT
            fft_abs = np.abs(fft_vals)
            
            # Find the top 3 dominant frequencies
            sorted_indices = np.argsort(fft_abs)[::-1]
            
            features = {}
            for i in range(1, 4): # Get top 3 frequencies (index 0 is the DC component)
                if i < len(sorted_indices):
                    idx = sorted_indices[i]
                    features[f'fft_freq_{i}'] = fft_freq[idx]
                    features[f'fft_mag_{i}'] = fft_abs[idx]
                else:
                    features[f'fft_freq_{i}'] = 0
                    features[f'fft_mag_{i}'] = 0
            return pd.Series(features)

        fft_features_series = df.groupby('subject_id')['glucose'].apply(get_fft_features)
        fft_features_df = fft_features_series.unstack()
        df = df.merge(fft_features_df, left_on='subject_id', right_index=True)
        
        return df

    def _add_wavelet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add wavelet features"""
        import pywt

        def get_wavelet_features(x):
            if len(x) < 4: # pywt.dwt requires a minimum length
                return pd.Series({
                    'wavelet_cA_mean': 0, 'wavelet_cA_std': 0, 'wavelet_cA_energy': 0,
                    'wavelet_cD_mean': 0, 'wavelet_cD_std': 0, 'wavelet_cD_energy': 0,
                })

            coeffs = pywt.dwt(x.values, 'db4')
            cA, cD = coeffs
            
            features = {
                'wavelet_cA_mean': np.mean(cA),
                'wavelet_cA_std': np.std(cA),
                'wavelet_cA_energy': np.sum(np.square(cA)),
                'wavelet_cD_mean': np.mean(cD),
                'wavelet_cD_std': np.std(cD),
                'wavelet_cD_energy': np.sum(np.square(cD)),
            }
            return pd.Series(features)

        wavelet_features_series = df.groupby('subject_id')['glucose'].apply(get_wavelet_features)
        wavelet_features_df = wavelet_features_series.unstack()
        df = df.merge(wavelet_features_df, left_on='subject_id', right_index=True)
        
        return df
