import pandas as pd
import numpy as np
from scipy import signal
from src.config import GlucoGuardConfig

class EmpaticaE4Processor:
    """Process Empatica E4 wearable sensor data"""
    
    def __init__(self):
        self.config = GlucoGuardConfig()
        
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all E4 sensors from dataframe"""
        
        features = pd.DataFrame(index=df.index)
        
        # Process EDA (stress detection)
        if 'eda' in df.columns:
            eda_features = self.process_eda(df['eda'])
            features = pd.concat([features, eda_features], axis=1)
        
        # Process BVP/HR (cardiac features)
        if 'bvp' in df.columns:
            hrv_features = self.process_bvp(df['bvp'])
            features = pd.concat([features, hrv_features], axis=1)
        elif 'heart_rate' in df.columns:
            features['hr_mean'] = df['heart_rate']
            features['hrv_estimated'] = df['heart_rate'].rolling(12).std()
        
        # Process temperature
        if 'temp' in df.columns:
            temp_features = self.process_temperature(df['temp'])
            features = pd.concat([features, temp_features], axis=1)
        
        # Process accelerometer
        acc_columns = ['acc_x', 'acc_y', 'acc_z']
        if all(col in df.columns for col in acc_columns):
            activity_features = self.process_accelerometer(df[acc_columns])
            features = pd.concat([features, activity_features], axis=1)
        
        # Calculate composite stress score
        features['stress_level'] = self._calculate_stress_score(features)
        
        # Detect context
        features['context'] = self._detect_context(features)
        
        return features
    
    def process_eda(self, eda_series: pd.Series) -> pd.DataFrame:
        """Extract features from electrodermal activity"""
        
        features = pd.DataFrame(index=eda_series.index)
        
        # Basic statistics
        features['eda_mean'] = eda_series.rolling(12).mean()
        features['eda_std'] = eda_series.rolling(12).std()
        
        # Phasic component (rapid changes)
        eda_clean = eda_series.fillna(method='ffill')
        
        # Decompose into tonic and phasic
        sos = signal.butter(4, 0.05, 'hp', fs=self.config.e4_eda_hz, output='sos')
        phasic = signal.sosfiltfilt(sos, eda_clean)
        
        features['eda_phasic'] = pd.Series(phasic, index=eda_series.index)
        features['eda_tonic'] = eda_series - features['eda_phasic']
        
        # Peak detection for stress events
        features['eda_peaks'] = self._detect_eda_peaks(features['eda_phasic'])
        
        # Stress probability
        features['stress_probability'] = 1 / (1 + np.exp(-0.5 * (features['eda_phasic'] - 2)))
        
        return features
    
    def process_bvp(self, bvp_series: pd.Series) -> pd.DataFrame:
        """Extract HRV features from blood volume pulse"""
        
        features = pd.DataFrame(index=bvp_series.index)
        
        # Estimate heart rate
        features['hr_mean'] = 60 * self.config.e4_bvp_hz / (bvp_series.rolling(64).std() + 1e-6)
        
        # HRV metrics
        features['hrv_rmssd'] = self._calculate_rmssd(bvp_series)
        features['hrv_sdnn'] = bvp_series.rolling(320).std()  # 5 seconds at 64Hz
        
        # Frequency domain HRV
        features['hrv_lf_hf_ratio'] = self._calculate_lf_hf_ratio(bvp_series)
        
        # Cardiac stress indicator
        features['cardiac_stress'] = (100 - features['hrv_rmssd']) / 100
        
        return features
    
    def process_temperature(self, temp_series: pd.Series) -> pd.DataFrame:
        """Extract temperature features for circadian rhythm"""
        
        features = pd.DataFrame(index=temp_series.index)
        
        features['temp_mean'] = temp_series.rolling(12).mean()
        features['temp_std'] = temp_series.rolling(12).std()
        features['temp_slope'] = temp_series.diff().rolling(12).mean()
        
        # Circadian phase estimation
        features['temp_normalized'] = (temp_series - temp_series.mean()) / temp_series.std()
        features['circadian_phase'] = np.sin(2 * np.pi * features['temp_normalized'])
        
        # Fever detection
        features['fever_risk'] = (temp_series > 37.5).astype(float)
        
        return features
    
    def process_accelerometer(self, acc_df: pd.DataFrame) -> pd.DataFrame:
        """Extract activity and movement features"""
        
        features = pd.DataFrame(index=acc_df.index)
        
        # Calculate magnitude
        magnitude = np.sqrt(acc_df['acc_x']**2 + acc_df['acc_y']**2 + acc_df['acc_z']**2)
        
        features['activity_level'] = magnitude.rolling(32).mean()
        features['activity_variance'] = magnitude.rolling(32).std()
        
        # Detect activity type
        features['activity_intensity'] = pd.cut(
            features['activity_level'],
            bins=[0, 0.5, 2, 5, 100],
            labels=['rest', 'light', 'moderate', 'vigorous']
        )
        
        # Step counting (simplified)
        features['steps_estimate'] = self._count_steps(magnitude)
        
        # Sleep probability
        features['sleep_probability'] = 1 / (1 + np.exp(features['activity_level'] - 0.3))
        
        # Fall detection
        features['fall_risk'] = (magnitude.diff().abs() > 10).astype(float)
        
        return features
    
    def _calculate_rmssd(self, bvp: pd.Series) -> pd.Series:
        """Calculate RMSSD for HRV"""
        
        diffs = bvp.diff()
        squared_diffs = diffs ** 2
        mean_squared = squared_diffs.rolling(320).mean()  # 5 seconds
        rmssd = np.sqrt(mean_squared)
        
        return rmssd
    
    def _calculate_lf_hf_ratio(self, bvp: pd.Series) -> pd.Series:
        """Calculate LF/HF ratio from BVP signal"""
        
        # Simplified frequency analysis
        # In production, use proper FFT with windowing
        
        rolling_std_fast = bvp.rolling(32).std()  # ~0.5 seconds (HF)
        rolling_std_slow = bvp.rolling(320).std()  # ~5 seconds (LF)
        
        ratio = rolling_std_slow / (rolling_std_fast + 1e-6)
        
        return ratio
    
    def _detect_eda_peaks(self, phasic_eda: pd.Series) -> pd.Series:
        """Detect significant EDA peaks"""
        
        threshold = phasic_eda.mean() + 2 * phasic_eda.std()
        peaks = (phasic_eda > threshold).astype(float)
        
        return peaks
    
    def _count_steps(self, magnitude: pd.Series) -> pd.Series:
        """Estimate step count from accelerometer magnitude"""
        
        # Simple peak detection for steps
        # Threshold based on typical walking pattern
        
        threshold = magnitude.mean() + magnitude.std()
        peaks = (magnitude > threshold) & (magnitude.shift(1) <= threshold)
        
        # Count peaks in windows
        step_count = peaks.rolling(160).sum()  # 5 seconds at 32Hz
        
        return step_count
    
    def _calculate_stress_score(self, features: pd.DataFrame) -> pd.Series:
        """Calculate composite stress score from all features"""
        
        stress_score = pd.Series(5.0, index=features.index)  # Default moderate stress
        
        if 'stress_probability' in features.columns:
            stress_score += features['stress_probability'] * 3
        
        if 'cardiac_stress' in features.columns:
            stress_score += features['cardiac_stress'] * 2
        
        if 'activity_level' in features.columns:
            # High activity can be stress or exercise
            stress_score += (features['activity_level'] > 3).astype(float)
        
        # Normalize to 0-10 scale
        stress_score = np.clip(stress_score, 0, 10)
        
        return stress_score
    
    def _detect_context(self, features: pd.DataFrame) -> pd.Series:
        """Detect current context (rest, active, sleep, stress)"""
        
        context = pd.Series('unknown', index=features.index)
        
        if 'sleep_probability' in features.columns:
            context[features['sleep_probability'] > 0.7] = 'sleep'
        
        if 'activity_intensity' in features.columns:
            context[features['activity_intensity'] == 'vigorous'] = 'exercise'
            context[features['activity_intensity'] == 'moderate'] = 'active'
            context[features['activity_intensity'] == 'rest'] = 'rest'
        
        if 'stress_level' in features.columns:
            context[features['stress_level'] > 7] = 'high_stress'
        
        return context
