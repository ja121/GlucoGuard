import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

class BIGIDEAsDataLoader:
    """Loader for BIG IDEAs Lab dataset with E4 and CGM data"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def load_participant(self, participant_id: str) -> Dict:
        """Load all data for a single participant"""
        
        data = {}
        
        # Load CGM data (Dexcom G6)
        cgm_path = self.base_path / f"{participant_id}/cgm.csv"
        if cgm_path.exists():
            data['cgm'] = pd.read_csv(cgm_path, parse_dates=['timestamp'])
            data['cgm'].set_index('timestamp', inplace=True)
        
        # Load E4 data
        e4_path = self.base_path / f"{participant_id}/e4"
        
        # EDA (Electrodermal Activity)
        eda_path = e4_path / "EDA.csv"
        if eda_path.exists():
            data['eda'] = self._load_e4_sensor(eda_path, sample_rate=4)
        
        # BVP (Blood Volume Pulse)
        bvp_path = e4_path / "BVP.csv"
        if bvp_path.exists():
            data['bvp'] = self._load_e4_sensor(bvp_path, sample_rate=64)
        
        # Temperature
        temp_path = e4_path / "TEMP.csv"
        if temp_path.exists():
            data['temp'] = self._load_e4_sensor(temp_path, sample_rate=4)
        
        # Accelerometer
        acc_path = e4_path / "ACC.csv"
        if acc_path.exists():
            data['acc'] = self._load_e4_accelerometer(acc_path)
        
        # Food logs (if available)
        food_path = self.base_path / f"{participant_id}/food_log.csv"
        if food_path.exists():
            data['food'] = pd.read_csv(food_path, parse_dates=['timestamp'])
        
        return data
    
    def _load_e4_sensor(self, path: Path, sample_rate: int) -> pd.DataFrame:
        """Load E4 sensor data with timestamp reconstruction"""
        
        df = pd.read_csv(path, header=None)
        
        # First row is Unix timestamp start
        start_time = pd.to_datetime(df.iloc[0, 0], unit='s')
        
        # Rest is sensor data
        values = df.iloc[1:, 0].values
        
        # Create timestamp index
        timestamps = pd.date_range(
            start=start_time,
            periods=len(values),
            freq=f'{1000//sample_rate}ms'
        )
        
        return pd.DataFrame({'value': values}, index=timestamps)
    
    def _load_e4_accelerometer(self, path: Path) -> pd.DataFrame:
        """Load 3-axis accelerometer data"""
        
        df = pd.read_csv(path, header=None)
        
        # First row is Unix timestamp
        start_time = pd.to_datetime(df.iloc[0, 0], unit='s')
        
        # Rest is x, y, z acceleration
        acc_data = df.iloc[1:, :].values
        
        # Create timestamp index (32 Hz)
        timestamps = pd.date_range(
            start=start_time,
            periods=len(acc_data),
            freq='31.25ms'  # 1000/32 ms
        )
        
        return pd.DataFrame(
            acc_data,
            columns=['x', 'y', 'z'],
            index=timestamps
        )
    
    def align_data(self, participant_data: Dict, target_freq: str = '5min') -> pd.DataFrame:
        """Align all data streams to CGM frequency"""
        
        aligned = participant_data['cgm'].copy()
        
        # Resample E4 data to CGM frequency
        if 'eda' in participant_data:
            eda_resampled = participant_data['eda'].resample(target_freq).agg({
                'value': ['mean', 'std', 'max', 'min']
            })
            eda_resampled.columns = ['eda_mean', 'eda_std', 'eda_max', 'eda_min']
            aligned = aligned.join(eda_resampled)
        
        if 'bvp' in participant_data:
            # Extract HRV features from BVP
            hrv_features = self._extract_hrv_features(
                participant_data['bvp'], 
                target_freq
            )
            aligned = aligned.join(hrv_features)
        
        if 'temp' in participant_data:
            temp_resampled = participant_data['temp'].resample(target_freq).agg({
                'value': ['mean', 'std', 'min', 'max']
            })
            temp_resampled.columns = ['temp_mean', 'temp_std', 'temp_min', 'temp_max']
            aligned = aligned.join(temp_resampled)
        
        if 'acc' in participant_data:
            acc_features = self._extract_activity_features(
                participant_data['acc'],
                target_freq
            )
            aligned = aligned.join(acc_features)
        
        return aligned
    
    def _extract_hrv_features(self, bvp_data: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        """Extract HRV metrics from BVP signal"""
        
        features = []
        
        for timestamp in pd.date_range(
            start=bvp_data.index[0],
            end=bvp_data.index[-1],
            freq=target_freq
        ):
            window = bvp_data[timestamp:timestamp + pd.Timedelta(target_freq)]
            
            if len(window) > 0:
                # Simple HRV metrics
                features.append({
                    'timestamp': timestamp,
                    'hr_mean': window['value'].mean(),
                    'hrv_std': window['value'].std(),
                    'hrv_rmssd': np.sqrt(np.mean(np.diff(window['value'])**2))
                })
        
        return pd.DataFrame(features).set_index('timestamp')
    
    def _extract_activity_features(self, acc_data: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        """Extract activity metrics from accelerometer"""
        
        # Calculate magnitude
        acc_data['magnitude'] = np.sqrt(
            acc_data['x']**2 + 
            acc_data['y']**2 + 
            acc_data['z']**2
        )
        
        # Resample
        activity_features = acc_data.resample(target_freq).agg({
            'magnitude': ['mean', 'std', 'max'],
            'x': 'std',
            'y': 'std',
            'z': 'std'
        })
        
        activity_features.columns = [
            'activity_level', 'activity_variance', 'activity_peak',
            'movement_x', 'movement_y', 'movement_z'
        ]
        
        return activity_features
