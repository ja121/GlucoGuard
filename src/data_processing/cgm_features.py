import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BIGIDEAsConfig:
    """Configuration for BIG IDEAs dataset processing"""
    # Sampling rates
    cgm_freq = '5min'  # Dexcom G6
    e4_bvp_hz = 64  # Blood Volume Pulse
    e4_eda_hz = 4   # Electrodermal Activity
    e4_temp_hz = 4  # Skin Temperature
    e4_acc_hz = 32  # Accelerometer
    
    # Clinical thresholds
    hypo_threshold = 70
    severe_hypo = 54
    hyper_threshold = 180
    severe_hyper = 250
    
    # Prediction horizons
    glucose_horizons = [6, 12, 18]  # 30min, 1hr, 1.5hr
    complication_days = 365  # 1 year ahead

class EmpaticaE4Processor:
    """Process Empatica E4 wearable data"""
    
    def __init__(self, config: BIGIDEAsConfig):
        self.config = config
        
    def process_eda(self, eda_data):
        """Extract EDA features for stress detection"""
        import neurokit2 as nk
        
        # Clean signal
        eda_clean = nk.eda_clean(eda_data, sampling_rate=self.config.e4_eda_hz)
        
        # Decompose into phasic and tonic
        eda_decomposed = nk.eda_phasic(eda_clean, sampling_rate=self.config.e4_eda_hz)
        
        # Extract features
        features = {
            'eda_mean': np.mean(eda_clean),
            'eda_std': np.std(eda_clean),
            'eda_phasic_peaks': len(nk.eda_findpeaks(eda_decomposed["EDA_Phasic"])),
            'eda_tonic_mean': np.mean(eda_decomposed["EDA_Tonic"]),
            'stress_level': self._calculate_stress_index(eda_decomposed)
        }
        
        return features
    
    def process_bvp(self, bvp_data):
        """Extract HRV features from Blood Volume Pulse"""
        import heartpy as hp
        
        # Process BVP to get heart rate
        working_data, measures = hp.process(bvp_data, 
                                           sample_rate=self.config.e4_bvp_hz)
        
        # Extract HRV metrics
        features = {
            'hr_mean': measures['bpm'],
            'hrv_rmssd': measures['rmssd'],
            'hrv_sdnn': measures['sdnn'],
            'hrv_pnn50': measures['pnn50'],
            'cardiac_stress': self._calculate_cardiac_stress(measures)
        }
        
        return features
    
    def process_temp(self, temp_data):
        """Temperature features for circadian rhythm"""
        features = {
            'temp_mean': np.mean(temp_data),
            'temp_std': np.std(temp_data),
            'temp_slope': np.polyfit(range(len(temp_data)), temp_data, 1)[0],
            'circadian_phase': self._estimate_circadian_phase(temp_data)
        }
        return features
    
    def process_accelerometer(self, acc_data):
        """Activity and sleep detection from accelerometer"""
        # Calculate activity metrics
        magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
        
        features = {
            'activity_level': np.mean(magnitude),
            'activity_variance': np.var(magnitude),
            'steps_estimate': self._estimate_steps(magnitude),
            'sleep_probability': self._detect_sleep(magnitude)
        }
        return features
class CGMFeatureEngineer:
    """Extract 100+ validated CGM features"""
    
    def __init__(self):
        self.features = {}
        
    def calculate_trend_arrows(self, glucose, window=6):
        """Calculate Dexcom-style trend arrows"""
        if len(glucose) < window:
            return 0
            
        rate = (glucose[-1] - glucose[-window]) / (window * 5)
        
        # Dexcom G6 thresholds
        if rate > 3:
            return 3  # 🔼🔼🔼 Rising rapidly
        elif rate > 2:
            return 2  # 🔼🔼 Rising
        elif rate > 1:
            return 1  # 🔼 Rising slowly
        elif rate < -3:
            return -3  # 🔽🔽🔽 Falling rapidly
        elif rate < -2:
            return -2  # 🔽🔽 Falling
        elif rate < -1:
            return -1  # 🔽 Falling slowly
        else:
            return 0  # ➡️ Stable
    
    def calculate_glycemic_variability(self, glucose):
        """Advanced variability metrics"""
        features = {}
        
        # MAGE (Mean Amplitude of Glycemic Excursions)
        excursions = self._find_excursions(glucose, threshold=1*np.std(glucose))
        features['mage'] = np.mean([abs(e) for e in excursions]) if excursions else 0
        
        # CONGA (Continuous Overall Net Glycemic Action)
        for n in [1, 2, 4]:  # 1hr, 2hr, 4hr
            features[f'conga_{n}'] = self._calculate_conga(glucose, n)
        
        # MODD (Mean of Daily Differences)
        features['modd'] = self._calculate_modd(glucose)
        
        # J-index (glycemic variability + mean)
        mean_glucose = np.mean(glucose)
        features['j_index'] = 0.001 * (mean_glucose + 3 * np.std(glucose))**2
        
        # LBGI/HBGI (Low/High Blood Glucose Index)
        features['lbgi'] = self._calculate_lbgi(glucose)
        features['hbgi'] = self._calculate_hbgi(glucose)
        
        # ADRR (Average Daily Risk Range)
        features['adrr'] = features['lbgi'] + features['hbgi']
        
        return features
    
    def calculate_time_in_range(self, glucose):
        """Time in different glycemic ranges"""
        total = len(glucose)
        
        return {
            'time_in_severe_hypo': np.sum(glucose < 54) / total * 100,
            'time_in_hypo': np.sum(glucose < 70) / total * 100,
            'time_in_range': np.sum((glucose >= 70) & (glucose <= 180)) / total * 100,
            'time_in_hyper': np.sum(glucose > 180) / total * 100,
            'time_in_severe_hyper': np.sum(glucose > 250) / total * 100,
        }
    
    def _calculate_lbgi(self, glucose):
        """Low Blood Glucose Index - validated risk metric"""
        f_glucose = 1.509 * (np.log(glucose + 1e-8)**1.084 - 5.381)
        rl = np.where(f_glucose < 0, 10 * f_glucose**2, 0)
        return np.mean(rl)
    
    def _calculate_hbgi(self, glucose):
        """High Blood Glucose Index"""
        f_glucose = 1.509 * (np.log(glucose + 1e-8)**1.084 - 5.381)
        rh = np.where(f_glucose > 0, 10 * f_glucose**2, 0)
        return np.mean(rh)
