import pandas as pd
import numpy as np
from scipy import stats
from src.config import GlucoGuardConfig

class CGMFeatureEngineer:
    """Extract advanced features from CGM data"""
    
    def __init__(self):
        self.config = GlucoGuardConfig()
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive CGM features"""
        
        if 'glucose' not in df.columns:
            raise ValueError("DataFrame must contain 'glucose' column")
        
        glucose = df['glucose']
        features = pd.DataFrame(index=df.index)
        
        # Trend arrows
        features['trend_arrow'] = self.calculate_trend_arrows(glucose)
        
        # Basic statistics (multiple windows)
        for window in [6, 12, 24, 48]:  # 30min, 1hr, 2hr, 4hr
            features[f'glucose_mean_{window}'] = glucose.rolling(window).mean()
            features[f'glucose_std_{window}'] = glucose.rolling(window).std()
            features[f'glucose_cv_{window}'] = features[f'glucose_std_{window}'] / (features[f'glucose_mean_{window}'] + 1e-8)
        
        # Rate of change
        features['roc_5min'] = glucose.diff(1)
        features['roc_15min'] = glucose.diff(3)
        features['roc_30min'] = glucose.diff(6)
        features['roc_60min'] = glucose.diff(12)
        
        # Acceleration
        features['acceleration'] = features['roc_5min'].diff()
        
        # Glycemic variability metrics
        variability_metrics = self.calculate_glycemic_variability(glucose)
        features = pd.concat([features, variability_metrics], axis=1)
        
        # Time in ranges
        tir_metrics = self.calculate_time_in_range(glucose)
        for key, value in tir_metrics.items():
            features[key] = value
        
        # Risk indices
        features['lbgi'] = self.calculate_lbgi(glucose)
        features['hbgi'] = self.calculate_hbgi(glucose)
        features['adrr'] = features['lbgi'] + features['hbgi']
        
        # Pattern features
        features['glucose_increasing'] = (features['roc_5min'] > 0).astype(float)
        features['glucose_decreasing'] = (features['roc_5min'] < 0).astype(float)
        features['glucose_stable'] = (abs(features['roc_5min']) < 1).astype(float)
        
        return features
    
    def calculate_trend_arrows(self, glucose: pd.Series, window: int = 6) -> pd.Series:
        """Calculate Dexcom G6 style trend arrows"""
        
        trends = pd.Series(0, index=glucose.index)
        
        for i in range(window, len(glucose)):
            rate = (glucose.iloc[i] - glucose.iloc[i-window]) / (window * 5)
            
            if rate > 3:
                trends.iloc[i] = 3  # 🔼🔼🔼
            elif rate > 2:
                trends.iloc[i] = 2  # 🔼🔼
            elif rate > 1:
                trends.iloc[i] = 1  # 🔼
            elif rate < -3:
                trends.iloc[i] = -3  # 🔽🔽🔽
            elif rate < -2:
                trends.iloc[i] = -2  # 🔽🔽
            elif rate < -1:
                trends.iloc[i] = -1  # 🔽
            else:
                trends.iloc[i] = 0  # ➡️
        
        return trends
    
    def calculate_glycemic_variability(self, glucose: pd.Series) -> pd.DataFrame:
        """Calculate advanced variability metrics"""
        
        features = pd.DataFrame(index=glucose.index)
        
        # MAGE (Mean Amplitude of Glycemic Excursions)
        features['mage'] = self._calculate_mage(glucose)
        
        # CONGA (Continuous Overall Net Glycemic Action)
        for n in [1, 2, 4]:
            features[f'conga_{n}'] = self._calculate_conga(glucose, n)
        
        # MODD (Mean of Daily Differences)
        features['modd'] = self._calculate_modd(glucose)
        
        # J-index
        mean_g = glucose.rolling(288).mean()  # Daily mean
        std_g = glucose.rolling(288).std()
        features['j_index'] = 0.001 * (mean_g + 3 * std_g) ** 2
        
        # M-value
        features['m_value'] = self._calculate_m_value(glucose)
        
        return features
    
    def calculate_time_in_range(self, glucose: pd.Series) -> dict:
        """Calculate time in various glucose ranges"""
        
        window = 288  # 24 hours
        
        tir = {}
        
        for i in range(window, len(glucose)):
            window_data = glucose.iloc[i-window:i]
            total = len(window_data)
            
            tir_values = {
                'tir_severe_hypo': np.sum(window_data < 54) / total * 100,
                'tir_hypo': np.sum(window_data < 70) / total * 100,
                'tir_target': np.sum((window_data >= 70) & (window_data <= 180)) / total * 100,
                'tir_hyper': np.sum(window_data > 180) / total * 100,
                'tir_severe_hyper': np.sum(window_data > 250) / total * 100
            }
            
            for key, value in tir_values.items():
                if key not in tir:
                    tir[key] = pd.Series(index=glucose.index, dtype=float)
                tir[key].iloc[i] = value
        
        return tir
    
    def calculate_lbgi(self, glucose: pd.Series) -> pd.Series:
        """Low Blood Glucose Index"""
        
        f_glucose = 1.509 * (np.log(glucose + 1e-8)**1.084 - 5.381)
        rl = np.where(f_glucose < 0, 10 * f_glucose**2, 0)
        
        lbgi = pd.Series(index=glucose.index, dtype=float)
        
        for i in range(288, len(glucose)):
            lbgi.iloc[i] = np.mean(rl[i-288:i])
        
        return lbgi
    
    def calculate_hbgi(self, glucose: pd.Series) -> pd.Series:
        """High Blood Glucose Index"""
        
        f_glucose = 1.509 * (np.log(glucose + 1e-8)**1.084 - 5.381)
        rh = np.where(f_glucose > 0, 10 * f_glucose**2, 0)
        
        hbgi = pd.Series(index=glucose.index, dtype=float)
        
        for i in range(288, len(glucose)):
            hbgi.iloc[i] = np.mean(rh[i-288:i])
        
        return hbgi
    
    def _calculate_mage(self, glucose: pd.Series) -> pd.Series:
        """Mean Amplitude of Glycemic Excursions"""
        
        mage = pd.Series(index=glucose.index, dtype=float)
        window = 288  # 24 hours
        
        for i in range(window, len(glucose)):
            window_data = glucose.iloc[i-window:i].values
            
            # Find peaks and troughs
            excursions = []
            threshold = np.std(window_data)
            
            for j in range(1, len(window_data)-1):
                if window_data[j] > window_data[j-1] and window_data[j] > window_data[j+1]:
                    # Peak
                    if abs(window_data[j] - window_data[j-1]) > threshold:
                        excursions.append(abs(window_data[j] - window_data[j-1]))
                elif window_data[j] < window_data[j-1] and window_data[j] < window_data[j+1]:
                    # Trough
                    if abs(window_data[j] - window_data[j-1]) > threshold:
                        excursions.append(abs(window_data[j] - window_data[j-1]))
            
            mage.iloc[i] = np.mean(excursions) if excursions else 0
        
        return mage
    
    def _calculate_conga(self, glucose: pd.Series, n: int) -> pd.Series:
        """Continuous Overall Net Glycemic Action"""
        
        conga = pd.Series(index=glucose.index, dtype=float)
        lag = n * 12  # n hours in 5-min intervals
        
        for i in range(lag, len(glucose)):
            diff = glucose.iloc[i] - glucose.iloc[i-lag]
            conga.iloc[i] = np.sqrt(np.mean(diff**2)) if not np.isnan(diff) else np.nan
        
        return conga
    
    def _calculate_modd(self, glucose: pd.Series) -> pd.Series:
        """Mean of Daily Differences"""
        
        modd = pd.Series(index=glucose.index, dtype=float)
        day_lag = 288  # 24 hours
        
        for i in range(day_lag, len(glucose)):
            diff = abs(glucose.iloc[i] - glucose.iloc[i-day_lag])
            modd.iloc[i] = diff
        
        return modd.rolling(288).mean()
    
    def _calculate_m_value(self, glucose: pd.Series) -> pd.Series:
        """M-value for glucose control quality"""
        
        ideal_glucose = 120  # mg/dL
        m_value = pd.Series(index=glucose.index, dtype=float)
        
        for i in range(len(glucose)):
            deviation = abs(10 * np.log10(glucose.iloc[i] / ideal_glucose))
            m_value.iloc[i] = deviation ** 3
        
        return m_value.rolling(288).mean()



        
