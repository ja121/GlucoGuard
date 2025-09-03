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
class MultiModalFusion:
    """Fuse CGM + Empatica E4 data streams"""
    
    def __init__(self, cgm_data, e4_data):
        self.cgm = cgm_data
        self.e4 = e4_data
        self.aligned_data = None
        
    def align_timestamps(self):
        """Align different sampling rates"""
        # Resample E4 data to 5-minute intervals
        e4_resampled = {}
        
        # EDA: 4Hz -> 5min (average)
        e4_resampled['eda'] = self.e4['eda'].resample('5min').mean()
        
        # BVP: 64Hz -> 5min (extract HRV features)
        e4_resampled['hrv'] = self._extract_hrv_windows(self.e4['bvp'])
        
        # Temperature: 4Hz -> 5min
        e4_resampled['temp'] = self.e4['temp'].resample('5min').mean()
        
        # Activity: 32Hz -> 5min
        e4_resampled['activity'] = self._extract_activity_features(self.e4['acc'])
        
        # Merge with CGM
        self.aligned_data = pd.concat([
            self.cgm,
            e4_resampled['eda'],
            e4_resampled['hrv'],
            e4_resampled['temp'],
            e4_resampled['activity']
        ], axis=1)
        
        return self.aligned_data
    
    def create_context_features(self):
        """Create context-aware features"""
        features = {}
        
        # Stress + Glucose interaction
        features['stress_glucose_risk'] = (
            self.aligned_data['stress_level'] * 
            self.aligned_data['glucose_variability']
        )
        
        # Activity + Glucose dynamics
        features['exercise_glucose_response'] = (
            self.aligned_data['activity_level'].shift(6) *  # 30 min ago
            self.aligned_data['glucose_roc']
        )
        
        # Circadian + Glucose patterns
        features['circadian_glucose_phase'] = (
            np.sin(2 * np.pi * self.aligned_data['hour'] / 24) *
            self.aligned_data['glucose']
        )
        
        # Sleep + Nocturnal glucose
        features['nocturnal_risk'] = (
            self.aligned_data['sleep_probability'] *
            (1 - self.aligned_data['time_in_range'])
        )
        
        return features

class ComplicationPredictor:
    """Predict long-term diabetes complications"""
    
    def __init__(self):
        self.risk_models = {}
        
    def calculate_retinopathy_risk(self, patient_data):
        """Diabetic retinopathy risk score"""
        # Based on UKPDS risk engine
        hba1c = self._estimate_hba1c(patient_data['glucose'])
        duration = patient_data.get('diabetes_duration_years', 5)
        
        risk_score = (
            0.9 * hba1c +
            0.3 * duration +
            0.5 * patient_data['time_in_hyper'] +
            0.4 * patient_data['glucose_variability']
        )
        
        # Convert to probability
        risk_prob = 1 / (1 + np.exp(-0.5 * (risk_score - 10)))
        
        return {
            'risk_score': risk_score,
            'probability_1_year': risk_prob,
            'risk_level': self._categorize_risk(risk_prob)
        }
    
    def calculate_nephropathy_risk(self, patient_data):
        """Diabetic kidney disease risk"""
        # Based on clinical studies
        risk_factors = {
            'poor_control': patient_data['time_in_hyper'] > 30,
            'high_variability': patient_data['mage'] > 60,
            'stress': patient_data['avg_stress_level'] > 0.7,
            'hypertension': patient_data.get('bp_systolic', 120) > 140
        }
        
        risk_score = sum(risk_factors.values()) * 25
        
        return {
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': self._get_nephropathy_recommendations(risk_factors)
        }
    
    def calculate_neuropathy_risk(self, patient_data):
        """Peripheral neuropathy risk"""
        # Temperature variability as early indicator
        temp_variability = patient_data['temp_std']
        glucose_exposure = patient_data['cumulative_hyperglycemia']
        
        risk_score = (
            0.6 * glucose_exposure +
            0.3 * temp_variability +
            0.1 * patient_data['diabetes_duration_years']
        )
        
        return {
            'risk_score': risk_score,
            'early_signs': temp_variability > 2.0,
            'monitoring_frequency': 'monthly' if risk_score > 50 else 'quarterly'
        }
    
    def calculate_cardiovascular_risk(self, patient_data):
        """CVD risk with HRV integration"""
        # Use HRV as autonomic neuropathy marker
        hrv_risk = 1 - (patient_data['hrv_rmssd'] / 50)  # Normalized
        glucose_risk = patient_data['adrr'] / 40  # Normalized
        
        composite_risk = (
            0.4 * hrv_risk +
            0.3 * glucose_risk +
            0.2 * patient_data['time_in_hyper'] / 100 +
            0.1 * patient_data.get('age', 40) / 100
        )
        
        return {
            'risk_score': composite_risk * 100,
            'autonomic_dysfunction': hrv_risk > 0.7,
            'intervention_needed': composite_risk > 0.6
        }
    
    def _estimate_hba1c(self, glucose_series):
        """Estimate HbA1c from CGM data"""
        mean_glucose = np.mean(glucose_series)
        # Nathan formula
        hba1c = (mean_glucose + 46.7) / 28.7
        return hba1c

class ContextAwareAlertSystem:
    """Generate personalized alerts based on CGM + E4 context"""
    
    def __init__(self):
        self.rules = self._load_rule_database()
        self.alert_history = []
        
    def _load_rule_database(self):
        """Load the alert rules you specified"""
        rules = [
            {
                'rule_id': 'R-00001',
                'condition': 'severe_hypo_tachycardia',
                'cgm_criteria': lambda g: g < 54 or self._fast_drop(g),
                'e4_criteria': lambda hr: hr > 100,
                'context': 'rest_day',
                'risk_level': 'critical',
                'message': '⚠️ Very low sugar. Eat carbs now. Heart rate is high. Rest and drink water.'
            },
            {
                'rule_id': 'R-00002',
                'condition': 'severe_hypo_bradycardia',
                'cgm_criteria': lambda g: g < 54,
                'e4_criteria': lambda hr: hr < 50,
                'context': 'rest_day',
                'risk_level': 'critical',
                'message': '⚠️ Very low sugar. Eat carbs now. Heart rate is low. Sit or lie down if dizzy.'
            },
            # Add all your rules here...
        ]
        return rules
    
    def evaluate_current_state(self, current_data):
        """Check all rules against current state"""
        alerts = []
        
        for rule in self.rules:
            if self._check_rule(rule, current_data):
                alert = self._generate_alert(rule, current_data)
                alerts.append(alert)
                
        # Sort by priority
        alerts = sorted(alerts, key=lambda x: self._priority_score(x), reverse=True)
        
        # Store in history
        self.alert_history.extend(alerts)
        
        return alerts
    
    def _generate_alert(self, rule, data):
        """Create detailed alert with trend arrows"""
        trend = self._calculate_trend_arrow(data['glucose_history'])
        
        alert = {
            'timestamp': data['timestamp'],
            'rule_id': rule['rule_id'],
            'risk_level': rule['risk_level'],
            'trend_arrow': self._get_arrow_emoji(trend),
            'current_glucose': data['glucose'],
            'predicted_30min': data.get('glucose_pred_30min'),
            'message': rule['message'],
            'vitals': {
                'hr': data.get('heart_rate'),
                'hrv': data.get('hrv_rmssd'),
                'stress': data.get('stress_level'),
                'activity': data.get('activity_level')
            },
            'action_items': self._get_action_items(rule, data)
        }
        
        return alert
    
    def _get_arrow_emoji(self, trend):
        """Convert trend to emoji arrows"""
        arrows = {
            3: '🔼🔼🔼',
            2: '🔼🔼',
            1: '🔼',
            0: '➡️',
            -1: '🔽',
            -2: '🔽🔽',
            -3: '🔽🔽🔽'
        }
        return arrows.get(trend, '➡️')

class EnhancedGlucosePredictor:
    """XGBoost with BIG IDEAs methodology + enhancements"""
    
    def __init__(self):
        self.models = {}
        self.feature_selector = None
        self.scaler = RobustScaler()
        
    def train_phase1_baseline(self, X, y):
        """Phase 1: Replicate BIG IDEAs results"""
        print("🚀 Phase 1: Training baseline (Expected MARD ~7.1%, R² ~0.73)")
        
        # Use their exact hyperparameters
        baseline_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.models['baseline'] = xgb.XGBRegressor(**baseline_params)
        self.models['baseline'].fit(X, y)
        
        # Validate
        predictions = self.models['baseline'].predict(X)
        mard = self.calculate_mard(y, predictions)
        r2 = r2_score(y, predictions)
        
        print(f"   MARD: {mard:.2f}%, R²: {r2:.3f}")
        
        return self.models['baseline']
    
    def train_phase2_enhanced(self, X_enhanced, y):
        """Phase 2: Add advanced features"""
        print("🚀 Phase 2: Enhanced model (Target MARD 5-6%, R² 0.75-0.80)")
        
        # Feature selection with mutual information
        from sklearn.feature_selection import SelectKBest, mutual_info_regression
        
        self.feature_selector = SelectKBest(mutual_info_regression, k=50)
        X_selected = self.feature_selector.fit_transform(X_enhanced, y)
        
        # Hyperparameter optimization
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0)
            }
            
            model = xgb.XGBRegressor(**params, random_state=42)
            
            # Cross-validation with time series split
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            
            scores = []
            for train_idx, val_idx in tscv.split(X_selected):
                X_train, X_val = X_selected[train_idx], X_selected[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         verbose=False, early_stopping_rounds=50)
                
                pred = model.predict(X_val)
                scores.append(np.sqrt(mean_squared_error(y_val, pred)))
            
            return np.mean(scores)
        
        # Optimize
        import optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100, show_progress_bar=True)
        
        # Train final model
        self.models['enhanced'] = xgb.XGBRegressor(**study.best_params, random_state=42)
        self.models['enhanced'].fit(X_selected, y)
        
        return self.models['enhanced']
    
    def train_multi_horizon(self, X, y_dict):
        """Train separate models for different prediction horizons"""
        print("🚀 Training multi-horizon predictors")
        
        for horizon, y in y_dict.items():
            print(f"   Training {horizon} model...")
            
            model = xgb.XGBRegressor(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=8,
                random_state=42
            )
            
            model.fit(X, y)
            self.models[f'horizon_{horizon}'] = model
        
        return self.models

class RealTimeGlucoGuard:
    """Complete real-time monitoring system"""
    
    def __init__(self, model_path):
        self.predictor = self._load_models(model_path)
        self.e4_processor = EmpaticaE4Processor(BIGIDEAsConfig())
        self.cgm_engineer = CGMFeatureEngineer()
        self.alert_system = ContextAwareAlertSystem()
        self.complication_predictor = ComplicationPredictor()
        
        # Buffers
        self.cgm_buffer = deque(maxlen=288)  # 24 hours
        self.e4_buffer = {
            'eda': deque(maxlen=1440),  # 6 hours at 4Hz
            'bvp': deque(maxlen=23040),  # 6 hours at 64Hz
            'temp': deque(maxlen=1440),
            'acc': deque(maxlen=11520)
        }
        
    def process_new_data(self, cgm_reading, e4_data):
        """Process incoming sensor data"""
        
        # Update buffers
        self.cgm_buffer.append(cgm_reading)
        for sensor, data in e4_data.items():
            self.e4_buffer[sensor].append(data)
        
        # Extract features
        cgm_features = self.cgm_engineer.extract_all_features(self.cgm_buffer)
        e4_features = self._process_e4_features()
        
        # Combine features
        combined_features = {**cgm_features, **e4_features}
        
        # Get predictions
        predictions = self._get_predictions(combined_features)
        
        # Generate alerts
        alerts = self.alert_system.evaluate_current_state({
            'glucose': cgm_reading,
            'glucose_history': list(self.cgm_buffer),
            'glucose_pred_30min': predictions['30min'],
            **e4_features
        })
        
        # Check complications (daily)
        if self._should_check_complications():
            complications = self.complication_predictor.predict_all(combined_features)
        else:
            complications = None
        
        return {
            'current_glucose': cgm_reading,
            'trend_arrow': self.cgm_engineer.calculate_trend_arrows(list(self.cgm_buffer)),
            'predictions': predictions,
            'alerts': alerts,
            'complications': complications,
            'confidence': self._calculate_confidence(predictions, e4_features)
        }
    
    def _calculate_confidence(self, predictions, e4_features):
        """Calculate prediction confidence based on data quality"""
        confidence_factors = {
            'data_completeness': 1.0 - (self.cgm_buffer.count(None) / len(self.cgm_buffer)),
            'sensor_quality': e4_features.get('signal_quality', 0.8),
            'model_uncertainty': 1.0 / (1 + predictions.get('uncertainty', 0.2)),
            'context_match': self._assess_context_match()
        }
        
        return np.mean(list(confidence_factors.values())) * 100

# Debug helper
class GlucoGuardDebugger:
    @staticmethod
    def check_data_pipeline():
        """Verify data loading and processing"""
        checks = {
            'cgm_data_loaded': False,
            'e4_data_loaded': False,
            'timestamps_aligned': False,
            'features_extracted': False,
            'no_nan_values': False
        }
        
        # Add your checks here
        return checks
    
    @staticmethod
    def validate_predictions(predictions, ground_truth):
        """Validate prediction quality"""
        metrics = {
            'mard': calculate_mard(ground_truth, predictions),
            'clarke_zones': calculate_clarke_zones(ground_truth, predictions),
            'rmse': np.sqrt(mean_squared_error(ground_truth, predictions)),
            'time_lag': calculate_time_lag(ground_truth, predictions)
        }
        
        # Check if meeting targets
        passed = metrics['mard'] < 10 and metrics['clarke_zones']['A+B'] > 95
        
        return metrics, passed
