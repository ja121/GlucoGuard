from dataclasses import dataclass
from typing import List, Dict

@dataclass
class GlucoGuardConfig:
    """Central configuration for GlucoGuard system"""
    
    # Empatica E4 Wearable Settings
    e4_eda_hz: int = 4  # Electrodermal activity
    e4_bvp_hz: int = 64  # Blood volume pulse
    e4_temp_hz: int = 4  # Skin temperature
    e4_acc_hz: int = 32  # Accelerometer
    
    # Dexcom G6 CGM Settings
    cgm_interval_minutes: int = 5
    cgm_device: str = "Dexcom G6"
    
    # Clinical Thresholds
    hypo_threshold: float = 70.0
    severe_hypo: float = 54.0
    hyper_threshold: float = 180.0
    severe_hyper: float = 250.0
    target_range: tuple = (70, 180)
    
    # Prediction Settings
    prediction_horizons: List[int] = None  # in 5-min intervals
    
    # Model Settings
    model_type: str = "xgboost"
    use_optuna: bool = True
    optuna_trials: int = 50
    
    # Alert Settings
    alert_cooldown_minutes: int = 15
    max_alerts_per_hour: int = 4
    
    # Complication Risk Settings
    complication_horizon_days: int = 365
    risk_levels: Dict[str, float] = None
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [6, 12, 18]  # 30min, 1hr, 1.5hr
        
        if self.risk_levels is None:
            self.risk_levels = {
                'low': 0.2,
                'moderate': 0.5,
                'high': 0.7,
                'critical': 0.9
            }
    
    @property
    def prediction_minutes(self):
        """Convert horizon intervals to minutes"""
        return [h * self.cgm_interval_minutes for h in self.prediction_horizons]
