import torch
import numpy as np
from collections import deque
from typing import Dict

from src.models.predictor import HierarchicalGlucosePredictor
from src.data_processing.cgm_dataset import CGMConfig

class AlertManager:
    def __init__(self):
        self.alert_history = deque(maxlen=10)
        self.cooldown = {}

    def check(self, prediction: Dict, buffer: deque):
        alerts = []
        current_glucose = buffer[-1]['glucose']

        if 'glucose' not in prediction or prediction['glucose'].numel() == 0:
            return alerts

        pred_30min = prediction['glucose'][0].item()

        # Level 1: Urgent hypoglycemia
        if pred_30min < 54:
            if self._can_alert('severe_hypo'):
                alerts.append({
                    'level': 'CRITICAL',
                    'type': 'severe_hypoglycemia',
                    'message': f'⚠️ URGENT: Severe low predicted ({pred_30min:.0f} mg/dL in 30 min)',
                    'action': 'Consume 30g fast-acting carbohydrates immediately',
                    'confidence': self._calculate_confidence(prediction)
                })

        # Level 2: Moderate hypoglycemia
        elif pred_30min < 70:
            if self._can_alert('hypo'):
                alerts.append({
                    'level': 'WARNING',
                    'type': 'hypoglycemia',
                    'message': f'Low glucose predicted ({pred_30min:.0f} mg/dL)',
                    'action': 'Consider eating 15g carbohydrates',
                    'confidence': self._calculate_confidence(prediction)
                })

        # Rapid change alerts
        rate = (pred_30min - current_glucose) / 30
        if abs(rate) > 3:
            direction = "dropping" if rate < 0 else "rising"
            alerts.append({
                'level': 'INFO',
                'type': 'rapid_change',
                'message': f'Glucose {direction} rapidly ({rate:.1f} mg/dL/min)'
            })

        return alerts

    def _can_alert(self, alert_type: str, cooldown_period: int = 300) -> bool:
        """Check if an alert of a certain type can be triggered, based on a cooldown."""
        now = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else torch.cpu.Event(enable_timing=True)
        now.record()

        if alert_type in self.cooldown:
            last_alert_time = self.cooldown[alert_type]
            if now.elapsed_time(last_alert_time) / 1000 < cooldown_period:
                return False

        self.cooldown[alert_type] = now
        return True

    def _calculate_confidence(self, prediction: Dict) -> float:
        """Calculate prediction confidence from uncertainty"""
        if 'uncertainty' not in prediction or prediction['uncertainty'].numel() == 0:
            return 50.0 # Default confidence

        uncertainty = prediction['uncertainty'][0].item()

        # Convert uncertainty to confidence (0-100%)
        confidence = 100 * np.exp(-uncertainty)

        return min(max(confidence, 0), 100)

class RealTimeGlucoseMonitor:
    def __init__(self, model: HierarchicalGlucosePredictor, config: CGMConfig):
        self.model = model
        self.config = config
        self.buffer = deque(maxlen=config.sequence_length)
        self.alert_manager = AlertManager()

    def update(self, new_glucose: float, timestamp, vitals: Optional[Dict] = None):
        """Process new glucose reading"""
        # Add to buffer
        self.buffer.append({
            'glucose': new_glucose,
            'timestamp': timestamp,
            'vitals': vitals
        })

        if len(self.buffer) < self.config.sequence_length:
            return None

        # Prepare input (this is a simplified version, real implementation would need feature engineering)
        # For now, we just pass the glucose values
        glucose_values = [item['glucose'] for item in self.buffer]
        # The model expects a batch, so we add a batch dimension
        # The model also expects a feature dimension, which is more complex to create in real-time
        # This part requires a more complex real-time feature engineering pipeline
        # As a placeholder, we will create a dummy tensor of the right shape
        dummy_features = torch.randn(1, self.config.sequence_length, 50) # Assuming 50 features

        # Get prediction
        with torch.no_grad():
            prediction = self.model(dummy_features)

        # Generate alerts
        alerts = self.alert_manager.check(prediction, self.buffer)

        return {
            'prediction': prediction,
            'alerts': alerts
        }
