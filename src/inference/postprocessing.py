import torch
import torch.nn as nn
import numpy as np
from typing import List
from sklearn.isotonic import IsotonicRegression

class EnsemblePredictor:
    def __init__(self, models: List[nn.Module]):
        self.models = models
        self.calibrator = IsotonicRegression()

    def predict(self, x, use_uncertainty=True):
        predictions = []
        uncertainties = []

        for model in self.models:
            with torch.no_grad():
                output = model(x)
                predictions.append(output['glucose'])
                uncertainties.append(output['uncertainty'])

        # Stack predictions
        preds = torch.stack(predictions)
        uncert = torch.stack(uncertainties)

        if use_uncertainty:
            # Weighted average by inverse uncertainty
            weights = 1 / (uncert + 1e-8)
            weights = weights / weights.sum(dim=0)
            final_pred = (preds * weights).sum(dim=0)
        else:
            # Simple average
            final_pred = preds.mean(dim=0)

        # Calibrate predictions
        # Note: IsotonicRegression is not designed for multi-output regression.
        # This is a simplification. A more complex calibration would be needed for a real system.
        final_pred_calibrated = self.calibrator.fit_transform(final_pred.cpu().numpy().flatten())

        return final_pred_calibrated.reshape(final_pred.shape)

class PostProcessor:
    def __init__(self):
        self.smoothing_window = 3
        self.outlier_threshold = 3.0

    def process(self, predictions, historical_glucose):
        # Remove outliers
        predictions = self._remove_outliers(predictions, historical_glucose)

        # Apply smoothing
        predictions = self._smooth_predictions(predictions)

        # Enforce physiological constraints
        predictions = self._apply_constraints(predictions)

        return predictions

    def _remove_outliers(self, preds, history):
        """Remove physiologically impossible predictions"""
        # Max rate of change is ~4 mg/dL/min
        max_change = 4 * 30  # 30 minutes

        last_glucose = history[-1]
        mask = np.abs(preds - last_glucose) <= max_change

        # Use np.clip to enforce the max change
        preds = np.clip(preds, last_glucose - max_change, last_glucose + max_change)

        return preds

    def _smooth_predictions(self, preds):
        """Apply a simple moving average smoothing"""
        if len(preds) < self.smoothing_window:
            return preds
        return np.convolve(preds, np.ones(self.smoothing_window)/self.smoothing_window, mode='valid')

    def _apply_constraints(self, preds):
        """Apply physiological constraints"""
        # Glucose rarely goes below 40 or above 400
        preds = np.clip(preds, 40, 400)

        return preds
