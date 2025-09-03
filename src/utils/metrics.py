import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple

def evaluate_model(model, X, y) -> Dict:
    """Comprehensive model evaluation"""
    
    predictions = model.predict(X)
    
    metrics = {
        'mard': calculate_mard(y, predictions),
        'rmse': np.sqrt(mean_squared_error(y, predictions)),
        'mae': mean_absolute_error(y, predictions),
        'r2': r2_score(y, predictions),
        'clarke_ab': calculate_clarke_zones(y, predictions)['A+B']
    }
    
    return metrics

def calculate_mard(y_true, y_pred) -> float:
    """Calculate Mean Absolute Relative Difference"""
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true > 0
    
    if not any(mask):
        return 0.0
    
    mard = np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]) * 100
    
    return mard

def calculate_clarke_zones(y_true, y_pred) -> Dict:
    """Calculate Clarke Error Grid zones"""
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Initialize zone counts
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    total = len(y_true)
    
    for actual, predicted in zip(y_true, y_pred):
        zone = _get_clarke_zone(actual, predicted)
        zones[zone] += 1
    
    # Convert to percentages
    for zone in zones:
        zones[zone] = (zones[zone] / total) * 100
    
    zones['A+B'] = zones['A'] + zones['B']
    
    return zones

def _get_clarke_zone(actual, predicted) -> str:
    """Determine Clarke Error Grid zone for a single point"""
    
    # Zone A: Clinically accurate
    if abs(actual - predicted) <= 20:
        return 'A'
    
    if actual >= 70 and predicted >= 70:
        if abs((predicted - actual) / actual) <= 0.2:
            return 'A'
    
    # Zone B: Benign errors
    if actual > 180 and predicted > 180:
        return 'B'
    
    if actual < 70 and predicted < 70:
        return 'B'
    
    if 70 <= actual <= 180 and abs(predicted - actual) <= 40:
        return 'B'
    
    # Zone C: Overcorrection
    if actual >= 70 and predicted < 70 and actual <= 180:
        return 'C'
    
    if actual < 70 and predicted >= 180:
        return 'C'
    
    # Zone D: Dangerous failure to detect
    if actual < 70 and 70 <= predicted <= 180:
        return 'D'
    
    if actual > 240 and 70 <= predicted <= 180:
        return 'D'
    
    # Zone E: Dangerous errors
    return 'E'

def calculate_surveillance_error_grid(y_true, y_pred) -> Dict:
    """Calculate Surveillance Error Grid (more recent than Clarke)"""
    
    # Implementation similar to Clarke but with updated zones
    # This is a simplified version
    
    zones = {'No_Risk': 0, 'Slight_Risk': 0, 'Moderate_Risk': 0, 'Great_Risk': 0}
    
    for actual, predicted in zip(y_true, y_pred):
        risk = _get_surveillance_risk(actual, predicted)
        zones[risk] += 1
    
    total = len(y_true)
    for zone in zones:
        zones[zone] = (zones[zone] / total) * 100
    
    return zones

def _get_surveillance_risk(actual, predicted) -> str:
    """Determine Surveillance Error Grid risk level"""
    
    error = abs(actual - predicted)
    relative_error = error / (actual + 1e-8)
    
    if error < 15 or relative_error < 0.15:
        return 'No_Risk'
    elif error < 30 or relative_error < 0.3:
        return 'Slight_Risk'
    elif error < 45 or relative_error < 0.4:
        return 'Moderate_Risk'
    else:
        return 'Great_Risk'

def calculate_glycemic_risk_metrics(glucose_series: pd.Series) -> Dict:
    """Calculate various glucose risk metrics"""
    
    metrics = {}
    
    # Low Blood Glucose Index (LBGI)
    f_glucose = 1.509 * (np.log(glucose_series + 1e-8)**1.084 - 5.381)
    rl = np.where(f_glucose < 0, 10 * f_glucose**2, 0)
    metrics['lbgi'] = np.mean(rl)
    
    # High Blood Glucose Index (HBGI)
    rh = np.where(f_glucose > 0, 10 * f_glucose**2, 0)
    metrics['hbgi'] = np.mean(rh)
    
    # Average Daily Risk Range (ADRR)
    metrics['adrr'] = metrics['lbgi'] + metrics['hbgi']
    
    # Glycemic Risk Assessment Diabetes Equation (GRADE)
    grade_values = []
    for g in glucose_series:
        if g < 70.2:
            grade = 10 * (np.log10(70.2) - np.log10(g)) ** 2
        elif g > 140.4:
            grade = 10 * (np.log10(g) - np.log10(140.4)) ** 2
        else:
            grade = 0
        grade_values.append(grade)
    
    metrics['grade'] = np.mean(grade_values)
    
    return metrics
