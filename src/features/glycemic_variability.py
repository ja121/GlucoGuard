import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Optional, Dict, Any
import warnings

def calculate_mage(glucose_values: List[float], threshold_sd: float = 1.0) -> float:
    """
    Calculate Mean Amplitude of Glycemic Excursions (MAGE)

    MAGE measures glycemic variability by calculating the arithmetic mean of
    glucose excursions that exceed one standard deviation of the mean glucose.

    Args:
        glucose_values: List of glucose measurements (mg/dL)
        threshold_sd: Threshold in standard deviations (default: 1.0)

    Returns:
        MAGE value in mg/dL
    """
    if len(glucose_values) < 3:
        return np.nan

    values = np.array(glucose_values)
    mean_glucose = np.mean(values)
    sd_glucose = np.std(values, ddof=1)
    threshold = threshold_sd * sd_glucose

    # Find turning points (local minima and maxima)
    turning_points = []
    for i in range(1, len(values) - 1):
        if ((values[i] > values[i-1] and values[i] > values[i+1]) or
            (values[i] < values[i-1] and values[i] < values[i+1])):
            turning_points.append((i, values[i]))

    if len(turning_points) < 2:
        return np.nan

    # Calculate excursions between consecutive turning points
    excursions = []
    for i in range(len(turning_points) - 1):
        excursion = abs(turning_points[i+1][1] - turning_points[i][1])
        if excursion > threshold:
            excursions.append(excursion)

    return np.mean(excursions) if excursions else 0.0

def calculate_modd(glucose_values: List[float], timestamps: List[Any] = None) -> float:
    """
    Calculate Mean of Daily Differences (MODD)

    MODD measures day-to-day variability by calculating the mean absolute
    difference between glucose values at the same time on consecutive days.

    Args:
        glucose_values: List of glucose measurements
        timestamps: List of timestamps (if None, assumes regular intervals)

    Returns:
        MODD value in mg/dL
    """
    if len(glucose_values) < 2:
        return np.nan

    values = np.array(glucose_values)

    # If timestamps provided, use them to match same times across days
    if timestamps is not None and len(timestamps) == len(values):
        # Convert to pandas for easier date handling
        df = pd.DataFrame({'glucose': values, 'timestamp': pd.to_datetime(timestamps)})
        df['time'] = df['timestamp'].dt.time
        df['date'] = df['timestamp'].dt.date

        # Group by time and calculate differences between consecutive days
        daily_diffs = []
        for time_group in df.groupby('time'):
            time_data = time_group[1].sort_values('date')
            if len(time_data) > 1:
                glucose_series = time_data['glucose'].values
                daily_diffs.extend(np.abs(np.diff(glucose_series)))

        return np.mean(daily_diffs) if daily_diffs else np.nan
    else:
        # Simple case: assume measurements are 24 hours apart
        daily_diffs = np.abs(np.diff(values))
        return np.mean(daily_diffs)

def calculate_conga(glucose_values: List[float], n_hours: int = 1, interval_minutes: int = 15) -> float:
    """
    Calculate Continuous Overall Net Glycemic Action (CONGA)

    CONGA measures glycemic variability over n-hour periods.

    Args:
        glucose_values: List of glucose measurements
        n_hours: Number of hours for calculation (default: 1)
        interval_minutes: Measurement interval in minutes (default: 15)

    Returns:
        CONGA value in mg/dL
    """
    if len(glucose_values) < 2:
        return np.nan

    values = np.array(glucose_values)
    points_per_hour = 60 // interval_minutes
    n_points = n_hours * points_per_hour

    if len(values) <= n_points:
        return np.nan

    # Calculate differences between values n_points apart
    differences = []
    for i in range(len(values) - n_points):
        diff = values[i + n_points] - values[i]
        differences.append(diff ** 2)

    return np.sqrt(np.mean(differences))

def calculate_lbgi_hbgi(glucose_values: List[float]) -> Tuple[float, float]:
    """
    Calculate Low Blood Glucose Index (LBGI) and High Blood Glucose Index (HBGI)

    These indices measure the risk of hypoglycemia and hyperglycemia.

    Args:
        glucose_values: List of glucose measurements in mg/dL

    Returns:
        Tuple of (LBGI, HBGI) values
    """
    if len(glucose_values) == 0:
        return np.nan, np.nan

    values = np.array(glucose_values)

    # Transform glucose values to risk space
    # f(BG) = 1.509 * (ln(BG)^1.084 - 5.381) for BG in mg/dL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_bg = 1.509 * (np.power(np.log(values), 1.084) - 5.381)

    # Calculate risk values
    # rl(BG) = 22.77 * f(BG)^2 if f(BG) < 0, else 0 (hypoglycemia risk)
    # rh(BG) = 22.77 * f(BG)^2 if f(BG) > 0, else 0 (hyperglycemia risk)

    rl = np.where(f_bg < 0, 22.77 * f_bg**2, 0)
    rh = np.where(f_bg > 0, 22.77 * f_bg**2, 0)

    lbgi = np.mean(rl)
    hbgi = np.mean(rh)

    return lbgi, hbgi

def calculate_adrr(glucose_values: List[float]) -> float:
    """
    Calculate Average Daily Risk Range (ADRR)

    ADRR combines LBGI and HBGI to provide an overall risk assessment.

    Args:
        glucose_values: List of glucose measurements

    Returns:
        ADRR value
    """
    lbgi, hbgi = calculate_lbgi_hbgi(glucose_values)

    if np.isnan(lbgi) or np.isnan(hbgi):
        return np.nan

    # ADRR is typically calculated as the sum of LBGI and HBGI
    return lbgi + hbgi

def calculate_j_index(glucose_values: List[float]) -> float:
    """
    Calculate J-Index

    J-Index measures glycemic variability considering both mean glucose and variability.

    Args:
        glucose_values: List of glucose measurements in mg/dL

    Returns:
        J-Index value
    """
    if len(glucose_values) < 2:
        return np.nan

    values = np.array(glucose_values)
    mean_glucose = np.mean(values)
    sd_glucose = np.std(values, ddof=1)

    # J-Index = 0.001 * (mean + SD)^2
    j_index = 0.001 * (mean_glucose + sd_glucose) ** 2

    return j_index

def calculate_grade(glucose_values: List[float]) -> Dict[str, float]:
    """
    Calculate Glycemic Risk Assessment Diabetes Equation (GRADE)

    GRADE provides a comprehensive assessment of glycemic control.

    Args:
        glucose_values: List of glucose measurements in mg/dL

    Returns:
        Dictionary with GRADE components
    """
    if len(glucose_values) == 0:
        return {'total_grade': np.nan, 'hypoglycemia': np.nan, 'euglycemia': np.nan, 'hyperglycemia': np.nan}

    values = np.array(glucose_values)

    # Transform glucose values: 425 * (log10(log10(BG/18)) + 0.16)^2
    # Note: BG/18 converts mg/dL to mmol/L
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bg_mmol = values / 18.0
        # Avoid log of negative numbers
        bg_mmol = np.maximum(bg_mmol, 0.1)
        log_bg = np.log10(bg_mmol)
        log_bg = np.maximum(log_bg, 0.01)
        grade_values = 425 * (np.log10(log_bg) + 0.16) ** 2

    # Categorize by glucose ranges
    hypoglycemia_mask = values < 70  # <70 mg/dL
    euglycemia_mask = (values >= 70) & (values <= 180)  # 70-180 mg/dL
    hyperglycemia_mask = values > 180  # >180 mg/dL

    result = {
        'total_grade': np.mean(grade_values),
        'hypoglycemia': np.mean(grade_values[hypoglycemia_mask]) if np.any(hypoglycemia_mask) else 0,
        'euglycemia': np.mean(grade_values[euglycemia_mask]) if np.any(euglycemia_mask) else 0,
        'hyperglycemia': np.mean(grade_values[hyperglycemia_mask]) if np.any(hyperglycemia_mask) else 0
    }

    return result

def calculate_mad(glucose_values: List[float]) -> float:
    """
    Calculate Mean Absolute Deviation (MAD)

    Args:
        glucose_values: List of glucose measurements

    Returns:
        MAD value in mg/dL
    """
    if len(glucose_values) < 2:
        return np.nan

    values = np.array(glucose_values)
    median_glucose = np.median(values)

    mad = np.mean(np.abs(values - median_glucose))
    return mad

def calculate_iqr(glucose_values: List[float]) -> float:
    """
    Calculate Interquartile Range (IQR)

    Args:
        glucose_values: List of glucose measurements

    Returns:
        IQR value in mg/dL
    """
    if len(glucose_values) < 4:
        return np.nan

    values = np.array(glucose_values)
    q75, q25 = np.percentile(values, [75, 25])

    return q75 - q25
