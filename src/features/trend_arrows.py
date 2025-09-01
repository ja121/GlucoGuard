import pandas as pd
import numpy as np

def calculate_trend_arrow(rate_of_change_per_min):
    """
    Calculate CGM trend arrow based on Dexcom G6 specifications

    Parameters:
    rate_of_change_per_min (float): Rate of glucose change in mg/dL per minute

    Returns:
    dict: Contains arrow symbol, description, and predicted 30-min change
    """

    # Rising trends
    if rate_of_change_per_min > 3:
        return {
            'arrow': '🔼🔼🔼',
            'symbol': '↑↑↑',
            'direction': 'RISING',
            'speed': 'FAST',
            'rate_per_min': rate_of_change_per_min,
            'predicted_30min_change': '>90',
            'description': 'Rising very fast - glucose increasing rapidly'
        }

    elif rate_of_change_per_min > 2:
        return {
            'arrow': '🔼🔼',
            'symbol': '↑↑',
            'direction': 'RISING',
            'speed': 'MODERATE',
            'rate_per_min': rate_of_change_per_min,
            'predicted_30min_change': '60-90',
            'description': 'Rising fast - significant glucose increase'
        }

    elif rate_of_change_per_min > 1:
        return {
            'arrow': '🔼',
            'symbol': '↗',
            'direction': 'RISING',
            'speed': 'SLOW',
            'rate_per_min': rate_of_change_per_min,
            'predicted_30min_change': '30-60',
            'description': 'Rising slowly - gradual glucose increase'
        }

    # Stable trend
    elif rate_of_change_per_min >= -1 and rate_of_change_per_min <= 1:
        return {
            'arrow': '➡️',
            'symbol': '→',
            'direction': 'STABLE',
            'speed': 'NONE',
            'rate_per_min': rate_of_change_per_min,
            'predicted_30min_change': '<30',
            'description': 'Stable - minimal glucose change expected'
        }

    # Falling trends
    elif rate_of_change_per_min > -2:  # Between -2 and -1
        return {
            'arrow': '🔽',
            'symbol': '↘',
            'direction': 'FALLING',
            'speed': 'SLOW',
            'rate_per_min': rate_of_change_per_min,
            'predicted_30min_change': '30-60',
            'description': 'Falling slowly - gradual glucose decrease'
        }

    elif rate_of_change_per_min > -3:  # Between -3 and -2
        return {
            'arrow': '🔽🔽',
            'symbol': '↓↓',
            'direction': 'FALLING',
            'speed': 'MODERATE',
            'rate_per_min': rate_of_change_per_min,
            'predicted_30min_change': '60-90',
            'description': 'Falling fast - significant glucose decrease'
        }

    else:  # <= -3
        return {
            'arrow': '🔽🔽🔽',
            'symbol': '↓↓↓',
            'direction': 'FALLING',
            'speed': 'FAST',
            'rate_per_min': rate_of_change_per_min,
            'predicted_30min_change': '>90',
            'description': 'Falling very fast - glucose decreasing rapidly'
        }

def calculate_rate_of_change(glucose_values, timestamps=None, method='simple'):
    """
    Calculate rate of change for glucose values

    Parameters:
    glucose_values (list/array): List of glucose readings
    timestamps (list/array): Timestamps (optional, assumes 5-min intervals if None)
    method (str): 'simple' for last two points, 'average' for trend over multiple points

    Returns:
    float: Rate of change in mg/dL per minute
    """

    if len(glucose_values) < 2:
        return 0

    if method == 'simple':
        # Simple difference between last two readings
        glucose_diff = glucose_values[-1] - glucose_values[-2]
        time_diff = 5  # Assume 5-minute intervals for CGM
        return glucose_diff / time_diff

    elif method == 'average' and len(glucose_values) >= 3:
        # Average rate over last 3 readings (15 minutes)
        glucose_diff = glucose_values[-1] - glucose_values[-3]
        time_diff = 10  # 2 intervals of 5 minutes each
        return glucose_diff / time_diff

    else:
        return calculate_rate_of_change(glucose_values, timestamps, 'simple')

def process_cgm_data_with_trends(df, glucose_col='glucose', timestamp_col='timestamp', subject_col='subject_id'):
    """
    Process CGM dataframe and add trend arrows for each subject.

    Parameters:
    df (pandas.DataFrame): CGM data
    glucose_col (str): Name of glucose column
    timestamp_col (str): Name of timestamp column
    subject_col (str): Name of subject ID column

    Returns:
    pandas.DataFrame: Original data with trend information added
    """

    # Define a function to apply to each subject group
    def apply_trends_to_group(group):
        group = group.copy()
        group['rate_of_change'] = 0.0
        group['trend_arrow'] = '➡️'
        group['trend_description'] = 'Stable'
        group['predicted_30min_change'] = '<30'

        # Ensure the group is sorted by timestamp
        group = group.sort_values(by=timestamp_col)

        # Use .iloc for setting values to avoid SettingWithCopyWarning
        for i in range(1, len(group)):
            # Get last few glucose values
            # Use .iloc on the group's values to get correct indices
            glucose_values = group[glucose_col].iloc[max(0, i-2):i+1].tolist()

            # Calculate rate of change
            rate = calculate_rate_of_change(glucose_values)
            group.iloc[i, group.columns.get_loc('rate_of_change')] = rate

            # Get trend arrow info
            trend_info = calculate_trend_arrow(rate)
            group.iloc[i, group.columns.get_loc('trend_arrow')] = trend_info['arrow']
            group.iloc[i, group.columns.get_loc('trend_description')] = trend_info['description']
            group.iloc[i, group.columns.get_loc('predicted_30min_change')] = trend_info['predicted_30min_change']

        return group

    # Check if subject_col exists
    if subject_col not in df.columns:
        # If no subject ID, treat the whole dataframe as a single subject
        return apply_trends_to_group(df)

    # Group by subject and apply the trend calculation
    # Using a lambda with .copy() to be extra safe about modifications
    processed_df = df.groupby(subject_col, group_keys=False).apply(lambda x: apply_trends_to_group(x.copy()))

    # Merge the processed data back into the original dataframe to preserve all columns
    # and the original index, which might be important for subsequent steps.
    # The columns added are 'rate_of_change', 'trend_arrow', etc.
    output_cols = ['rate_of_change', 'trend_arrow', 'trend_description', 'predicted_30min_change']

    # Ensure the original index is preserved in the processed data before merging
    df_with_trends = df.merge(processed_df[output_cols], left_index=True, right_index=True, how='left')

    return df_with_trends
