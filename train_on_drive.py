import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# For Google Colab
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    sys.path.append('/content/glucoguard')

from src.data_processing.e4_processor import EmpaticaE4Processor
from src.data_processing.cgm_features import CGMFeatureEngineer
from src.models.xgboost_predictor import EnhancedGlucosePredictor
from src.models.complication_predictor import ComplicationPredictor
from src.alerts.alert_system import ContextAwareAlertSystem
from src.utils.metrics import evaluate_model, calculate_clarke_zones

def train_on_drive_data():
    print("="*60)
    print("🚀 GlucoGuard Training on Google Drive Dataset")
    print("="*60)
    
    # Path to your data
    data_path = "/content/drive/MyDrive/MyDataset/processed/ALL_PARTICIPANTS_processed.csv"
    
    # Load data
    print("\n📂 Loading data from Google Drive...")
    try:
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        print(f"✅ Loaded {len(df)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Check for E4 columns
    e4_columns = ['eda', 'bvp', 'temp', 'acc_x', 'acc_y', 'acc_z']
    has_e4 = all(col in df.columns for col in e4_columns if col != 'bvp')  # BVP might be processed to HR
    
    print(f"\n🔍 E4 Wearable Data: {'✅ Found' if has_e4 else '⚠️ Not found - creating synthetic'}")
    
    # Process E4 features
    e4_processor = EmpaticaE4Processor()
    if has_e4:
        e4_features = e4_processor.process_dataframe(df)
    else:
        # Create synthetic E4 features for testing
        e4_features = pd.DataFrame(index=df.index)
        e4_features['stress_level'] = 5 + np.random.randn(len(df)) * 2
        e4_features['heart_rate'] = 70 + np.random.randn(len(df)) * 10
        e4_features['hrv_rmssd'] = 40 + np.random.randn(len(df)) * 10
        e4_features['activity_level'] = np.random.uniform(0, 5, len(df))
        e4_features['skin_temp'] = 32 + np.random.randn(len(df)) * 2
    
    # Process CGM features
    print("\n📊 Extracting CGM features...")
    cgm_engineer = CGMFeatureEngineer()
    
    if 'glucose' in df.columns:
        cgm_features = cgm_engineer.extract_all_features(df)
    else:
        print("❌ No glucose column found!")
        return None
    
    # Combine all features
    X = pd.concat([df.select_dtypes(include=[np.number]), e4_features, cgm_features], axis=1)
    
    # Remove target and ID columns from features
    feature_cols = [col for col in X.columns if col not in 
                   ['glucose_target_30min', 'participant_id', 'glucose_target_15min', 'glucose_target_60min']]
    X = X[feature_cols]
    
    # Prepare target
    if 'glucose_target_30min' not in df.columns:
        df['glucose_target_30min'] = df['glucose'].shift(-6)
    
    y = df['glucose_target_30min'].dropna()
    X = X.loc[y.index]
    
    print(f"\n📐 Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train models
    predictor = EnhancedGlucosePredictor()
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n🧪 Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Phase 1: Baseline
    print("\n" + "="*40)
    print("PHASE 1: Baseline Model (BIG IDEAs)")
    print("="*40)
    baseline_model = predictor.train_baseline(X_train, y_train)
    
    # Phase 2: Enhanced
    print("\n" + "="*40)
    print("PHASE 2: Enhanced Model")
    print("="*40)
    enhanced_model = predictor.train_enhanced(X_train, y_train)
    
    # Evaluate on test set
    print("\n" + "="*40)
    print("📈 FINAL EVALUATION ON TEST SET")
    print("="*40)
    
    results = evaluate_model(enhanced_model, X_test, y_test)
    
    print(f"""
    Clinical Metrics:
    ├── MARD: {results['mard']:.2f}% (Target: <10%)
    ├── RMSE: {results['rmse']:.2f} mg/dL
    ├── R²: {results['r2']:.3f} (Target: >0.70)
    └── Clarke A+B: {results['clarke_ab']:.1f}% (Target: >95%)
    """)
    
    # Initialize other systems
    print("\n🚨 Initializing Alert System...")
    alert_system = ContextAwareAlertSystem()
    
    print("🔮 Initializing Complication Predictor...")
    complication_predictor = ComplicationPredictor()
    
    # Save everything
    save_path = "/content/drive/MyDrive/MyDataset/models/"
    os.makedirs(save_path, exist_ok=True)
    
    # Save models
    with open(f"{save_path}/glucoguard_enhanced.pkl", 'wb') as f:
        pickle.dump(enhanced_model, f)
    
    with open(f"{save_path}/glucoguard_alerts.pkl", 'wb') as f:
        pickle.dump(alert_system, f)
    
    print(f"\n💾 Models saved to: {save_path}")
    print("\n✅ Training complete! Ready for deployment.")
    
    return enhanced_model, alert_system, results

if __name__ == "__main__":
    model, alerts, results = train_on_drive_data()
