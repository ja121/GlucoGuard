import argparse
import pandas as pd
from src.data_processing.e4_processor import EmpaticaE4Processor
from src.data_processing.cgm_features import CGMFeatureEngineer
from src.models.xgboost_predictor import EnhancedGlucosePredictor
from src.models.complication_predictor import ComplicationPredictor
from src.alerts.alert_system import ContextAwareAlertSystem
from src.utils.metrics import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='GlucoGuard Training')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='models/')
    parser.add_argument('--phase', type=int, default=2)
    args = parser.parse_args()
    
    print("🚀 GlucoGuard AI System")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    data = pd.read_csv(args.data_path, index_col='datetime', parse_dates=True)
    
    # Process E4 data
    e4_processor = EmpaticaE4Processor()
    e4_features = e4_processor.process_dataframe(data)
    
    # Process CGM features
    cgm_engineer = CGMFeatureEngineer()
    cgm_features = cgm_engineer.extract_all_features(data)
    
    # Combine features
    X = pd.concat([e4_features, cgm_features], axis=1)
    y = data['glucose_target_30min']
    
    # Train model
    predictor = EnhancedGlucosePredictor()
    
    if args.phase == 1:
        model = predictor.train_baseline(X, y)
    else:
        model = predictor.train_enhanced(X, y)
    
    # Evaluate
    results = evaluate_model(model, X, y)
    print(f"\nResults: MARD={results['mard']:.2f}%, R²={results['r2']:.3f}")
    
    # Save model
    import pickle
    with open(f"{args.output_path}/glucoguard_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to {args.output_path}")

if __name__ == "__main__":
    main()
