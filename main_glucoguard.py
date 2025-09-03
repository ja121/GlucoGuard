# main_glucoguard.py
if __name__ == "__main__":
    print("🚀 GlucoGuard AI System Starting...")
    
    # Load BIG IDEAs dataset
    data = load_big_ideas_dataset("path/to/data")
    
    # Process Empatica E4 data
    e4_processor = EmpaticaE4Processor(BIGIDEAsConfig())
    e4_features = e4_processor.process_all(data['e4'])
    
    # Extract CGM features
    cgm_engineer = CGMFeatureEngineer()
    cgm_features = cgm_engineer.extract_all(data['cgm'])
    
    # Combine
    X = pd.concat([cgm_features, e4_features], axis=1)
    y = data['glucose_target_30min']
    
    # Train
    predictor = EnhancedGlucosePredictor()
    
    # Phase 1
    baseline_model = predictor.train_phase1_baseline(X[:30], y)
    
    # Phase 2
    enhanced_model = predictor.train_phase2_enhanced(X, y)
    
    # Deploy
    monitor = RealTimeGlucoGuard("models/")
    
    print("✅ System ready for real-time monitoring!")
