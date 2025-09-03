import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class EnhancedGlucosePredictor:
    def __init__(self):
        self.model = None
        
    def train_baseline(self, X, y):
        """Phase 1: Baseline model"""
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'random_state': 42
        }
        
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X, y)
        
        # Evaluate
        pred = self.model.predict(X)
        mard = np.mean(np.abs(pred - y) / (y + 1e-8)) * 100
        print(f"Baseline MARD: {mard:.2f}%")
        
        return self.model
    
    def train_enhanced(self, X, y):
        """Phase 2: Enhanced model with optimization"""
        # Use Optuna for hyperparameter tuning
        try:
            import optuna
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 4, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                
                model = xgb.XGBRegressor(**params, random_state=42)
                model.fit(X, y)
                pred = model.predict(X)
                
                return np.sqrt(mean_squared_error(y, pred))
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)
            
            # Train with best params
            self.model = xgb.XGBRegressor(**study.best_params, random_state=42)
            self.model.fit(X, y)
            
        except ImportError:
            # Fallback to baseline if Optuna not installed
            return self.train_baseline(X, y)
        
        return self.model
