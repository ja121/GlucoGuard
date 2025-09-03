import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

class EnhancedGlucosePredictor:
    """XGBoost predictor with BIG IDEAs methodology"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def train_baseline(self, X, y):
        """Phase 1: Baseline model matching BIG IDEAs results"""
        
        print("\n🔬 Training Baseline Model (Target: MARD ~7.1%, R² ~0.73)")
        
        # Scale features
        self.scalers['baseline'] = RobustScaler()
        X_scaled = self.scalers['baseline'].fit_transform(X)
        
        # Baseline hyperparameters from BIG IDEAs study
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train
        self.models['baseline'] = xgb.XGBRegressor(**params)
        self.models['baseline'].fit(X_scaled, y)
        
        # Evaluate
        predictions = self.models['baseline'].predict(X_scaled)
        mard = self._calculate_mard(y, predictions)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        print(f"   ✅ Baseline Results: MARD={mard:.2f}%, R²={r2:.3f}, RMSE={rmse:.2f}")
        
        # Store feature importance
        if hasattr(X, 'columns'):
            self.feature_importance['baseline'] = pd.DataFrame({
                'feature': X.columns,
                'importance': self.models['baseline'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self.models['baseline']
    
    def train_enhanced(self, X, y):
        """Phase 2: Enhanced model with optimization"""
        
        print("\n🚀 Training Enhanced Model (Target: MARD 5-6%, R² 0.75-0.80)")
        
        # Scale features
        self.scalers['enhanced'] = RobustScaler()
        X_scaled = self.scalers['enhanced'].fit_transform(X)
        
        # Try Optuna optimization
        best_params = self._optimize_hyperparameters(X_scaled, y)
        
        # Train final model
        self.models['enhanced'] = xgb.XGBRegressor(**best_params)
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.models['enhanced'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            pred = self.models['enhanced'].predict(X_val)
            cv_scores.append(self._calculate_mard(y_val, pred))
        
        # Final training on full data
        self.models['enhanced'].fit(X_scaled, y)
        
        # Evaluate
        predictions = self.models['enhanced'].predict(X_scaled)
        mard = self._calculate_mard(y, predictions)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        print(f"   ✅ Enhanced Results: MARD={mard:.2f}%, R²={r2:.3f}, RMSE={rmse:.2f}")
        print(f"   📊 Cross-validation MARD: {np.mean(cv_scores):.2f}% (±{np.std(cv_scores):.2f}%)")
        
        # Feature importance
        if hasattr(X, 'columns'):
            self.feature_importance['enhanced'] = pd.DataFrame({
                'feature': X.columns,
                'importance': self.models['enhanced'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n   🔝 Top 10 Important Features:")
            for i, row in self.feature_importance['enhanced'].head(10).iterrows():
                print(f"      {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        return self.models['enhanced']
    
    def _optimize_hyperparameters(self, X, y):
        """Optimize using Optuna or fallback to default"""
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                params = {
                    'objective': 'reg:squarederror',
                    'tree_method': 'hist',
                    'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                # Use subset for faster optimization
                subset_size = min(5000, len(X))
                X_subset = X[:subset_size]
                y_subset = y[:subset_size]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_subset, y_subset, verbose=False)
                
                pred = model.predict(X_subset)
                return np.sqrt(mean_squared_error(y_subset, pred))
            
            print("   🔍 Running hyperparameter optimization (50 trials)...")
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50, show_progress_bar=False)
            
            best_params = study.best_params
            best_params.update({
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'random_state': 42,
                'n_jobs': -1
            })
            
            print(f"   ✅ Best parameters found (RMSE: {study.best_value:.2f})")
            
            return best_params
            
        except ImportError:
            print("   ⚠️ Optuna not available, using optimized defaults")
            
            return {
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'n_estimators': 800,
                'learning_rate': 0.03,
                'max_depth': 8,
                'min_child_weight': 3,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'gamma': 0.1,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1
            }
    
    def predict(self, X, model_type='enhanced'):
        """Make predictions with specified model"""
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        X_scaled = self.scalers[model_type].transform(X)
        predictions = self.models[model_type].predict(X_scaled)
        
        return predictions
    
    def _calculate_mard(self, y_true, y_pred):
        """Calculate Mean Absolute Relative Difference"""
        
        # Avoid division by zero
        mask = y_true > 0
        mard = np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]) * 100
        
        return mard
