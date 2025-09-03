import pandas as pd
import numpy as np
from scipy import stats
from src.config import GlucoGuardConfig

class CGMFeatureEngineer:
    """Extract advanced features from CGM data"""
    
    def __init__(self):
        self.config = GlucoGuardConfig()
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive CGM features"""
        
        if 'glucose' not in df.columns:
            raise ValueError("DataFrame must contain 'glucose' column")
        
        glucose = df['glucose']
        features = pd.DataFrame(index=df.index)
        
        # Trend arrows
        features['trend_arrow'] = self.calculate_trend_arrows(glucose)
        
        # Basic statistics (multiple windows)
        for window in [6, 12, 24, 48]:  # 30min, 1hr, 2hr, 4hr
