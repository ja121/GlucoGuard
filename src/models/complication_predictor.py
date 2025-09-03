class ComplicationPredictor:
    """Predict long-term diabetes complications"""
    
    def __init__(self):
        self.risk_models = {}
        
    def calculate_retinopathy_risk(self, patient_data):
        """Diabetic retinopathy risk score"""
        # Based on UKPDS risk engine
        hba1c = self._estimate_hba1c(patient_data['glucose'])
        duration = patient_data.get('diabetes_duration_years', 5)
        
        risk_score = (
            0.9 * hba1c +
            0.3 * duration +
            0.5 * patient_data['time_in_hyper'] +
            0.4 * patient_data['glucose_variability']
        )
        
        # Convert to probability
        risk_prob = 1 / (1 + np.exp(-0.5 * (risk_score - 10)))
        
        return {
            'risk_score': risk_score,
            'probability_1_year': risk_prob,
            'risk_level': self._categorize_risk(risk_prob)
        }
    
    def calculate_nephropathy_risk(self, patient_data):
        """Diabetic kidney disease risk"""
        # Based on clinical studies
        risk_factors = {
            'poor_control': patient_data['time_in_hyper'] > 30,
            'high_variability': patient_data['mage'] > 60,
            'stress': patient_data['avg_stress_level'] > 0.7,
            'hypertension': patient_data.get('bp_systolic', 120) > 140
        }
        
        risk_score = sum(risk_factors.values()) * 25
        
        return {
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': self._get_nephropathy_recommendations(risk_factors)
        }
    
    def calculate_neuropathy_risk(self, patient_data):
        """Peripheral neuropathy risk"""
        # Temperature variability as early indicator
        temp_variability = patient_data['temp_std']
        glucose_exposure = patient_data['cumulative_hyperglycemia']
        
        risk_score = (
            0.6 * glucose_exposure +
            0.3 * temp_variability +
            0.1 * patient_data['diabetes_duration_years']
        )
        
        return {
            'risk_score': risk_score,
            'early_signs': temp_variability > 2.0,
            'monitoring_frequency': 'monthly' if risk_score > 50 else 'quarterly'
        }
    
    def calculate_cardiovascular_risk(self, patient_data):
        """CVD risk with HRV integration"""
        # Use HRV as autonomic neuropathy marker
        hrv_risk = 1 - (patient_data['hrv_rmssd'] / 50)  # Normalized
        glucose_risk = patient_data['adrr'] / 40  # Normalized
        
        composite_risk = (
            0.4 * hrv_risk +
            0.3 * glucose_risk +
            0.2 * patient_data['time_in_hyper'] / 100 +
            0.1 * patient_data.get('age', 40) / 100
        )
        
        return {
            'risk_score': composite_risk * 100,
            'autonomic_dysfunction': hrv_risk > 0.7,
            'intervention_needed': composite_risk > 0.6
        }
    
    def _estimate_hba1c(self, glucose_series):
        """Estimate HbA1c from CGM data"""
        mean_glucose = np.mean(glucose_series)
        # Nathan formula
        hba1c = (mean_glucose + 46.7) / 28.7
        return hba1c
