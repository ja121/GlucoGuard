import numpy as np
import pandas as pd
from typing import Dict

class ComplicationPredictor:
    """Predict long-term diabetic complications"""
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.2,
            'moderate': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
    
    def predict_all_complications(self, patient_data: Dict) -> Dict:
        """Predict all major diabetic complications"""
        
        complications = {
            'retinopathy': self.calculate_retinopathy_risk(patient_data),
            'nephropathy': self.calculate_nephropathy_risk(patient_data),
            'neuropathy': self.calculate_neuropathy_risk(patient_data),
            'cardiovascular': self.calculate_cardiovascular_risk(patient_data)
        }
        
        # Overall risk
        complications['overall_risk'] = self._calculate_overall_risk(complications)
        
        return complications
    
    def calculate_retinopathy_risk(self, data: Dict) -> Dict:
        """Diabetic retinopathy risk assessment"""
        
        # Estimate HbA1c from mean glucose
        mean_glucose = np.mean(data.get('glucose', 120))
        hba1c = (mean_glucose + 46.7) / 28.7
        
        # Risk factors
        duration = data.get('diabetes_duration_years', 5)
        time_hyper = data.get('tir_hyper', 20)
        variability = data.get('mage', 40)
        
        # Calculate risk score (based on UKPDS and recent studies)
        risk_score = (
            0.3 * max(0, hba1c - 7) +  # HbA1c above target
            0.2 * duration / 20 +  # Years with diabetes
            0.25 * time_hyper / 100 +  # Time in hyperglycemia
            0.25 * variability / 100  # Glucose variability
        )
        
        # Convert to probability
        risk_prob = 1 / (1 + np.exp(-5 * (risk_score - 0.5)))
        
        # Risk level
        risk_level = self._get_risk_level(risk_prob)
        
        # Recommendations
        recommendations = self._get_retinopathy_recommendations(risk_level, hba1c)
        
        return {
            'risk_score': risk_score,
            'probability_1_year': risk_prob,
            'probability_5_year': min(1.0, risk_prob * 3),
            'risk_level': risk_level,
            'hba1c_estimated': round(hba1c, 1),
            'recommendations': recommendations,
            'screening_frequency': 'Every 6 months' if risk_level in ['high', 'critical'] else 'Annual'
        }
    
    def calculate_nephropathy_risk(self, data: Dict) -> Dict:
        """Diabetic kidney disease risk"""
        
        # Risk factors
        risk_factors = {
            'poor_glycemic_control': data.get('tir_target', 70) < 70,
            'high_variability': data.get('mage', 40) > 60,
            'frequent_hypoglycemia': data.get('tir_hypo', 5) > 4,
            'hypertension': data.get('bp_systolic', 120) > 140,
            'high_stress': data.get('stress_level', 5) > 7
        }
        
        # Calculate risk score
        risk_score = sum(risk_factors.values()) / len(risk_factors)
        
        # Additional weight for critical factors
        if risk_factors['poor_glycemic_control']:
            risk_score *= 1.3
        
        risk_level = self._get_risk_level(risk_score)
        
        # Early warning signs
        early_signs = {
            'microalbuminuria_risk': risk_score > 0.3,
            'egfr_decline_risk': risk_score > 0.5,
            'progression_risk': risk_score > 0.7
        }
        
        recommendations = self._get_nephropathy_recommendations(risk_factors)
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'early_signs': early_signs,
            'recommendations': recommendations,
            'monitoring': 'Quarterly urine albumin' if risk_level in ['high', 'critical'] else 'Annual'
        }
    
    def calculate_neuropathy_risk(self, data: Dict) -> Dict:
        """Peripheral and autonomic neuropathy risk"""
        
        # Temperature variability as early indicator
        temp_variability = data.get('temp_std', 1.0)
        
        # HRV decline as autonomic marker
        hrv = data.get('hrv_rmssd', 40)
        hrv_decline = max(0, (50 - hrv) / 50)  # Normalized decline
        
        # Glucose exposure
        cumulative_hyper = data.get('tir_hyper', 20) * data.get('diabetes_duration_years', 5)
        
        # Calculate risk scores
        peripheral_risk = (
            0.4 * cumulative_hyper / 500 +
            0.3 * temp_variability / 3 +
            0.3 * data.get('diabetes_duration_years', 5) / 20
        )
        
        autonomic_risk = (
            0.5 * hrv_decline +
            0.3 * data.get('lbgi', 2) / 10 +  # Hypoglycemia exposure
            0.2 * data.get('diabetes_duration_years', 5) / 20
        )
        
        overall_risk = max(peripheral_risk, autonomic_risk)
        risk_level = self._get_risk_level(overall_risk)
        
        # Early detection markers
        early_markers = {
            'temperature_dysregulation': temp_variability > 2.0,
            'hrv_reduction': hrv < 30,
            'postural_changes': data.get('orthostatic_risk', False)
        }
        
        return {
            'peripheral_risk': peripheral_risk,
            'autonomic_risk': autonomic_risk,
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'early_markers': early_markers,
            'recommendations': self._get_neuropathy_recommendations(risk_level, early_markers),
            'screening': 'Monofilament test' if peripheral_risk > 0.5 else 'Annual foot exam'
        }
    
    def calculate_cardiovascular_risk(self, data: Dict) -> Dict:
        """Cardiovascular disease risk with E4 integration"""
        
        # HRV as primary marker
        hrv_risk = 1 - min(1, data.get('hrv_rmssd', 40) / 50)
        
        # Glucose-related risk
        glucose_risk = (
            0.3 * data.get('adrr', 20) / 40 +  # Overall glucose risk
            0.3 * data.get('tir_hyper', 20) / 100 +  # Hyperglycemia
            0.2 * data.get('mage', 40) / 100 +  # Variability
            0.2 * data.get('hbgi', 5) / 20  # High glucose index
        )
        
        # Stress and activity
        stress_risk = data.get('stress_level', 5) / 10
        sedentary_risk = 1 - min(1, data.get('activity_level', 2) / 5)
        
        # Age factor
        age_risk = max(0, (data.get('age', 40) - 40)) / 40
        
        # Composite risk
        composite_risk = (
            0.3 * hrv_risk +
            0.25 * glucose_risk +
            0.15 * stress_risk +
            0.15 * sedentary_risk +
            0.15 * age_risk
        )
        
        risk_level = self._get_risk_level(composite_risk)
        
        # 10-year risk estimation (simplified ASCVD)
        ten_year_risk = min(0.5, composite_risk * 0.6)
        
        return {
            'risk_score': composite_risk,
            'risk_level': risk_level,
            '10_year_risk_percent': ten_year_risk * 100,
            'autonomic_dysfunction': hrv_risk > 0.7,
            'components': {
                'hrv_risk': hrv_risk,
                'glucose_risk': glucose_risk,
                'stress_risk': stress_risk,
                'sedentary_risk': sedentary_risk
            },
            'recommendations': self._get_cvd_recommendations(risk_level, composite_risk),
            'intervention_needed': composite_risk > 0.6
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical level"""
        
        if risk_score < self.risk_thresholds['low']:
            return 'low'
        elif risk_score < self.risk_thresholds['moderate']:
            return 'moderate'
        elif risk_score < self.risk_thresholds['high']:
            return 'high'
        else:
            return 'critical'
    
    def _calculate_overall_risk(self, complications: Dict) -> Dict:
        """Calculate overall complication risk"""
        
        risks = [
            complications['retinopathy']['risk_score'],
            complications['nephropathy']['risk_score'],
            complications['neuropathy']['overall_risk'],
            complications['cardiovascular']['risk_score']
        ]
        
        overall_score = np.mean(risks)
        max_risk = max(risks)
        
        return {
            'overall_score': overall_score,
            'max_risk': max_risk,
            'risk_level': self._get_risk_level(max_risk),
            'highest_risk_complication': self._get_highest_risk(complications)
        }
    
    def _get_highest_risk(self, complications: Dict) -> str:
        """Identify highest risk complication"""
        
        risks = {
            'retinopathy': complications['retinopathy']['risk_score'],
            'nephropathy': complications['nephropathy']['risk_score'],
            'neuropathy': complications['neuropathy']['overall_risk'],
            'cardiovascular': complications['cardiovascular']['risk_score']
        }
        
        return max(risks, key=risks.get)
    
    def _get_retinopathy_recommendations(self, risk_level: str, hba1c: float) -> list:
        """Get retinopathy prevention recommendations"""
        
        recs = []
        
        if hba1c > 7:
            recs.append(f"Improve glycemic control (current HbA1c: {hba1c:.1f}%)")
        
        if risk_level in ['high', 'critical']:
            recs.append("Schedule immediate ophthalmology consultation")
            recs.append("Consider more aggressive glucose management")
        elif risk_level == 'moderate':
            recs.append("Schedule ophthalmology screening within 3 months")
        
        recs.append("Monitor and control blood pressure")
        recs.append("Reduce glucose variability")
        
        return recs
    
    def _get_nephropathy_recommendations(self, risk_factors: Dict) -> list:
        """Get nephropathy prevention recommendations"""
        
        recs = []
        
        if risk_factors['poor_glycemic_control']:
            recs.append("Improve time in range to >70%")
        
        if risk_factors['hypertension']:
            recs.append("Control blood pressure (<130/80)")
            recs.append("Consider ACE inhibitor or ARB")
        
        if risk_factors['high_variability']:
            recs.append("Reduce glucose variability (MAGE <60)")
        
        recs.append("Monitor urine albumin levels")
        recs.append("Maintain adequate hydration")
        
        return recs
    
    def _get_neuropathy_recommendations
