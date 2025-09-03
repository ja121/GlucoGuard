import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from collections import deque

class ContextAwareAlertSystem:
    """Generate context-aware alerts based on CGM + E4 data"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.alert_history = deque(maxlen=100)
        self.cooldown_tracker = {}
        
    def _initialize_rules(self) -> List[Dict]:
        """Initialize comprehensive alert rules"""
        
        rules = [
            # Critical hypoglycemia rules
            {
                'rule_id': 'R-00001',
                'name': 'severe_hypo_tachycardia',
                'cgm_check': lambda d: d['glucose'] < 54 or d['trend_arrow'] <= -2,
                'e4_check': lambda d: d.get('heart_rate', 70) > 100,
                'context': lambda d: d.get('context') == 'rest',
                'risk_level': 'critical',
                'message': '⚠️ CRITICAL: Very low sugar detected. Eat 30g carbs NOW. Heart rate elevated - sit down and rest.',
                'action': 'immediate_carbs',
                'cooldown': 15
            },
            {
                'rule_id': 'R-00002',
                'name': 'severe_hypo_bradycardia',
                'cgm_check': lambda d: d['glucose'] < 54,
                'e4_check': lambda d: d.get('heart_rate', 70) < 50,
                'context': lambda d: True,
                'risk_level': 'critical',
                'message': '⚠️ CRITICAL: Very low sugar with slow heart rate. Eat carbs and lie down. Consider calling for help.',
                'action': 'emergency',
                'cooldown': 15
            },
            {
                'rule_id': 'R-00003',
                'name': 'severe_hypo_high_stress',
                'cgm_check': lambda d: d['glucose'] < 54,
                'e4_check': lambda d: d.get('stress_level', 5) > 8,
                'context': lambda d: True,
                'risk_level': 'critical',
                'message': '⚠️ CRITICAL: Low sugar with high stress. Take glucose tablets. Practice calm breathing.',
                'action': 'immediate_carbs',
                'cooldown': 15
            },
            
            # Moderate hypoglycemia rules
            {
                'rule_id': 'R-00010',
                'name': 'moderate_hypo_falling',
                'cgm_check': lambda d: 54 <= d['glucose'] < 70 and d['trend_arrow'] <= -1,
                'e4_check': lambda d: True,
                'context': lambda d: True,
                'risk_level': 'high',
                'message': '⬇️ Sugar falling below 70. Take 15g carbs and recheck in 15 minutes.',
                'action': 'carbs_15g',
                'cooldown': 20
            },
            {
                'rule_id': 'R-00011',
                'name': 'hypo_during_exercise',
                'cgm_check': lambda d: d['glucose'] < 90 and d['trend_arrow'] <= -1,
                'e4_check': lambda d: d.get('activity_level', 0) > 3,
                'context': lambda d: d.get('context') == 'exercise',
                'risk_level': 'high',
                'message': '🏃 Low sugar during exercise. Stop activity and take fast carbs.',
                'action': 'stop_exercise',
                'cooldown': 30
            },
            {
                'rule_id': 'R-00012',
                'name': 'nocturnal_hypo_risk',
                'cgm_check': lambda d: d['glucose'] < 100 and d['trend_arrow'] <= 0,
                'e4_check': lambda d: d.get('sleep_probability', 0) > 0.7,
                'context': lambda d: d.get('context') == 'sleep',
                'risk_level': 'high',
                'message': '🌙 Risk of nighttime low. Consider a small snack before continuing sleep.',
                'action': 'bedtime_snack',
                'cooldown': 60
            },
            
            # Hyperglycemia rules
            {
                'rule_id': 'R-00020',
                'name': 'severe_hyper_rising',
                'cgm_check': lambda d: d['glucose'] > 250 and d['trend_arrow'] >= 2,
                'e4_check': lambda d: True,
                'context': lambda d: True,
                'risk_level': 'high',
                'message': '⬆️ HIGH: Sugar above 250 and rising rapidly. Check ketones and consider correction.',
                'action': 'check_ketones',
                'cooldown': 30
            },
            {
                'rule_id': 'R-00021',
                'name': 'persistent_hyperglycemia',
                'cgm_check': lambda d: d['glucose'] > 180,
                'e4_check': lambda d: d.get('stress_level', 5) < 7,
                'context': lambda d: d.get('time_above_180', 0) > 120,
                'risk_level': 'moderate',
                'message': '📈 Sugar high for >2 hours. Review insulin and consider correction.',
                'action': 'insulin_review',
                'cooldown': 60
            },
            
            # Pattern-based rules
            {
                'rule_id': 'R-00030',
                'name': 'dawn_phenomenon',
                'cgm_check': lambda d: d['glucose'] > 150 and d.get('hour', 12) in range(4, 8),
                'e4_check': lambda d: d.get('sleep_probability', 0) > 0.5,
                'context': lambda d: True,
                'risk_level': 'low',
                'message': '🌅 Dawn phenomenon detected. Consider adjusting basal rates.',
                'action': 'basal_adjustment',
                'cooldown': 1440  # Once daily
            },
            {
                'rule_id': 'R-00031',
                'name': 'post_meal_spike',
                'cgm_check': lambda d: d['glucose'] > 200 and d['trend_arrow'] >= 1,
                'e4_check': lambda d: True,
                'context': lambda d: d.get('time_since_meal', 999) < 120,
                'risk_level': 'low',
                'message': '🍽️ Post-meal spike. Consider pre-bolus timing or carb ratio adjustment.',
                'action': 'meal_timing',
                'cooldown': 180
            },
            
            # Stress-glucose interaction
            {
                'rule_id': 'R-00040',
                'name': 'stress_hyperglycemia',
                'cgm_check': lambda d: d['glucose'] > 160,
                'e4_check': lambda d: d.get('stress_level', 5) > 8,
                'context': lambda d: True,
                'risk_level': 'moderate',
                'message': '😰 High stress affecting glucose. Try relaxation techniques.',
                'action': 'stress_management',
                'cooldown': 60
            },
            
            # Trend-based predictions
            {
                'rule_id': 'R-00050',
                'name': 'predicted_hypo_30min',
                'cgm_check': lambda d: d.get('predicted_glucose_30min', 100) < 70,
                'e4_check': lambda d: True,
                'context': lambda d: d['glucose'] > 70,
                'risk_level': 'moderate',
                'message': '🔮 Low predicted in 30 min. Take preventive action now.',
                'action': 'preventive_carbs',
                'cooldown': 30
            }
        ]
        
        return rules
    
    def evaluate_current_state(self, state_data: Dict) -> List[Dict]:
        """Evaluate all rules against current state"""
        
        alerts = []
        current_time = datetime.now()
        
        # Add derived features to state
        state_data = self._enrich_state_data(state_data)
        
        for rule in self.rules:
            # Check cooldown
            if self._is_in_cooldown(rule['rule_id'], current_time):
                continue
            
            # Check rule conditions
            if self._check_rule(rule, state_data):
                alert = self._generate_alert(rule, state_data, current_time)
                alerts.append(alert)
                
                # Update cooldown
                self.cooldown_tracker[rule['rule_id']] = current_time
        
        # Sort by priority
        alerts = sorted(alerts, key=lambda x: self._get_priority_score(x), reverse=True)
        
        # Store in history
        self.alert_history.extend(alerts)
        
        return alerts
    
    def _check_rule(self, rule: Dict, state: Dict) -> bool:
        """Check if rule conditions are met"""
        
        try:
            cgm_match = rule['cgm_check'](state)
            e4_match = rule['e4_check'](state)
            context_match = rule['context'](state)
            
            return cgm_match and e4_match and context_match
            
        except Exception:
            return False
    
    def _generate_alert(self, rule: Dict, state: Dict, timestamp: datetime) -> Dict:
        """Generate detailed alert from rule"""
        
        # Get trend arrow emoji
        trend_emoji = self._get_trend_emoji(state.get('trend_arrow', 0))
        
        alert = {
            'timestamp': timestamp.isoformat(),
            'rule_id': rule['rule_id'],
            'name': rule['name'],
            'risk_level': rule['risk_level'],
            'message': rule['message'],
            'action': rule['action'],
            'trend': trend_emoji,
            'current_glucose': state.get('glucose'),
            'predicted_30min': state.get('predicted_glucose_30min'),
            'context': {
                'heart_rate': state.get('heart_rate'),
                'stress_level': state.get('stress_level'),
                'activity': state.get('context'),
                'confidence': state.get('prediction_confidence', 0)
            },
            'recommendations': self._get_recommendations(rule, state)
        }
        
        return alert
    
    def _enrich_state_data(self, state: Dict) -> Dict:
        """Add derived features to state"""
        
        # Add time-based features
        now = datetime.now()
        state['hour'] = now.hour
        state['is_night'] = now.hour < 6 or now.hour > 22
        
        # Add trend interpretation
        if 'glucose_history' in state and len(state['glucose_history']) > 6:
            recent = state['glucose_history'][-6:]
            state['trend_arrow'] = self._calculate_trend(recent)
        
        return state
    
    def _calculate_trend(self, glucose_values: List[float]) -> int:
        """Calculate trend arrow from recent values"""
        
        if len(glucose_values) < 2:
            return 0
        
        rate = (glucose_values[-1] - glucose_values[0]) / (len(glucose_values) * 5)
        
        if rate > 3:
            return 3
        elif rate > 2:
            return 2
        elif rate > 1:
            return 1
        elif rate < -3:
            return -3
        elif rate < -2:
            return -2
        elif rate < -1:
            return -1
        else:
            return 0
    
    def _get_trend_emoji(self, trend: int) -> str:
        """Convert trend to emoji"""
        
        emojis = {
            3: '🔼🔼🔼',
            2: '🔼🔼',
            1: '🔼',
            0: '➡️',
            -1: '🔽',
            -2: '🔽🔽',
            -3: '🔽🔽🔽'
        }
        
        return emojis.get(trend, '➡️')
    
    def _is_in_cooldown(self, rule_id: str, current_time: datetime) -> bool:
        """Check if rule is in cooldown period"""
        
        if rule_id not in self.cooldown_tracker:
            return False
        
        last_alert = self.cooldown_tracker[rule_id]
        
        # Find cooldown duration for this rule
        rule = next((r for r in self.rules if r['rule_id'] == rule_id), None)
        if not rule:
            return False
        
        cooldown_minutes = rule.get('cooldown', 30)
        
        return (current_time - last_alert) < timedelta(minutes=cooldown_minutes)
    
    def _get_priority_score(self, alert: Dict) -> int:
        """Calculate alert priority score"""
        
        scores = {
            'critical': 100,
            'high': 75,
            'moderate': 50,
            'low': 25
        }
        
        return scores.get(alert['risk_level'], 0)
    
    def _get_recommendations(self, rule: Dict, state: Dict) -> List[str]:
        """Get specific recommendations based on rule and state"""
        
        recs = []
        
        action = rule['action']
        
        if action == 'immediate_carbs':
            recs.append("Take 30g fast-acting carbohydrates immediately")
            recs.append("Recheck glucose in 15 minutes")
            recs.append("Do not take insulin until glucose > 100")
            
        elif action == 'carbs_15g':
            recs.append("Take 15g carbohydrates (3-4 glucose tablets)")
            recs.append("Wait 15 minutes and recheck")
            
        elif action == 'check_ketones':
            recs.append("Check blood or urine ketones")
            recs.append("Drink water to stay hydrated")
            recs.append("Contact healthcare provider if ketones are moderate/high")
            
        elif action == 'insulin_review':
            recs.append("Review recent insulin doses")
            recs.append("Consider correction factor adjustment")
            
        elif action == 'stress_management':
            recs.append("Practice deep breathing (4-7-8 technique)")
            recs.append("Take a short walk if possible")
            recs.append("Consider meditation or calming music")
        
        return recs
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent alerts"""
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alert_history 
                        if datetime.fromisoformat(a['timestamp']) > cutoff]
        
        summary = {
            'total_alerts': len(recent_alerts),
            'critical': sum(1 for a in recent_alerts if a['risk_level'] == 'critical'),
            'high': sum(1 for a in recent_alerts if a['risk_level'] == 'high'),
            'moderate': sum(1 for a in recent_alerts if a['risk_level'] == 'moderate'),
            'low': sum(1 for a in recent_alerts if a['risk_level'] == 'low'),
            'most_common': self._get_most_common_alert(recent_alerts),
            'trend': self._analyze_alert_trend(recent_alerts)
        }
        
        return summary
    
    def _get_most_common_alert(self, alerts: List[Dict]) -> str:
        """Find most common alert type"""
        
        if not alerts:
            return "None"
        
        from collections import Counter
        alert_types = [a['name'] for a in alerts]
        most_common = Counter(alert_types).most_common(1)
        
        return most_common[0][0] if most_common else "None"
    
    def _analyze_alert_trend(self, alerts: List[Dict]) -> str:
        """Analyze trend in alerts"""
        
        if len(alerts) < 2:
            return "stable"
        
        # Compare first half to second half
        mid = len(alerts) // 2
        first_half = alerts[:mid]
        second_half = alerts[mid:]
        
        first_critical = sum(1 for a in first_half if a['risk_level'] in ['critical', 'high'])
        second_critical = sum(1 for a in second_half if a['risk_level'] in ['critical', 'high'])
        
        if second_critical > first_critical * 1.5:
            return "worsening"
        elif second_critical < first_critical * 0.5:
            return "improving"
        else:
            return "stable"
