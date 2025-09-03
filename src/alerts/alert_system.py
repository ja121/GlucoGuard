class ContextAwareAlertSystem:
    """Generate personalized alerts based on CGM + E4 context"""
    
    def __init__(self):
        self.rules = self._load_rule_database()
        self.alert_history = []
        
    def _load_rule_database(self):
        """Load the alert rules you specified"""
        rules = [
            {
                'rule_id': 'R-00001',
                'condition': 'severe_hypo_tachycardia',
                'cgm_criteria': lambda g: g < 54 or self._fast_drop(g),
                'e4_criteria': lambda hr: hr > 100,
                'context': 'rest_day',
                'risk_level': 'critical',
                'message': '⚠️ Very low sugar. Eat carbs now. Heart rate is high. Rest and drink water.'
            },
            {
                'rule_id': 'R-00002',
                'condition': 'severe_hypo_bradycardia',
                'cgm_criteria': lambda g: g < 54,
                'e4_criteria': lambda hr: hr < 50,
                'context': 'rest_day',
                'risk_level': 'critical',
                'message': '⚠️ Very low sugar. Eat carbs now. Heart rate is low. Sit or lie down if dizzy.'
            },
            # Add all your rules here...
        ]
        return rules
    
    def evaluate_current_state(self, current_data):
        """Check all rules against current state"""
        alerts = []
        
        for rule in self.rules:
            if self._check_rule(rule, current_data):
                alert = self._generate_alert(rule, current_data)
                alerts.append(alert)
                
        # Sort by priority
        alerts = sorted(alerts, key=lambda x: self._priority_score(x), reverse=True)
        
        # Store in history
        self.alert_history.extend(alerts)
        
        return alerts
    
    def _generate_alert(self, rule, data):
        """Create detailed alert with trend arrows"""
        trend = self._calculate_trend_arrow(data['glucose_history'])
        
        alert = {
            'timestamp': data['timestamp'],
            'rule_id': rule['rule_id'],
            'risk_level': rule['risk_level'],
            'trend_arrow': self._get_arrow_emoji(trend),
            'current_glucose': data['glucose'],
            'predicted_30min': data.get('glucose_pred_30min'),
            'message': rule['message'],
            'vitals': {
                'hr': data.get('heart_rate'),
                'hrv': data.get('hrv_rmssd'),
                'stress': data.get('stress_level'),
                'activity': data.get('activity_level')
            },
            'action_items': self._get_action_items(rule, data)
        }
        
        return alert
    
    def _get_arrow_emoji(self, trend):
        """Convert trend to emoji arrows"""
        arrows = {
            3: '🔼🔼🔼',
            2: '🔼🔼',
            1: '🔼',
            0: '➡️',
            -1: '🔽',
            -2: '🔽🔽',
            -3: '🔽🔽🔽'
        }
        return arrows.get(trend, '➡️')
