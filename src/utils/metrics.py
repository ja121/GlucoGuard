import streamlit as st
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pandas as pd

class GlucoGuardDashboard:
    """Real-time monitoring dashboard"""
    
    def __init__(self, monitor_system):
        self.monitor = monitor_system
        self.session_data = []
        
    def run(self):
        st.set_page_config(page_title="GlucoGuard AI", layout="wide")
        
        # Header
        st.title("🛡️ GlucoGuard - AI Glucose & Complication Monitoring")
        st.markdown("**Mandatory Wearable Integration: Empatica E4 + Dexcom G6**")
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_glucose = st.empty()
            trend_arrow = st.empty()
        
        with col2:
            prediction_30min = st.empty()
            confidence = st.empty()
        
        with col3:
            risk_status = st.empty()
            time_in_range = st.empty()
        
        with col4:
            stress_level = st.empty()
            activity_level = st.empty()
        
        # Main visualization area
        glucose_plot = st.empty()
        
        # Alert section
        alert_container = st.container()
        
        # Complication risk panel
        st.sidebar.header("📊 Long-term Complication Risk")
        retinopathy_risk = st.sidebar.empty()
        nephropathy_risk = st.sidebar.empty()
        neuropathy_risk = st.sidebar.empty()
        cardiovascular_risk = st.sidebar.empty()
        
        # Update loop
        while True:
            # Get latest data
            latest = self.monitor.get_latest_state()
            
            # Update metrics
            current_glucose.metric(
                "Current Glucose",
                f"{latest['glucose']:.0f} mg/dL",
                f"{latest['glucose_change']:.1f}"
            )
            
            trend_arrow.markdown(
                f"### Trend: {latest['trend_arrow_emoji']}"
            )
            
            prediction_30min.metric(
                "30-min Prediction",
                f"{latest['prediction_30min']:.0f} mg/dL",
                f"{latest['prediction_confidence']:.0f}%"
            )
            
            # Risk assessment
            risk_color = self._get_risk_color(latest['risk_level'])
            risk_status.markdown(
                f"<h3 style='color:{risk_color}'>{latest['risk_level']}</h3>",
                unsafe_allow_html=True
            )
            
            # E4 vitals
            stress_level.metric(
                "Stress Level",
                f"{latest['stress_level']:.1f}/10",
                "High" if latest['stress_level'] > 7 else "Normal"
            )
            
            activity_level.metric(
                "Activity",
                latest['activity_category'],
                f"{latest['steps_today']} steps"
            )
            
            # Plot glucose with predictions
            self._update_glucose_plot(glucose_plot, latest)
            
            # Display alerts
            self._display_alerts(alert_container, latest['alerts'])
            
            # Update complication risks (daily)
            if latest.get('complication_update'):
                self._update_complication_panel(
                    retinopathy_risk,
                    nephropathy_risk,
                    neuropathy_risk,
                    cardiovascular_risk,
                    latest['complications']
                )
            
            time.sleep(60)  # Update every minute
    
    def _update_glucose_plot(self, container, data):
        """Update the main glucose visualization"""
        fig = go.Figure()
        
        # Historical glucose
        fig.add_trace(go.Scatter(
            x=data['timestamps'],
            y=data['glucose_history'],
            name='Glucose',
            line=dict(color='blue', width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=data['future_timestamps'],
            y=data['predictions'],
            name='Predicted',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        # Confidence bands
        fig.add_trace(go.Scatter(
            x=data['future_timestamps'],
            y=data['upper_bound'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=data['future_timestamps'],
            y=data['lower_bound'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence',
            fillcolor='rgba(255,165,0,0.2)'
        ))
        
        # Add threshold lines
        fig.add_hline(y=70, line_dash="dot", line_color="red", 
                     annotation_text="Hypo threshold")
        fig.add_hline(y=180, line_dash="dot", line_color="orange", 
                     annotation_text="Hyper threshold")
        
        # Add E4 context as subplot
        fig.add_trace(go.Scatter(
            x=data['timestamps'],
            y=data['stress_history'],
            name='Stress',
            yaxis='y2',
            line=dict(color='purple', width=1)
        ))
        
        fig.update_layout(
            title="Real-time Glucose Monitoring with E4 Context",
            xaxis_title="Time",
            yaxis_title="Glucose (mg/dL)",
            yaxis2=dict(
                title="Stress Level",
                overlaying='y',
                side='right'
            ),
            height=500
        )
        
        container.plotly_chart(fig, use_container_width=True)



class ProductionFeatures:
    """Production enhancements for reliability"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        self.model_monitor = ModelMonitor()
        
class ErrorHandler:
    """Robust error handling"""
    
    def handle_missing_e4_data(self, e4_buffer):
        """Handle Empatica E4 disconnections"""
        if self._is_disconnected(e4_buffer):
            # Use last known good values
            return self._get_last_valid_e4_data()
        
        if self._is_partial_data(e4_buffer):
            # Interpolate missing sensors
            return self._interpolate_e4_data(e4_buffer)
        
        return e4_buffer
    
    def handle_cgm_gaps(self, cgm_data):
        """Handle CGM data gaps"""
        gap_size = self._detect_gap_size(cgm_data)
        
        if gap_size < 3:  # <15 minutes
            return self._linear_interpolation(cgm_data)
        elif gap_size < 12:  # <1 hour
            return self._spline_interpolation(cgm_data)
        else:
            # Too large - mark as unreliable
            return self._mark_unreliable(cgm_data)

class DataValidator:
    """Validate incoming sensor data"""
    
    def validate_glucose(self, value):
        """Check if glucose value is physiologically possible"""
        if not 20 <= value <= 600:
            raise ValueError(f"Glucose {value} outside possible range")
        
        return value
    
    def validate_e4_signals(self, e4_data):
        """Validate E4 sensor readings"""
        checks = {
            'eda': 0.01 <= e4_data['eda'] <= 100,
            'hr': 30 <= e4_data['hr'] <= 220,
            'temp': 25 <= e4_data['temp'] <= 42,
            'activity': e4_data['activity'] >= 0
        }
        
        if not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            raise ValueError(f"Invalid E4 data: {failed}")
        
        return True

class ModelMonitor:
    """Monitor model performance in production"""
    
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.drift_detector = DriftDetector()
        
    def log_prediction(self, pred, features):
        """Log predictions for monitoring"""
        self.predictions.append({
            'timestamp': datetime.now(),
            'prediction': pred,
            'features': features
        })
        
    def check_drift(self, current_features):
        """Detect feature/concept drift"""
        if self.drift_detector.detect(current_features):
            return {
                'drift_detected': True,
                'features_affected': self.drift_detector.get_drifted_features(),
                'action': 'retrain_recommended'
            }
        return {'drift_detected': False}



# test_glucoguard_complete.py

def run_complete_validation():
    """Complete system validation"""
    
    print("="*60)
    print("🧪 GlucoGuard Complete System Validation")
    print("="*60)
    
    # 1. Data Pipeline Test
    print("\n1️⃣ Testing Data Pipeline...")
    try:
        data = load_big_ideas_dataset("path/to/data")
        assert 'cgm' in data and 'e4' in data
        print("   ✅ Data loading successful")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return
    
    # 2. Feature Engineering Test
    print("\n2️⃣ Testing Feature Engineering...")
    e4_processor = EmpaticaE4Processor(BIGIDEAsConfig())
    cgm_engineer = CGMFeatureEngineer()
    
    e4_features = e4_processor.process_all(data['e4'])
    cgm_features = cgm_engineer.extract_all(data['cgm'])
    
    print(f"   ✅ Extracted {len(e4_features.columns)} E4 features")
    print(f"   ✅ Extracted {len(cgm_features.columns)} CGM features")
    
    # 3. Model Performance Test
    print("\n3️⃣ Testing Model Performance...")
    X = pd.concat([cgm_features, e4_features], axis=1)
    y = data['glucose_target_30min']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    predictor = EnhancedGlucosePredictor()
    model = predictor.train_phase1_baseline(X_train[:30], y_train[:30])
    
    predictions = model.predict(X_test)
    mard = calculate_mard(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"   MARD: {mard:.2f}% (Target: <10%)")
    print(f"   R²: {r2:.3f} (Target: >0.70)")
    
    if mard < 10 and r2 > 0.70:
        print("   ✅ Model performance meets targets!")
    else:
        print("   ⚠️ Model needs optimization")
    
    # 4. Alert System Test
    print("\n4️⃣ Testing Alert System...")
    alert_system = ContextAwareAlertSystem()
    
    test_states = [
        {'glucose': 50, 'heart_rate': 110, 'stress_level': 8},
        {'glucose': 250, 'heart_rate': 90, 'activity_level': 2},
        {'glucose': 120, 'heart_rate': 70, 'stress_level': 3}
    ]
    
    for state in test_states:
        alerts = alert_system.evaluate_current_state(state)
        print(f"   Glucose={state['glucose']}: {len(alerts)} alerts generated")
    
    # 5. Complication Prediction Test
    print("\n5️⃣ Testing Complication Predictions...")
    comp_predictor = ComplicationPredictor()
    
    sample_patient = {
        'glucose': list(data['cgm']['glucose'][:288]),
        'time_in_hyper': 25,
        'mage': 65,
        'diabetes_duration_years': 5,
        'hrv_rmssd': 30
    }
    
    complications = {
        'retinopathy': comp_predictor.calculate_retinopathy_risk(sample_patient),
        'nephropathy': comp_predictor.calculate_nephropathy_risk(sample_patient),
        'neuropathy': comp_predictor.calculate_neuropathy_risk(sample_patient),
        'cardiovascular': comp_predictor.calculate_cardiovascular_risk(sample_patient)
    }
    
    for comp, risk in complications.items():
        print(f"   {comp}: Risk Score = {risk['risk_score']:.1f}")
    
    print("\n" + "="*60)
    print("🎉 Validation Complete!")
    print("="*60)

if __name__ == "__main__":
    run_complete_validation()
