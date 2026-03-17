#!/usr/bin/env python3
"""
Web Interface for Digital Twin System
Flask-based web application matching the screenshot design
"""

from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirofish_engine.digital_twin_simulator import DigitalTwinSimulator
from web_app.report_parser import MedicalReportParser
from web_app.llm_service import llm_service
import json

app = Flask(__name__)
parser = MedicalReportParser()


@app.route('/')
def index():
    """Main page with risk prediction calculator"""
    return render_template('index.html')


@app.route('/api/parse_report', methods=['POST'])
def parse_report():
    """Parse medical report text"""
    try:
        data = request.json
        report_text = data.get('report_text', '')
        
        # Parse the report
        patient_data = parser.parse_report(report_text)
        summary = parser.format_extraction_summary(patient_data)
        
        return jsonify({
            'success': True,
            'patient_data': patient_data,
            'summary': summary,
            'completeness': parser.get_completeness_score(patient_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for risk prediction"""
    try:
        # Get patient data from form or parsed report
        patient_data = request.json
        
        # Create simulator
        simulator = DigitalTwinSimulator(patient_data)
        
        # Run 5-year simulation
        results = simulator.simulate(years=5, timestep='month')
        
        # Extract key predictions
        predictions = results['disease_predictions']
        
        # LLM-powered interpretation and recommendations
        llm_analysis = {}
        try:
            # Get LLM interpretation
            llm_analysis['patient_context'] = llm_service.analyze_patient(patient_data)
            llm_analysis['explanation'] = llm_service.explain_results(
                patient_data, 
                results.get('trajectory', []), 
                results['disease_predictions']
            )
            llm_analysis['recommendations'] = llm_service.get_recommendations(
                patient_data,
                results['disease_predictions']
            )
            llm_analysis['guidelines'] = llm_service.get_guidelines(
                patient_data,
                results['disease_predictions']
            )
        except Exception as e:
            print(f"LLM analysis error: {e}")
            llm_analysis['error'] = str(e)
        
        # Return results
        return jsonify({
            'success': True,
            'predictions': results['disease_predictions'],
            'current_state': results['current_state'],
            'interventions': results.get('interventions', []),
            'llm_insights': llm_analysis
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/demo/<demo_type>')
def demo(demo_type):
    """Load demo patient data"""
    demos = {
        'healthy': {
            'patient_id': 'DEMO_HEALTHY',
            'age': 30,
            'sex': 'M',
            'height': 175,
            'weight': 70,
            'lifestyle': {
                'physical_activity': 'vigorous',
                'diet_quality': 'good',
                'smoking_status': 'never'
            }
        },
        'prediabetic': {
            'patient_id': 'DEMO_PREDIABETIC',
            'age': 45,
            'sex': 'F',
            'height': 165,
            'weight': 80,
            'fasting_glucose': 110,
            'hba1c': 5.9,
            'blood_pressure': {
                'systolic': 135,
                'diastolic': 85
            },
            'total_cholesterol': 220,
            'ldl_cholesterol': 140,
            'hdl_cholesterol': 45,
            'lifestyle': {
                'physical_activity': 'light',
                'diet_quality': 'fair'
            }
        }
    }
    
    return jsonify(demos.get(demo_type, {}))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
