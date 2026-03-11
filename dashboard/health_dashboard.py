import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from synthetic_data_generator.patient_population_generator import PatientPopulationGenerator

st.set_page_config(
    page_title="Health Digital Twin Dashboard",
    page_icon="🏥",
    layout="wide"
)

API_URL = "http://localhost:8000"


def main():
    st.title("🏥 Health Digital Twin Prediction Platform")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Patient Evaluation",
        "🔮 Risk Prediction",
        "💊 Intervention Simulation",
        "📈 Population Analytics"
    ])
    
    with tab1:
        patient_evaluation_tab()
    
    with tab2:
        risk_prediction_tab()
    
    with tab3:
        intervention_simulation_tab()
    
    with tab4:
        population_analytics_tab()


def patient_evaluation_tab():
    st.header("Patient Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Demographics")
        patient_id = st.text_input("Patient ID", value="P00000001")
        age = st.slider("Age", 18, 90, 55)
        gender = st.selectbox("Gender", ["male", "female"])
        bmi = st.number_input("BMI", 15.0, 50.0, 28.0, 0.1)
    
    with col2:
        st.subheader("Vital Signs")
        systolic_bp = st.number_input("Systolic BP", 80, 200, 140)
        diastolic_bp = st.number_input("Diastolic BP", 50, 130, 90)
        heart_rate = st.number_input("Heart Rate", 40, 150, 75)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Lab Results")
        glucose = st.number_input("Glucose (mmol/L)", 3.0, 20.0, 6.5, 0.1)
        hba1c = st.number_input("HbA1c (%)", 4.0, 14.0, 6.0, 0.1)
        total_chol = st.number_input("Total Cholesterol (mmol/L)", 2.0, 12.0, 6.5, 0.1)
    
    with col4:
        st.subheader("Lifestyle")
        smoking = st.selectbox("Smoking Status", ["never", "former", "current"])
        exercise = st.number_input("Exercise (hours/week)", 0.0, 20.0, 2.0, 0.5)
        diet_quality = st.slider("Diet Quality (1-10)", 1, 10, 5)
    
    if st.button("🔍 Evaluate Patient", type="primary"):
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "glucose_mmol_l": glucose,
            "hba1c_percent": hba1c,
            "total_cholesterol_mmol_l": total_chol,
            "smoking_status": smoking,
            "exercise_hours_per_week": exercise,
            "diet_quality_score": diet_quality
        }
        
        with st.spinner("Evaluating patient..."):
            try:
                response = requests.post(f"{API_URL}/evaluate", json=patient_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Evaluation Complete")
                    
                    st.subheader("Multi-Agent Evaluation Results")
                    
                    evaluations = result['evaluation']['individual_evaluations']
                    
                    for agent_name, eval_data in evaluations.items():
                        with st.expander(f"📋 {agent_name}", expanded=True):
                            if 'error' not in eval_data:
                                col_a, col_b, col_c = st.columns(3)
                                col_a.metric("Risk Score", f"{eval_data.get('risk_score', 0):.1%}")
                                col_b.metric("Risk Level", eval_data.get('risk_level', 'N/A').upper())
                                col_c.metric("Confidence", f"{eval_data.get('confidence', 0):.1%}")
                                
                                st.write("**Findings:**")
                                for finding in eval_data.get('findings', []):
                                    st.write(f"- {finding}")
                                
                                st.write("**Recommendations:**")
                                for rec in eval_data.get('recommendations', []):
                                    st.write(f"- {rec}")
                    
                    consensus = result['evaluation']['consensus']
                    st.info(f"**Consensus Risk Score:** {consensus.get('consensus', 0):.2%} (Confidence: {consensus.get('confidence', 0):.2%})")
                
                else:
                    st.error(f"API Error: {response.status_code}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure the API server is running: `python api/api_server.py`")


def risk_prediction_tab():
    st.header("Disease Risk Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Patient Data")
        
        patient_id = st.text_input("Patient ID", value="P00000002", key="risk_patient_id")
        
        col_a, col_b, col_c = st.columns(3)
        age = col_a.number_input("Age", 18, 90, 55, key="risk_age")
        gender = col_b.selectbox("Gender", ["male", "female"], key="risk_gender")
        bmi = col_c.number_input("BMI", 15.0, 50.0, 32.0, key="risk_bmi")
        
        time_horizon = st.slider("Prediction Time Horizon (years)", 1, 20, 10)
    
    with col2:
        st.subheader("Quick Stats")
        st.metric("Systolic BP", "145 mmHg")
        st.metric("HbA1c", "6.2%")
        st.metric("Cholesterol", "6.5 mmol/L")
    
    if st.button("🔮 Predict Risk", type="primary"):
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "systolic_bp": 145,
            "hba1c_percent": 6.2,
            "total_cholesterol_mmol_l": 6.5,
            "smoking_status": "current"
        }
        
        with st.spinner("Calculating risk predictions..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict_risk",
                    json=patient_data,
                    params={"time_horizon_years": time_horizon}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success(f"✅ {time_horizon}-Year Risk Predictions")
                    
                    risks = result['individual_risks']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric(
                        "Cardiovascular Disease",
                        f"{risks['cvd']['risk_percentage']:.1f}%",
                        delta=risks['cvd']['risk_level']
                    )
                    
                    col2.metric(
                        "Type 2 Diabetes",
                        f"{risks['diabetes']['risk_percentage']:.1f}%",
                        delta=risks['diabetes']['risk_level']
                    )
                    
                    col3.metric(
                        "Cancer",
                        f"{risks['cancer']['risk_percentage']:.1f}%",
                        delta=risks['cancer']['risk_level']
                    )
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['CVD', 'Diabetes', 'Cancer'],
                            y=[
                                risks['cvd']['risk_percentage'],
                                risks['diabetes']['risk_percentage'],
                                risks['cancer']['risk_percentage']
                            ],
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                        )
                    ])
                    
                    fig.update_layout(
                        title=f"{time_horizon}-Year Disease Risk Scores",
                        yaxis_title="Risk (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"API Error: {response.status_code}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure the API server is running")


def intervention_simulation_tab():
    st.header("Intervention Simulation & Ranking")
    
    st.write("Simulate the impact of different health interventions on disease risk.")
    
    if st.button("🎯 Rank All Interventions", type="primary"):
        patient_data = {
            "patient_id": "P00000003",
            "age": 55,
            "gender": "male",
            "bmi": 32,
            "systolic_bp": 145,
            "total_cholesterol_mmol_l": 6.5,
            "hba1c_percent": 6.0,
            "smoking_status": "current",
            "alcohol_units_per_week": 18
        }
        
        with st.spinner("Simulating interventions..."):
            try:
                response = requests.post(f"{API_URL}/rank_interventions", json=patient_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Intervention Ranking Complete")
                    
                    ranked = result['ranked_interventions']
                    
                    df = pd.DataFrame(ranked)
                    
                    st.subheader("Top Recommended Interventions")
                    
                    for i, intervention in enumerate(ranked[:5], 1):
                        with st.expander(f"{i}. {intervention['intervention'].replace('_', ' ').title()}", expanded=(i==1)):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric("Benefit Score", f"{intervention['benefit_score']:.1f}")
                            col2.metric("Life Expectancy Gain", f"{intervention['life_expectancy_gain']:.1f} years")
                            col3.metric("Risk Reduction", f"{intervention['total_risk_reduction']:.1%}")
                            col4.metric("Adherence Rate", f"{intervention['adherence_rate']:.0%}")
                    
                    fig = px.bar(
                        df.head(8),
                        x='intervention',
                        y='benefit_score',
                        color='life_expectancy_gain',
                        title="Intervention Benefit Scores",
                        labels={'intervention': 'Intervention', 'benefit_score': 'Benefit Score'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"API Error: {response.status_code}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure the API server is running")


def population_analytics_tab():
    st.header("Population Health Analytics")
    
    st.write("Generate and analyze synthetic patient populations.")
    
    population_size = st.slider("Population Size", 100, 10000, 1000)
    
    if st.button("📊 Generate Population", type="primary"):
        with st.spinner(f"Generating {population_size} synthetic patients..."):
            generator = PatientPopulationGenerator()
            data = generator.generate_complete_population(n=population_size, output_dir="data/synthetic")
            
            df = data['complete']
            
            st.success(f"✅ Generated {len(df)} patients")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Average Age", f"{df['age'].mean():.1f}")
            col2.metric("Average BMI", f"{df['bmi'].mean():.1f}")
            col3.metric("Diabetes Prevalence", f"{df['diabetes'].mean():.1%}")
            col4.metric("CVD Prevalence", f"{df['heart_disease'].mean():.1%}")
            
            fig1 = px.histogram(df, x='age', nbins=30, title="Age Distribution")
            st.plotly_chart(fig1, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                fig2 = px.histogram(df, x='bmi', nbins=30, title="BMI Distribution")
                st.plotly_chart(fig2, use_container_width=True)
            
            with col_b:
                fig3 = px.scatter(
                    df.sample(min(500, len(df))),
                    x='age',
                    y='bmi',
                    color='diabetes',
                    title="Age vs BMI (Diabetes Status)"
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            st.subheader("Sample Patient Data")
            st.dataframe(df.head(20))


if __name__ == "__main__":
    main()
