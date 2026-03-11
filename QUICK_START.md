# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies

```bash
cd /home/tc115/Yue/Patient_Digital_Twin_Systems
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python run_demo.py
```

This will:
- Generate 100 synthetic patients
- Run multi-agent evaluation
- Predict disease risks
- Simulate interventions
- Model disease progression

### 3. Start the API Server

```bash
python api/api_server.py
```

API available at: http://localhost:8000

### 4. Launch the Dashboard

```bash
streamlit run dashboard/health_dashboard.py
```

Dashboard available at: http://localhost:8501

## 📊 Test Individual Components

### Generate Synthetic Patients

```bash
cd synthetic_data_generator
python patient_population_generator.py
```

### Run Agent Evaluation

```bash
cd agents
python cardiology_agent.py
python metabolic_agent.py
python lifestyle_agent.py
```

### Simulate Disease Progression

```bash
cd synthetic_data_generator
python disease_progression_generator.py
```

### Test Intervention Ranking

```bash
cd simulation_engine
python intervention_simulator.py
```

## 🔧 API Usage Examples

### Evaluate a Patient

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P00000001",
    "age": 55,
    "gender": "male",
    "bmi": 32,
    "systolic_bp": 145,
    "hba1c_percent": 6.2,
    "smoking_status": "current"
  }'
```

### Predict Risk

```bash
curl -X POST "http://localhost:8000/predict_risk?time_horizon_years=10" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P00000001",
    "age": 55,
    "gender": "male",
    "bmi": 32
  }'
```

### Rank Interventions

```bash
curl -X POST "http://localhost:8000/rank_interventions" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P00000001",
    "age": 55,
    "bmi": 32,
    "smoking_status": "current"
  }'
```

## 📁 Project Structure

```
Patient_Digital_Twin_Systems/
├── run_demo.py              # Complete demo workflow
├── requirements.txt         # Python dependencies
├── config/                  # Configuration files
├── data_engine/            # Automated data acquisition
├── synthetic_data_generator/  # Patient data generation
├── agents/                 # Multi-agent system
├── simulation_engine/      # Disease modeling
├── prediction_engine/      # Risk prediction
├── api/                    # REST API
└── dashboard/              # Web interface
```

## 🎯 Key Features

1. **Automated Data Generation**: Creates realistic synthetic patient populations
2. **Multi-Agent Reasoning**: Cardiology, Metabolic, and Lifestyle agents collaborate
3. **Disease Progression**: Simulates 10-year health trajectories
4. **Intervention Ranking**: Identifies most beneficial health interventions
5. **Interactive Dashboard**: Visualize patient health and predictions

## 💡 Tips

- Start with the demo script to understand the workflow
- Use the dashboard for interactive exploration
- Check logs/ directory for detailed execution logs
- Synthetic data is saved in data/synthetic/

## 🐛 Troubleshooting

**API won't start?**
- Check if port 8000 is available
- Verify all dependencies are installed

**Dashboard error?**
- Ensure API server is running first
- Check if port 8501 is available

**Import errors?**
- Make sure you're in the project root directory
- Verify Python path includes the project

## 📚 Next Steps

1. Explore the dashboard tabs
2. Generate larger patient populations
3. Customize disease models
4. Add new medical agents
5. Integrate real data sources

Enjoy exploring the Health Digital Twin Platform! 🏥
