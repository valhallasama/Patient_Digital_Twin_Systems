# Health Digital Twin Prediction Platform (HDTP)

A comprehensive patient digital twin platform for personalized health prediction, disease progression simulation, and intervention recommendation using multi-agent AI systems.

## 🎯 Overview

The Health Digital Twin Prediction Platform integrates:
- **Automated data acquisition** from public health repositories
- **Synthetic patient data generation** for training and simulation
- **Multi-agent medical reasoning** (Cardiology, Metabolic, Lifestyle agents)
- **Disease progression modeling** (Diabetes, CVD, Cancer)
- **Intervention simulation** and ranking
- **Risk prediction** with 10-year time horizons
- **Medical knowledge graphs** for causal reasoning
- **Interactive dashboard** for visualization

## 🏗️ Architecture

```
Data Acquisition → Data Processing → Patient Data Lake
                                            ↓
                                    Knowledge Graph
                                            ↓
                              Multi-Agent Reasoning
                                            ↓
                          Disease Progression Simulation
                                            ↓
                          Intervention Recommendation
                                            ↓
                              Prediction & Dashboard
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- PostgreSQL (optional, for database features)
- Neo4j (optional, for knowledge graph features)

### Setup

```bash
# Clone or navigate to the project directory
cd Patient_Digital_Twin_Systems

# Install dependencies
pip install -r requirements.txt

# (Optional) Set up PostgreSQL database
# Update config/system_config.yaml with your database credentials

# (Optional) Set up Neo4j
# Update config/system_config.yaml with your Neo4j credentials
```

## 🚀 Quick Start

### 1. Generate Synthetic Patient Data

```bash
cd synthetic_data_generator
python patient_population_generator.py
```

This generates 10,000 synthetic patients with complete health profiles.

### 2. Run Multi-Agent Evaluation

```bash
cd agents
python cardiology_agent.py
python metabolic_agent.py
python lifestyle_agent.py
```

### 3. Start API Server

```bash
python api/api_server.py
```

API will be available at `http://localhost:8000`

### 4. Launch Dashboard

```bash
streamlit run dashboard/health_dashboard.py
```

Dashboard will open at `http://localhost:8501`

## 📊 Core Modules

### Data Engine
- **dataset_discovery.py**: Automatically searches Figshare, Zenodo, Data.gov
- **dataset_downloader.py**: Downloads and validates datasets
- **dataset_validator.py**: Quality scoring and validation
- **dataset_scheduler.py**: Automated periodic data acquisition

### Synthetic Data Generator
- **patient_population_generator.py**: Demographics, vitals, labs, lifestyle
- **disease_progression_generator.py**: Simulates disease trajectories over time
- **lifestyle_generator.py**: Daily activity, diet, stress patterns
- **environment_generator.py**: Air quality, climate, socioeconomic factors

### Multi-Agent System
- **base_agent.py**: Agent framework and communication board
- **cardiology_agent.py**: Cardiovascular risk assessment
- **metabolic_agent.py**: Diabetes and metabolic syndrome evaluation
- **lifestyle_agent.py**: Lifestyle factor analysis

### Simulation Engine
- **disease_progression_model.py**: Multi-disease trajectory simulation
- **intervention_simulator.py**: Intervention impact modeling

### Prediction Engine
- **risk_predictor.py**: 10-year disease risk prediction

### Knowledge Graph
- **graph_builder.py**: Medical knowledge graph with Neo4j

## 🔧 API Endpoints

### POST /evaluate
Evaluate patient with multi-agent system

```json
{
  "patient_id": "P00000001",
  "age": 55,
  "gender": "male",
  "bmi": 32,
  "systolic_bp": 145,
  ...
}
```

### POST /predict_risk
Predict disease risks

```json
{
  "patient_id": "P00000001",
  "age": 55,
  ...
}
```

### POST /rank_interventions
Rank interventions by benefit

```json
{
  "patient_id": "P00000001",
  ...
}
```

## 📈 Usage Examples

### Generate Patient Population

```python
from synthetic_data_generator.patient_population_generator import PatientPopulationGenerator

generator = PatientPopulationGenerator()
data = generator.generate_complete_population(n=10000)
```

### Multi-Agent Evaluation

```python
from agents.base_agent import MultiAgentSystem
from agents.cardiology_agent import CardiologyAgent
from agents.metabolic_agent import MetabolicAgent

system = MultiAgentSystem()
system.register_agent(CardiologyAgent())
system.register_agent(MetabolicAgent())

evaluation = system.evaluate_patient(patient_data)
```

### Simulate Interventions

```python
from simulation_engine.intervention_simulator import InterventionSimulator

simulator = InterventionSimulator()
result = simulator.simulate_intervention(patient, 'smoking_cessation')
```

## 🗄️ Data Sources

The platform automatically discovers and downloads datasets from:
- **Figshare**: Research datasets
- **Zenodo**: Scientific data repository
- **Data.gov**: US government health data
- **CDC Wonder**: Mortality and disease data
- **NHANES**: Nutrition and health examination

## 🧪 Testing

Run individual modules:

```bash
# Test patient generation
python synthetic_data_generator/patient_population_generator.py

# Test disease progression
python synthetic_data_generator/disease_progression_generator.py

# Test agents
python agents/cardiology_agent.py

# Test intervention simulation
python simulation_engine/intervention_simulator.py
```

## 📝 Configuration

Edit `config/system_config.yaml`:

```yaml
data_engine:
  auto_discovery: true
  discovery_interval_hours: 24

synthetic_data:
  population_size: 1000000
  disease_models:
    - cardiovascular
    - diabetes
    - cancer
```

## 🔐 Security & Privacy

- All synthetic data is HIPAA-compliant (no real patient data)
- Database connections use encrypted credentials
- API supports CORS and rate limiting
- Audit logging for all predictions

## 🛠️ Technology Stack

- **Python 3.8+**
- **FastAPI**: REST API
- **Streamlit**: Interactive dashboard
- **PostgreSQL**: Patient data storage
- **Neo4j**: Medical knowledge graph
- **Pandas/NumPy**: Data processing
- **Plotly**: Visualization
- **scikit-learn**: Machine learning

## 📚 Project Structure

```
Patient_Digital_Twin_Systems/
├── config/                  # Configuration files
├── data_engine/            # Automated data acquisition
├── data_cleaning/          # Data preprocessing
├── synthetic_data_generator/  # Patient data generation
├── knowledge_graph/        # Medical knowledge base
├── agents/                 # Multi-agent reasoning
├── simulation_engine/      # Disease progression models
├── prediction_engine/      # Risk prediction
├── database/              # Database connectors
├── api/                   # REST API
├── dashboard/             # Web dashboard
├── data/                  # Data storage
└── logs/                  # System logs
```

## 🎓 Research Applications

- **Precision Medicine**: Personalized treatment planning
- **Population Health**: Disease burden forecasting
- **Clinical Trials**: Patient stratification
- **Health Economics**: Intervention cost-effectiveness
- **Epidemiology**: Disease progression modeling

## 🤝 Contributing

This is a research platform. Contributions welcome for:
- New disease models
- Additional medical agents
- Improved risk prediction algorithms
- Real-world data integrations

## 📄 License

Research and educational use.

## 🔮 Future Enhancements

- [ ] Real-time wearable data integration
- [ ] Genetic risk factor incorporation
- [ ] Deep learning disease progression models
- [ ] Multi-modal medical imaging analysis
- [ ] Federated learning across institutions
- [ ] Mobile app for patient engagement

## 📞 Support

For issues or questions, refer to the documentation in `docs/` directory.

---

Quick Commands
Start the interface:

bash
cd /home/tc115/Yue/Patient_Digital_Twin_Systems
./start_web_interface.sh
Or manually:

bash
streamlit run web_app.py --server.port 8501
Stop the interface:

bash
pkill -f streamlit
View logs:

bash
tail -f streamlit.log


**Built with ❤️ for advancing personalized healthcare through AI**
