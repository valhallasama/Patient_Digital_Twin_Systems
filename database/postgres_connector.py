import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import Dict, List, Optional
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostgresConnector:
    def __init__(self, config_path: str = "config/system_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        db_config = config['database']['postgres']
        
        try:
            self.conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password']
            )
            self.conn.autocommit = False
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.warning(f"Could not connect to PostgreSQL: {e}")
            self.conn = None
    
    def create_tables(self):
        if not self.conn:
            return
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(50) PRIMARY KEY,
                age INTEGER,
                gender VARCHAR(10),
                ethnicity VARCHAR(50),
                height_cm FLOAT,
                weight_kg FLOAT,
                bmi FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vital_signs (
                id SERIAL PRIMARY KEY,
                patient_id VARCHAR(50) REFERENCES patients(patient_id),
                measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                systolic_bp FLOAT,
                diastolic_bp FLOAT,
                heart_rate FLOAT,
                respiratory_rate FLOAT,
                temperature_c FLOAT,
                oxygen_saturation FLOAT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lab_results (
                id SERIAL PRIMARY KEY,
                patient_id VARCHAR(50) REFERENCES patients(patient_id),
                test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                glucose_mmol_l FLOAT,
                hba1c_percent FLOAT,
                total_cholesterol_mmol_l FLOAT,
                ldl_cholesterol_mmol_l FLOAT,
                hdl_cholesterol_mmol_l FLOAT,
                triglycerides_mmol_l FLOAT,
                creatinine_umol_l FLOAT,
                alt_u_l FLOAT,
                ast_u_l FLOAT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medical_history (
                id SERIAL PRIMARY KEY,
                patient_id VARCHAR(50) REFERENCES patients(patient_id),
                condition VARCHAR(100),
                diagnosed_date DATE,
                status VARCHAR(20),
                notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_predictions (
                id SERIAL PRIMARY KEY,
                patient_id VARCHAR(50) REFERENCES patients(patient_id),
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                disease VARCHAR(50),
                risk_score FLOAT,
                risk_level VARCHAR(20),
                time_horizon_years INTEGER
            )
        """)
        
        self.conn.commit()
        cursor.close()
        logger.info("Database tables created successfully")
    
    def insert_patient(self, patient_data: Dict):
        if not self.conn:
            return
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO patients (patient_id, age, gender, ethnicity, height_cm, weight_kg, bmi)
            VALUES (%(patient_id)s, %(age)s, %(gender)s, %(ethnicity)s, %(height_cm)s, %(weight_kg)s, %(bmi)s)
            ON CONFLICT (patient_id) DO UPDATE SET
                age = EXCLUDED.age,
                gender = EXCLUDED.gender,
                updated_at = CURRENT_TIMESTAMP
        """, patient_data)
        
        self.conn.commit()
        cursor.close()
        logger.info(f"Inserted/updated patient: {patient_data['patient_id']}")
    
    def insert_vital_signs(self, patient_id: str, vitals: Dict):
        if not self.conn:
            return
        
        cursor = self.conn.cursor()
        
        vitals['patient_id'] = patient_id
        
        cursor.execute("""
            INSERT INTO vital_signs 
            (patient_id, systolic_bp, diastolic_bp, heart_rate, respiratory_rate, temperature_c, oxygen_saturation)
            VALUES (%(patient_id)s, %(systolic_bp)s, %(diastolic_bp)s, %(heart_rate)s, 
                    %(respiratory_rate)s, %(temperature_c)s, %(oxygen_saturation)s)
        """, vitals)
        
        self.conn.commit()
        cursor.close()
    
    def insert_risk_prediction(self, patient_id: str, prediction: Dict):
        if not self.conn:
            return
        
        cursor = self.conn.cursor()
        
        prediction['patient_id'] = patient_id
        
        cursor.execute("""
            INSERT INTO risk_predictions (patient_id, disease, risk_score, risk_level, time_horizon_years)
            VALUES (%(patient_id)s, %(disease)s, %(risk_score)s, %(risk_level)s, %(time_horizon_years)s)
        """, prediction)
        
        self.conn.commit()
        cursor.close()
    
    def get_patient(self, patient_id: str) -> Optional[Dict]:
        if not self.conn:
            return None
        
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
        result = cursor.fetchone()
        
        cursor.close()
        return dict(result) if result else None
    
    def get_patient_vitals(self, patient_id: str, limit: int = 10) -> List[Dict]:
        if not self.conn:
            return []
        
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT * FROM vital_signs 
            WHERE patient_id = %s 
            ORDER BY measurement_date DESC 
            LIMIT %s
        """, (patient_id, limit))
        
        results = cursor.fetchall()
        cursor.close()
        
        return [dict(row) for row in results]
    
    def get_patient_risk_history(self, patient_id: str) -> pd.DataFrame:
        if not self.conn:
            return pd.DataFrame()
        
        query = """
            SELECT * FROM risk_predictions 
            WHERE patient_id = %s 
            ORDER BY prediction_date DESC
        """
        
        df = pd.read_sql_query(query, self.conn, params=(patient_id,))
        return df
    
    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    db = PostgresConnector()
    
    db.create_tables()
    
    test_patient = {
        'patient_id': 'P00000001',
        'age': 55,
        'gender': 'male',
        'ethnicity': 'caucasian',
        'height_cm': 175,
        'weight_kg': 90,
        'bmi': 29.4
    }
    
    db.insert_patient(test_patient)
    
    vitals = {
        'systolic_bp': 145,
        'diastolic_bp': 92,
        'heart_rate': 78,
        'respiratory_rate': 16,
        'temperature_c': 36.8,
        'oxygen_saturation': 98
    }
    
    db.insert_vital_signs('P00000001', vitals)
    
    prediction = {
        'disease': 'cvd',
        'risk_score': 0.45,
        'risk_level': 'moderate',
        'time_horizon_years': 10
    }
    
    db.insert_risk_prediction('P00000001', prediction)
    
    patient_data = db.get_patient('P00000001')
    logger.info(f"\nRetrieved patient: {patient_data}")
    
    db.close()
