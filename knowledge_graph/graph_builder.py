from neo4j import GraphDatabase
import logging
from typing import Dict, List, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalKnowledgeGraph:
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "changeme"):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}. Graph features will be limited.")
            self.driver = None
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def create_disease_node(self, disease_name: str, properties: Dict = None):
        if not self.driver:
            return
        
        with self.driver.session() as session:
            query = """
            MERGE (d:Disease {name: $name})
            SET d += $properties
            RETURN d
            """
            session.run(query, name=disease_name, properties=properties or {})
            logger.info(f"Created disease node: {disease_name}")
    
    def create_risk_factor_node(self, factor_name: str, properties: Dict = None):
        if not self.driver:
            return
        
        with self.driver.session() as session:
            query = """
            MERGE (r:RiskFactor {name: $name})
            SET r += $properties
            RETURN r
            """
            session.run(query, name=factor_name, properties=properties or {})
            logger.info(f"Created risk factor node: {factor_name}")
    
    def create_symptom_node(self, symptom_name: str, properties: Dict = None):
        if not self.driver:
            return
        
        with self.driver.session() as session:
            query = """
            MERGE (s:Symptom {name: $name})
            SET s += $properties
            RETURN s
            """
            session.run(query, name=symptom_name, properties=properties or {})
    
    def create_treatment_node(self, treatment_name: str, properties: Dict = None):
        if not self.driver:
            return
        
        with self.driver.session() as session:
            query = """
            MERGE (t:Treatment {name: $name})
            SET t += $properties
            RETURN t
            """
            session.run(query, name=treatment_name, properties=properties or {})
    
    def create_increases_risk_relationship(self, risk_factor: str, disease: str, 
                                          risk_multiplier: float = 1.5):
        if not self.driver:
            return
        
        with self.driver.session() as session:
            query = """
            MATCH (r:RiskFactor {name: $risk_factor})
            MATCH (d:Disease {name: $disease})
            MERGE (r)-[rel:INCREASES_RISK]->(d)
            SET rel.risk_multiplier = $risk_multiplier
            RETURN rel
            """
            session.run(query, risk_factor=risk_factor, disease=disease, 
                       risk_multiplier=risk_multiplier)
            logger.info(f"Created relationship: {risk_factor} -> INCREASES_RISK -> {disease}")
    
    def create_causes_relationship(self, disease: str, symptom: str, 
                                   probability: float = 0.5):
        if not self.driver:
            return
        
        with self.driver.session() as session:
            query = """
            MATCH (d:Disease {name: $disease})
            MATCH (s:Symptom {name: $symptom})
            MERGE (d)-[rel:CAUSES]->(s)
            SET rel.probability = $probability
            RETURN rel
            """
            session.run(query, disease=disease, symptom=symptom, probability=probability)
    
    def create_treats_relationship(self, treatment: str, disease: str, 
                                   effectiveness: float = 0.7):
        if not self.driver:
            return
        
        with self.driver.session() as session:
            query = """
            MATCH (t:Treatment {name: $treatment})
            MATCH (d:Disease {name: $disease})
            MERGE (t)-[rel:TREATS]->(d)
            SET rel.effectiveness = $effectiveness
            RETURN rel
            """
            session.run(query, treatment=treatment, disease=disease, 
                       effectiveness=effectiveness)
    
    def build_cardiovascular_knowledge(self):
        logger.info("Building cardiovascular disease knowledge graph...")
        
        self.create_disease_node("Cardiovascular Disease", {
            "category": "cardiovascular",
            "severity": "high"
        })
        self.create_disease_node("Hypertension", {"category": "cardiovascular"})
        self.create_disease_node("Heart Attack", {"category": "cardiovascular", "severity": "critical"})
        
        risk_factors = [
            ("Smoking", 2.5),
            ("High Cholesterol", 2.0),
            ("Obesity", 1.8),
            ("Diabetes", 2.0),
            ("Physical Inactivity", 1.5),
            ("High Blood Pressure", 2.5)
        ]
        
        for factor, multiplier in risk_factors:
            self.create_risk_factor_node(factor)
            self.create_increases_risk_relationship(factor, "Cardiovascular Disease", multiplier)
        
        treatments = [
            ("Statins", 0.8),
            ("ACE Inhibitors", 0.75),
            ("Beta Blockers", 0.7),
            ("Exercise Program", 0.6),
            ("Diet Modification", 0.5)
        ]
        
        for treatment, effectiveness in treatments:
            self.create_treatment_node(treatment)
            self.create_treats_relationship(treatment, "Cardiovascular Disease", effectiveness)
    
    def build_diabetes_knowledge(self):
        logger.info("Building diabetes knowledge graph...")
        
        self.create_disease_node("Type 2 Diabetes", {
            "category": "metabolic",
            "severity": "high"
        })
        
        risk_factors = [
            ("Obesity", 3.0),
            ("Physical Inactivity", 2.0),
            ("Poor Diet", 2.5),
            ("Family History Diabetes", 2.5),
            ("Age Over 45", 1.5)
        ]
        
        for factor, multiplier in risk_factors:
            self.create_risk_factor_node(factor)
            self.create_increases_risk_relationship(factor, "Type 2 Diabetes", multiplier)
        
        treatments = [
            ("Metformin", 0.85),
            ("Insulin", 0.9),
            ("Weight Loss", 0.7),
            ("Exercise Program", 0.65),
            ("Diet Modification", 0.6)
        ]
        
        for treatment, effectiveness in treatments:
            self.create_treatment_node(treatment)
            self.create_treats_relationship(treatment, "Type 2 Diabetes", effectiveness)
    
    def build_cancer_knowledge(self):
        logger.info("Building cancer knowledge graph...")
        
        cancers = ["Lung Cancer", "Breast Cancer", "Colorectal Cancer", "Prostate Cancer"]
        
        for cancer in cancers:
            self.create_disease_node(cancer, {"category": "cancer", "severity": "critical"})
        
        self.create_risk_factor_node("Smoking")
        self.create_increases_risk_relationship("Smoking", "Lung Cancer", 15.0)
        
        self.create_risk_factor_node("Alcohol Consumption")
        self.create_increases_risk_relationship("Alcohol Consumption", "Breast Cancer", 1.5)
        self.create_increases_risk_relationship("Alcohol Consumption", "Colorectal Cancer", 1.4)
    
    def query_risk_factors_for_disease(self, disease_name: str) -> List[Dict]:
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            query = """
            MATCH (r:RiskFactor)-[rel:INCREASES_RISK]->(d:Disease {name: $disease})
            RETURN r.name as risk_factor, rel.risk_multiplier as multiplier
            ORDER BY rel.risk_multiplier DESC
            """
            result = session.run(query, disease=disease_name)
            return [dict(record) for record in result]
    
    def query_treatments_for_disease(self, disease_name: str) -> List[Dict]:
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            query = """
            MATCH (t:Treatment)-[rel:TREATS]->(d:Disease {name: $disease})
            RETURN t.name as treatment, rel.effectiveness as effectiveness
            ORDER BY rel.effectiveness DESC
            """
            result = session.run(query, disease=disease_name)
            return [dict(record) for record in result]
    
    def initialize_medical_knowledge_base(self):
        logger.info("Initializing complete medical knowledge base...")
        
        self.build_cardiovascular_knowledge()
        self.build_diabetes_knowledge()
        self.build_cancer_knowledge()
        
        logger.info("Medical knowledge base initialized successfully")


if __name__ == "__main__":
    kg = MedicalKnowledgeGraph()
    
    kg.initialize_medical_knowledge_base()
    
    cvd_risks = kg.query_risk_factors_for_disease("Cardiovascular Disease")
    logger.info(f"\nRisk factors for CVD:\n{cvd_risks}")
    
    diabetes_treatments = kg.query_treatments_for_disease("Type 2 Diabetes")
    logger.info(f"\nTreatments for Type 2 Diabetes:\n{diabetes_treatments}")
    
    kg.close()
