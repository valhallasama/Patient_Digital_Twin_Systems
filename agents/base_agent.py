from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
        self.memory = []
        self.confidence_threshold = 0.5
        
    @abstractmethod
    def evaluate_patient(self, patient_data: pd.Series) -> Dict:
        pass
    
    def log_observation(self, observation: str):
        self.memory.append({
            'agent': self.name,
            'observation': observation
        })
        logger.info(f"[{self.name}] {observation}")
    
    def get_risk_level(self, risk_score: float) -> str:
        if risk_score < 0.2:
            return "low"
        elif risk_score < 0.5:
            return "moderate"
        elif risk_score < 0.75:
            return "high"
        else:
            return "critical"
    
    def calculate_confidence(self, available_data: Dict) -> float:
        required_fields = self.get_required_fields()
        available_count = sum([1 for field in required_fields if field in available_data])
        return available_count / len(required_fields) if required_fields else 1.0
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        pass
    
    def generate_recommendations(self, patient_data: pd.Series, 
                                evaluation: Dict) -> List[str]:
        return []


class AgentCommunicationBoard:
    def __init__(self):
        self.messages = []
        self.consensus = {}
        
    def post_message(self, agent_name: str, message: Dict):
        self.messages.append({
            'agent': agent_name,
            'content': message
        })
        logger.info(f"[Board] Message from {agent_name}: {message.get('summary', '')}")
    
    def get_messages_for_specialty(self, specialty: str) -> List[Dict]:
        return [msg for msg in self.messages if msg.get('specialty') == specialty]
    
    def calculate_consensus(self, topic: str) -> Dict:
        relevant_messages = [msg for msg in self.messages 
                           if msg['content'].get('topic') == topic]
        
        if not relevant_messages:
            return {'consensus': None, 'confidence': 0.0}
        
        scores = [msg['content'].get('risk_score', 0) for msg in relevant_messages]
        avg_score = sum(scores) / len(scores)
        
        agreement = 1.0 - (max(scores) - min(scores)) if scores else 0.0
        
        return {
            'consensus': avg_score,
            'confidence': agreement,
            'num_agents': len(relevant_messages)
        }
    
    def get_all_recommendations(self) -> List[str]:
        all_recs = []
        for msg in self.messages:
            recs = msg['content'].get('recommendations', [])
            all_recs.extend(recs)
        return list(set(all_recs))


class MultiAgentSystem:
    def __init__(self):
        self.agents = []
        self.board = AgentCommunicationBoard()
        
    def register_agent(self, agent: BaseAgent):
        self.agents.append(agent)
        logger.info(f"Registered agent: {agent.name} ({agent.specialty})")
    
    def evaluate_patient(self, patient_data: pd.Series) -> Dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"Multi-Agent Evaluation for Patient {patient_data.get('patient_id', 'Unknown')}")
        logger.info(f"{'='*60}\n")
        
        evaluations = {}
        
        for agent in self.agents:
            try:
                evaluation = agent.evaluate_patient(patient_data)
                evaluations[agent.name] = evaluation
                
                self.board.post_message(agent.name, {
                    'specialty': agent.specialty,
                    'topic': 'overall_health',
                    'risk_score': evaluation.get('risk_score', 0),
                    'summary': evaluation.get('summary', ''),
                    'recommendations': evaluation.get('recommendations', [])
                })
                
            except Exception as e:
                logger.error(f"Error in {agent.name}: {e}")
                evaluations[agent.name] = {'error': str(e)}
        
        consensus = self.board.calculate_consensus('overall_health')
        
        all_recommendations = self.board.get_all_recommendations()
        
        return {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'individual_evaluations': evaluations,
            'consensus': consensus,
            'recommendations': all_recommendations
        }


if __name__ == "__main__":
    board = AgentCommunicationBoard()
    
    board.post_message("Cardiology Agent", {
        'specialty': 'cardiology',
        'topic': 'overall_health',
        'risk_score': 0.6,
        'summary': 'Elevated cardiovascular risk',
        'recommendations': ['Increase exercise', 'Monitor blood pressure']
    })
    
    board.post_message("Metabolic Agent", {
        'specialty': 'metabolic',
        'topic': 'overall_health',
        'risk_score': 0.55,
        'summary': 'Prediabetic state',
        'recommendations': ['Reduce sugar intake', 'Weight loss program']
    })
    
    consensus = board.calculate_consensus('overall_health')
    logger.info(f"\nConsensus: {consensus}")
    
    recommendations = board.get_all_recommendations()
    logger.info(f"\nAll recommendations: {recommendations}")
