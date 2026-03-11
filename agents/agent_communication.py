"""
Agent Communication System - Swarm Reasoning
Enables dynamic collaboration between specialist agents
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MessageType(Enum):
    """Types of inter-agent messages"""
    OBSERVATION = "observation"  # Agent shares a finding
    QUERY = "query"  # Agent asks for input
    RECOMMENDATION = "recommendation"  # Agent suggests action
    ALERT = "alert"  # Agent raises concern
    CONSENSUS_REQUEST = "consensus_request"  # Request for agreement
    CONSENSUS_RESPONSE = "consensus_response"  # Response to consensus
    KNOWLEDGE_SHARE = "knowledge_share"  # Share domain knowledge


@dataclass
class Message:
    """Inter-agent message"""
    id: str
    timestamp: datetime
    from_agent: str
    to_agents: List[str]  # Can broadcast to multiple
    message_type: MessageType
    priority: MessagePriority
    subject: str
    content: Dict[str, Any]
    requires_response: bool = False
    parent_message_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'from_agent': self.from_agent,
            'to_agents': self.to_agents,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'subject': self.subject,
            'content': self.content,
            'requires_response': self.requires_response,
            'parent_message_id': self.parent_message_id
        }


@dataclass
class SharedMemory:
    """Shared memory accessible to all agents"""
    patient_data: Dict[str, Any] = field(default_factory=dict)
    observations: List[Dict] = field(default_factory=list)
    hypotheses: List[Dict] = field(default_factory=list)
    consensus_items: Dict[str, Any] = field(default_factory=dict)
    risk_assessments: Dict[str, float] = field(default_factory=dict)
    intervention_proposals: List[Dict] = field(default_factory=list)
    
    def add_observation(self, agent: str, observation: str, evidence: Dict):
        """Add observation to shared memory"""
        self.observations.append({
            'timestamp': datetime.now(),
            'agent': agent,
            'observation': observation,
            'evidence': evidence
        })
    
    def add_hypothesis(self, agent: str, hypothesis: str, confidence: float):
        """Add hypothesis to shared memory"""
        self.hypotheses.append({
            'timestamp': datetime.now(),
            'agent': agent,
            'hypothesis': hypothesis,
            'confidence': confidence
        })
    
    def update_consensus(self, topic: str, value: Any, supporting_agents: List[str]):
        """Update consensus on a topic"""
        self.consensus_items[topic] = {
            'value': value,
            'supporting_agents': supporting_agents,
            'timestamp': datetime.now()
        }
    
    def get_observations_by_agent(self, agent: str) -> List[Dict]:
        """Get all observations from specific agent"""
        return [obs for obs in self.observations if obs['agent'] == agent]
    
    def get_consensus(self, topic: str) -> Optional[Dict]:
        """Get consensus on a topic"""
        return self.consensus_items.get(topic)


class MessageBus:
    """
    Central message bus for agent communication
    Enables publish-subscribe pattern
    """
    
    def __init__(self):
        self.messages: List[Message] = []
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_counter = 0
    
    def publish(self, message: Message):
        """Publish message to bus"""
        self.messages.append(message)
        logger.debug(f"Message published: {message.from_agent} -> {message.to_agents}: {message.subject}")
        
        # Notify subscribers
        for agent in message.to_agents:
            if agent in self.subscribers:
                for callback in self.subscribers[agent]:
                    callback(message)
    
    def subscribe(self, agent_name: str, callback: Callable):
        """Subscribe agent to receive messages"""
        self.subscribers[agent_name].append(callback)
        logger.debug(f"Agent {agent_name} subscribed to message bus")
    
    def get_messages_for_agent(self, agent_name: str, 
                               unread_only: bool = True) -> List[Message]:
        """Get messages for specific agent"""
        messages = [m for m in self.messages if agent_name in m.to_agents]
        return messages
    
    def get_conversation_thread(self, message_id: str) -> List[Message]:
        """Get full conversation thread"""
        thread = []
        
        # Find root message
        root = next((m for m in self.messages if m.id == message_id), None)
        if not root:
            return thread
        
        thread.append(root)
        
        # Find all replies
        current_id = message_id
        while True:
            replies = [m for m in self.messages if m.parent_message_id == current_id]
            if not replies:
                break
            thread.extend(replies)
            current_id = replies[-1].id
        
        return thread
    
    def create_message_id(self) -> str:
        """Generate unique message ID"""
        self.message_counter += 1
        return f"msg_{self.message_counter}_{datetime.now().timestamp()}"


class SwarmReasoningCoordinator:
    """
    Coordinates swarm reasoning among multiple agents
    Implements collaborative decision-making
    """
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.shared_memory = SharedMemory()
        self.agents: Dict[str, Any] = {}
        self.reasoning_sessions: List[Dict] = []
    
    def register_agent(self, agent_name: str, agent_instance: Any):
        """Register agent with coordinator"""
        self.agents[agent_name] = agent_instance
        
        # Subscribe agent to message bus
        def message_handler(message: Message):
            logger.info(f"{agent_name} received: {message.subject}")
        
        self.message_bus.subscribe(agent_name, message_handler)
        logger.info(f"✓ Registered agent: {agent_name}")
    
    def broadcast_observation(self, from_agent: str, observation: str, 
                            evidence: Dict, priority: MessagePriority = MessagePriority.MEDIUM):
        """Agent broadcasts observation to all other agents"""
        
        # Add to shared memory
        self.shared_memory.add_observation(from_agent, observation, evidence)
        
        # Create message
        message = Message(
            id=self.message_bus.create_message_id(),
            timestamp=datetime.now(),
            from_agent=from_agent,
            to_agents=[name for name in self.agents.keys() if name != from_agent],
            message_type=MessageType.OBSERVATION,
            priority=priority,
            subject=f"Observation: {observation}",
            content={'observation': observation, 'evidence': evidence},
            requires_response=False
        )
        
        self.message_bus.publish(message)
        return message
    
    def request_consensus(self, from_agent: str, topic: str, 
                         proposal: Any) -> Dict[str, Any]:
        """Request consensus from all agents on a topic"""
        
        message = Message(
            id=self.message_bus.create_message_id(),
            timestamp=datetime.now(),
            from_agent=from_agent,
            to_agents=[name for name in self.agents.keys() if name != from_agent],
            message_type=MessageType.CONSENSUS_REQUEST,
            priority=MessagePriority.HIGH,
            subject=f"Consensus request: {topic}",
            content={'topic': topic, 'proposal': proposal},
            requires_response=True
        )
        
        self.message_bus.publish(message)
        
        # Collect responses (in real implementation, would be async)
        responses = {}
        for agent_name in self.agents.keys():
            if agent_name != from_agent:
                # Agent evaluates proposal
                response = self._get_agent_consensus_response(
                    agent_name, topic, proposal
                )
                responses[agent_name] = response
        
        # Calculate consensus
        agreements = sum(1 for r in responses.values() if r['agrees'])
        total = len(responses)
        consensus_reached = agreements / total >= 0.66  # 2/3 majority
        
        if consensus_reached:
            supporting_agents = [a for a, r in responses.items() if r['agrees']]
            self.shared_memory.update_consensus(topic, proposal, supporting_agents)
        
        return {
            'consensus_reached': consensus_reached,
            'agreement_rate': agreements / total,
            'responses': responses,
            'supporting_agents': [a for a, r in responses.items() if r['agrees']]
        }
    
    def _get_agent_consensus_response(self, agent_name: str, 
                                     topic: str, proposal: Any) -> Dict:
        """Get agent's response to consensus request"""
        # Simplified - in real implementation, would call agent's evaluate method
        agent = self.agents.get(agent_name)
        
        # Default response structure
        return {
            'agrees': True,  # Placeholder
            'confidence': 0.8,
            'reasoning': f"{agent_name} evaluation of {topic}",
            'alternative_proposal': None
        }
    
    def collaborative_diagnosis(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Collaborative diagnosis using swarm reasoning
        Agents share observations and build consensus
        """
        
        logger.info("\n" + "="*80)
        logger.info("SWARM REASONING SESSION - Collaborative Diagnosis")
        logger.info("="*80)
        
        session = {
            'session_id': f"session_{datetime.now().timestamp()}",
            'start_time': datetime.now(),
            'patient_data': patient_data,
            'observations': [],
            'hypotheses': [],
            'consensus': {},
            'final_assessment': {}
        }
        
        # Update shared memory
        self.shared_memory.patient_data = patient_data
        
        # Phase 1: Individual agent observations
        logger.info("\nPhase 1: Individual Agent Observations")
        logger.info("-" * 80)
        
        for agent_name, agent in self.agents.items():
            # Agent analyzes patient
            if hasattr(agent, 'analyze_patient'):
                analysis = agent.analyze_patient(patient_data)
                
                # Extract key findings
                key_finding = analysis.get('reasoning', 'No specific finding')[:100]
                risk_level = analysis.get('risk_assessment', {}).get('overall_cvd_risk', 'unknown')
                
                # Broadcast observation
                self.broadcast_observation(
                    from_agent=agent_name,
                    observation=f"Risk level: {risk_level}",
                    evidence={'analysis': analysis},
                    priority=MessagePriority.MEDIUM
                )
                
                session['observations'].append({
                    'agent': agent_name,
                    'finding': key_finding,
                    'risk': risk_level
                })
        
        # Phase 2: Hypothesis generation
        logger.info("\nPhase 2: Hypothesis Generation")
        logger.info("-" * 80)
        
        # Agents propose hypotheses based on shared observations
        for agent_name in self.agents.keys():
            hypothesis = f"{agent_name} hypothesis based on observations"
            self.shared_memory.add_hypothesis(agent_name, hypothesis, confidence=0.75)
            session['hypotheses'].append({
                'agent': agent_name,
                'hypothesis': hypothesis
            })
        
        # Phase 3: Consensus building
        logger.info("\nPhase 3: Consensus Building")
        logger.info("-" * 80)
        
        # Request consensus on overall risk level
        risk_levels = [obs.get('risk', 'moderate') for obs in session['observations']]
        most_common_risk = max(set(risk_levels), key=risk_levels.count)
        
        consensus_result = self.request_consensus(
            from_agent='Coordinator',
            topic='overall_risk_assessment',
            proposal=most_common_risk
        )
        
        session['consensus'] = consensus_result
        
        # Phase 4: Final collaborative assessment
        logger.info("\nPhase 4: Final Collaborative Assessment")
        logger.info("-" * 80)
        
        session['final_assessment'] = {
            'consensus_risk': most_common_risk if consensus_result['consensus_reached'] else 'uncertain',
            'agreement_level': consensus_result['agreement_rate'],
            'supporting_agents': consensus_result['supporting_agents'],
            'key_observations': session['observations'],
            'reasoning': 'Multi-agent collaborative analysis with consensus building'
        }
        
        session['end_time'] = datetime.now()
        self.reasoning_sessions.append(session)
        
        logger.info(f"\n✓ Consensus reached: {consensus_result['consensus_reached']}")
        logger.info(f"✓ Agreement rate: {consensus_result['agreement_rate']:.1%}")
        
        return session
    
    def get_shared_knowledge(self) -> Dict:
        """Get current state of shared knowledge"""
        return {
            'observations': self.shared_memory.observations,
            'hypotheses': self.shared_memory.hypotheses,
            'consensus_items': self.shared_memory.consensus_items,
            'risk_assessments': self.shared_memory.risk_assessments
        }
    
    def export_reasoning_session(self, session_id: str) -> Dict:
        """Export reasoning session for analysis"""
        session = next((s for s in self.reasoning_sessions 
                       if s['session_id'] == session_id), None)
        return session


# Example usage
if __name__ == "__main__":
    from agents.cardiology_agent import CardiologyAgent
    from agents.metabolic_agent import MetabolicAgent
    from agents.lifestyle_agent import LifestyleAgent
    
    # Create swarm coordinator
    coordinator = SwarmReasoningCoordinator()
    
    # Register agents
    coordinator.register_agent('Cardiology', CardiologyAgent())
    coordinator.register_agent('Metabolic', MetabolicAgent())
    coordinator.register_agent('Lifestyle', LifestyleAgent())
    
    # Sample patient data
    patient_data = {
        'demographics': {'age': 55, 'gender': 'male'},
        'physical': {'bmi': 32, 'waist_circumference_cm': 105},
        'vitals': {'systolic_bp': 145, 'diastolic_bp': 92, 'heart_rate': 78},
        'labs': {
            'glucose_mmol_l': 6.2,
            'hba1c_percent': 6.0,
            'ldl_cholesterol_mmol_l': 4.2,
            'hdl_cholesterol_mmol_l': 1.0,
            'triglycerides_mmol_l': 2.5
        },
        'lifestyle': {
            'smoking_status': 'current',
            'exercise_hours_per_week': 1,
            'sleep_hours_per_night': 5.5,
            'alcohol_units_per_week': 20
        },
        'family_history': {
            'father_cvd': True,
            'father_cvd_age': 58
        }
    }
    
    # Run collaborative diagnosis
    print("\n" + "="*80)
    print("SWARM REASONING DEMONSTRATION")
    print("="*80)
    
    session = coordinator.collaborative_diagnosis(patient_data)
    
    # Display results
    print("\n" + "="*80)
    print("COLLABORATIVE DIAGNOSIS RESULTS")
    print("="*80)
    
    print(f"\nConsensus Risk: {session['final_assessment']['consensus_risk']}")
    print(f"Agreement Level: {session['final_assessment']['agreement_level']:.1%}")
    print(f"Supporting Agents: {', '.join(session['final_assessment']['supporting_agents'])}")
    
    print("\nKey Observations:")
    for obs in session['observations']:
        print(f"  • {obs['agent']}: {obs['finding'][:80]}...")
    
    # Show shared knowledge
    shared_knowledge = coordinator.get_shared_knowledge()
    print(f"\nShared Memory:")
    print(f"  Observations: {len(shared_knowledge['observations'])}")
    print(f"  Hypotheses: {len(shared_knowledge['hypotheses'])}")
    print(f"  Consensus Items: {len(shared_knowledge['consensus_items'])}")
