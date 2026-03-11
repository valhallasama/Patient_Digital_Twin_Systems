"""
Blackboard System for Multi-Agent Collaboration
Shared cognitive workspace where agents exchange knowledge and refine outputs
Implements true swarm reasoning with feedback loops
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge on blackboard"""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    RECOMMENDATION = "recommendation"
    CONSTRAINT = "constraint"
    GOAL = "goal"
    SOLUTION = "solution"


class ConfidenceLevel(Enum):
    """Confidence levels for knowledge items"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class KnowledgeItem:
    """Single piece of knowledge on the blackboard"""
    id: str
    timestamp: datetime
    knowledge_type: KnowledgeType
    content: Dict[str, Any]
    source_agent: str
    confidence: float
    supporting_agents: List[str] = field(default_factory=list)
    contradicting_agents: List[str] = field(default_factory=list)
    refinement_history: List[Dict] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    def add_support(self, agent: str, evidence: Dict):
        """Agent supports this knowledge item"""
        if agent not in self.supporting_agents:
            self.supporting_agents.append(agent)
            self.refinement_history.append({
                'timestamp': datetime.now(),
                'agent': agent,
                'action': 'support',
                'evidence': evidence
            })
            # Increase confidence with more support
            self.confidence = min(0.99, self.confidence + 0.1)
    
    def add_contradiction(self, agent: str, reason: str):
        """Agent contradicts this knowledge item"""
        if agent not in self.contradicting_agents:
            self.contradicting_agents.append(agent)
            self.refinement_history.append({
                'timestamp': datetime.now(),
                'agent': agent,
                'action': 'contradict',
                'reason': reason
            })
            # Decrease confidence with contradictions
            self.confidence = max(0.1, self.confidence - 0.15)
    
    def refine(self, agent: str, new_content: Dict):
        """Agent refines this knowledge item"""
        self.refinement_history.append({
            'timestamp': datetime.now(),
            'agent': agent,
            'action': 'refine',
            'old_content': self.content.copy(),
            'new_content': new_content
        })
        self.content.update(new_content)
    
    def get_consensus_score(self) -> float:
        """Calculate consensus score based on support/contradiction"""
        total_agents = len(self.supporting_agents) + len(self.contradicting_agents)
        if total_agents == 0:
            return self.confidence
        
        support_ratio = len(self.supporting_agents) / total_agents
        return self.confidence * support_ratio


@dataclass
class AgentContribution:
    """Track agent's contributions to blackboard"""
    agent_name: str
    items_posted: int = 0
    items_supported: int = 0
    items_contradicted: int = 0
    items_refined: int = 0
    last_active: Optional[datetime] = None


class Blackboard:
    """
    Shared cognitive workspace for multi-agent collaboration
    Agents read from and write to the blackboard, refining each other's outputs
    """
    
    def __init__(self):
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.agent_contributions: Dict[str, AgentContribution] = {}
        self.item_counter = 0
        self.reasoning_cycles: List[Dict] = []
        
        logger.info("✓ Blackboard System initialized")
    
    def post_knowledge(self, agent_name: str, knowledge_type: KnowledgeType,
                      content: Dict, confidence: float = 0.7,
                      tags: Optional[Set[str]] = None) -> str:
        """
        Agent posts knowledge to blackboard
        Returns knowledge item ID
        """
        self.item_counter += 1
        item_id = f"KB_{self.item_counter}_{datetime.now().timestamp()}"
        
        item = KnowledgeItem(
            id=item_id,
            timestamp=datetime.now(),
            knowledge_type=knowledge_type,
            content=content,
            source_agent=agent_name,
            confidence=confidence,
            tags=tags or set()
        )
        
        self.knowledge_items[item_id] = item
        
        # Track contribution
        if agent_name not in self.agent_contributions:
            self.agent_contributions[agent_name] = AgentContribution(agent_name)
        
        self.agent_contributions[agent_name].items_posted += 1
        self.agent_contributions[agent_name].last_active = datetime.now()
        
        logger.debug(f"{agent_name} posted {knowledge_type.value}: {item_id}")
        
        return item_id
    
    def support_knowledge(self, agent_name: str, item_id: str, evidence: Dict):
        """Agent supports existing knowledge item"""
        if item_id not in self.knowledge_items:
            raise ValueError(f"Knowledge item {item_id} not found")
        
        item = self.knowledge_items[item_id]
        item.add_support(agent_name, evidence)
        
        # Track contribution
        if agent_name not in self.agent_contributions:
            self.agent_contributions[agent_name] = AgentContribution(agent_name)
        
        self.agent_contributions[agent_name].items_supported += 1
        self.agent_contributions[agent_name].last_active = datetime.now()
        
        logger.debug(f"{agent_name} supported {item_id}")
    
    def contradict_knowledge(self, agent_name: str, item_id: str, reason: str):
        """Agent contradicts existing knowledge item"""
        if item_id not in self.knowledge_items:
            raise ValueError(f"Knowledge item {item_id} not found")
        
        item = self.knowledge_items[item_id]
        item.add_contradiction(agent_name, reason)
        
        # Track contribution
        if agent_name not in self.agent_contributions:
            self.agent_contributions[agent_name] = AgentContribution(agent_name)
        
        self.agent_contributions[agent_name].items_contradicted += 1
        self.agent_contributions[agent_name].last_active = datetime.now()
        
        logger.debug(f"{agent_name} contradicted {item_id}: {reason}")
    
    def refine_knowledge(self, agent_name: str, item_id: str, new_content: Dict):
        """Agent refines existing knowledge item"""
        if item_id not in self.knowledge_items:
            raise ValueError(f"Knowledge item {item_id} not found")
        
        item = self.knowledge_items[item_id]
        item.refine(agent_name, new_content)
        
        # Track contribution
        if agent_name not in self.agent_contributions:
            self.agent_contributions[agent_name] = AgentContribution(agent_name)
        
        self.agent_contributions[agent_name].items_refined += 1
        self.agent_contributions[agent_name].last_active = datetime.now()
        
        logger.debug(f"{agent_name} refined {item_id}")
    
    def query_knowledge(self, knowledge_type: Optional[KnowledgeType] = None,
                       tags: Optional[Set[str]] = None,
                       min_confidence: float = 0.0,
                       min_consensus: float = 0.0) -> List[KnowledgeItem]:
        """
        Query knowledge items from blackboard
        Agents use this to see what others have posted
        """
        items = list(self.knowledge_items.values())
        
        # Filter by type
        if knowledge_type:
            items = [i for i in items if i.knowledge_type == knowledge_type]
        
        # Filter by tags
        if tags:
            items = [i for i in items if tags.intersection(i.tags)]
        
        # Filter by confidence
        items = [i for i in items if i.confidence >= min_confidence]
        
        # Filter by consensus
        items = [i for i in items if i.get_consensus_score() >= min_consensus]
        
        # Sort by consensus score
        items.sort(key=lambda i: i.get_consensus_score(), reverse=True)
        
        return items
    
    def get_high_consensus_items(self, threshold: float = 0.7) -> List[KnowledgeItem]:
        """Get items with high consensus (for decision making)"""
        return [item for item in self.knowledge_items.values()
                if item.get_consensus_score() >= threshold]
    
    def get_controversial_items(self, threshold: float = 0.4) -> List[KnowledgeItem]:
        """Get items with low consensus (need more discussion)"""
        return [item for item in self.knowledge_items.values()
                if item.get_consensus_score() < threshold and 
                len(item.contradicting_agents) > 0]
    
    def clear_old_items(self, hours: int = 24):
        """Clear knowledge items older than specified hours"""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        
        old_items = [item_id for item_id, item in self.knowledge_items.items()
                    if item.timestamp.timestamp() < cutoff]
        
        for item_id in old_items:
            del self.knowledge_items[item_id]
        
        logger.info(f"Cleared {len(old_items)} old knowledge items")
    
    def get_agent_statistics(self) -> Dict[str, Dict]:
        """Get statistics for each agent's contributions"""
        stats = {}
        for agent_name, contrib in self.agent_contributions.items():
            stats[agent_name] = {
                'items_posted': contrib.items_posted,
                'items_supported': contrib.items_supported,
                'items_contradicted': contrib.items_contradicted,
                'items_refined': contrib.items_refined,
                'total_contributions': (contrib.items_posted + contrib.items_supported + 
                                      contrib.items_contradicted + contrib.items_refined),
                'last_active': contrib.last_active.isoformat() if contrib.last_active else None
            }
        return stats
    
    def export_state(self) -> Dict:
        """Export current blackboard state"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_items': len(self.knowledge_items),
            'high_consensus_items': len(self.get_high_consensus_items()),
            'controversial_items': len(self.get_controversial_items()),
            'agent_statistics': self.get_agent_statistics(),
            'knowledge_items': [
                {
                    'id': item.id,
                    'type': item.knowledge_type.value,
                    'source': item.source_agent,
                    'confidence': item.confidence,
                    'consensus': item.get_consensus_score(),
                    'content': item.content
                }
                for item in self.knowledge_items.values()
            ]
        }


class BlackboardController:
    """
    Controls blackboard-based reasoning cycles
    Orchestrates agent interactions through the blackboard
    """
    
    def __init__(self, blackboard: Blackboard):
        self.blackboard = blackboard
        self.agents: Dict[str, Any] = {}
        self.reasoning_history: List[Dict] = []
    
    def register_agent(self, agent_name: str, agent_instance: Any):
        """Register agent with blackboard controller"""
        self.agents[agent_name] = agent_instance
        logger.info(f"✓ Registered agent: {agent_name}")
    
    def run_reasoning_cycle(self, patient_data: Dict, max_iterations: int = 5) -> Dict:
        """
        Run collaborative reasoning cycle
        Agents iteratively post, read, support, contradict, and refine knowledge
        """
        logger.info("\n" + "="*80)
        logger.info("BLACKBOARD REASONING CYCLE")
        logger.info("="*80)
        
        cycle_result = {
            'start_time': datetime.now(),
            'patient_data': patient_data,
            'iterations': [],
            'final_consensus': None
        }
        
        for iteration in range(max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            iteration_result = {
                'iteration': iteration + 1,
                'actions': []
            }
            
            # Phase 1: Each agent posts initial observations
            if iteration == 0:
                logger.info("Phase 1: Initial Observations")
                for agent_name, agent in self.agents.items():
                    # Agent analyzes patient
                    if hasattr(agent, 'analyze_patient'):
                        analysis = agent.analyze_patient(patient_data)
                        
                        # Post observation to blackboard
                        item_id = self.blackboard.post_knowledge(
                            agent_name=agent_name,
                            knowledge_type=KnowledgeType.OBSERVATION,
                            content={
                                'analysis': analysis,
                                'key_finding': str(analysis.get('reasoning', ''))[:200]
                            },
                            confidence=0.7,
                            tags={agent_name.lower(), 'observation'}
                        )
                        
                        iteration_result['actions'].append({
                            'agent': agent_name,
                            'action': 'post_observation',
                            'item_id': item_id
                        })
            
            # Phase 2: Agents read others' observations and respond
            else:
                logger.info(f"Phase 2: Collaborative Refinement")
                
                # Each agent reads blackboard
                for agent_name in self.agents.keys():
                    # Query observations from other agents
                    other_observations = self.blackboard.query_knowledge(
                        knowledge_type=KnowledgeType.OBSERVATION,
                        min_confidence=0.5
                    )
                    
                    # Agent evaluates others' observations
                    for obs in other_observations:
                        if obs.source_agent != agent_name:
                            # Simple agreement logic (in real system, agent would evaluate)
                            # Here we simulate based on agent specialty alignment
                            if self._agents_align(agent_name, obs.source_agent):
                                self.blackboard.support_knowledge(
                                    agent_name=agent_name,
                                    item_id=obs.id,
                                    evidence={'agreement': 'Findings align with my assessment'}
                                )
                                iteration_result['actions'].append({
                                    'agent': agent_name,
                                    'action': 'support',
                                    'item_id': obs.id
                                })
            
            # Phase 3: Post hypotheses based on collective observations
            if iteration == 1:
                logger.info("Phase 3: Hypothesis Generation")
                
                # Get high-consensus observations
                consensus_obs = self.blackboard.get_high_consensus_items(threshold=0.6)
                
                for agent_name in self.agents.keys():
                    # Agent generates hypothesis based on consensus
                    hypothesis_content = {
                        'hypothesis': f"{agent_name} hypothesis based on {len(consensus_obs)} consensus observations",
                        'based_on': [obs.id for obs in consensus_obs[:3]]
                    }
                    
                    item_id = self.blackboard.post_knowledge(
                        agent_name=agent_name,
                        knowledge_type=KnowledgeType.HYPOTHESIS,
                        content=hypothesis_content,
                        confidence=0.75,
                        tags={agent_name.lower(), 'hypothesis'}
                    )
                    
                    iteration_result['actions'].append({
                        'agent': agent_name,
                        'action': 'post_hypothesis',
                        'item_id': item_id
                    })
            
            # Phase 4: Refine hypotheses
            if iteration >= 2:
                logger.info("Phase 4: Hypothesis Refinement")
                
                hypotheses = self.blackboard.query_knowledge(
                    knowledge_type=KnowledgeType.HYPOTHESIS
                )
                
                for hyp in hypotheses:
                    # Other agents can refine
                    for agent_name in self.agents.keys():
                        if agent_name != hyp.source_agent and len(hyp.refinement_history) < 2:
                            # Refine with additional insight
                            self.blackboard.refine_knowledge(
                                agent_name=agent_name,
                                item_id=hyp.id,
                                new_content={'refined_by': agent_name, 'iteration': iteration}
                            )
                            
                            iteration_result['actions'].append({
                                'agent': agent_name,
                                'action': 'refine',
                                'item_id': hyp.id
                            })
                            break
            
            cycle_result['iterations'].append(iteration_result)
            
            # Check for convergence
            high_consensus = self.blackboard.get_high_consensus_items(threshold=0.8)
            if len(high_consensus) >= len(self.agents):
                logger.info(f"✓ Convergence reached at iteration {iteration + 1}")
                break
        
        # Final consensus
        final_consensus = self.blackboard.get_high_consensus_items(threshold=0.7)
        cycle_result['final_consensus'] = [
            {
                'id': item.id,
                'type': item.knowledge_type.value,
                'content': item.content,
                'consensus_score': item.get_consensus_score(),
                'supporting_agents': item.supporting_agents
            }
            for item in final_consensus
        ]
        
        cycle_result['end_time'] = datetime.now()
        
        logger.info(f"\n✓ Reasoning cycle complete")
        logger.info(f"  Total iterations: {len(cycle_result['iterations'])}")
        logger.info(f"  Final consensus items: {len(final_consensus)}")
        
        self.reasoning_history.append(cycle_result)
        
        return cycle_result
    
    def _agents_align(self, agent1: str, agent2: str) -> bool:
        """Check if two agents' specialties align (simplified)"""
        # Cardiology and Metabolic agents often align on CVD risk
        alignments = {
            ('Cardiology', 'Metabolic'): True,
            ('Metabolic', 'Lifestyle'): True,
            ('Cardiology', 'Lifestyle'): True,
        }
        
        return alignments.get((agent1, agent2), False) or alignments.get((agent2, agent1), False)


# Example usage
if __name__ == "__main__":
    from agents.cardiology_agent import CardiologyAgent
    from agents.metabolic_agent import MetabolicAgent
    from agents.lifestyle_agent import LifestyleAgent
    
    print("\n" + "="*80)
    print("BLACKBOARD SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Create blackboard
    blackboard = Blackboard()
    
    # Create controller
    controller = BlackboardController(blackboard)
    
    # Register agents
    controller.register_agent('Cardiology', CardiologyAgent())
    controller.register_agent('Metabolic', MetabolicAgent())
    controller.register_agent('Lifestyle', LifestyleAgent())
    
    # Sample patient data
    patient_data = {
        'demographics': {'age': 55, 'gender': 'male'},
        'physical': {'bmi': 32},
        'vitals': {'systolic_bp': 145, 'diastolic_bp': 92},
        'labs': {
            'glucose_mmol_l': 6.2,
            'hba1c_percent': 6.0,
            'ldl_cholesterol_mmol_l': 4.2
        },
        'lifestyle': {
            'smoking_status': 'current',
            'exercise_hours_per_week': 1
        }
    }
    
    # Run reasoning cycle
    result = controller.run_reasoning_cycle(patient_data, max_iterations=3)
    
    # Display results
    print("\n" + "="*80)
    print("BLACKBOARD REASONING RESULTS")
    print("="*80)
    
    print(f"\nTotal iterations: {len(result['iterations'])}")
    print(f"Final consensus items: {len(result['final_consensus'])}")
    
    print("\nHigh Consensus Items:")
    for item in result['final_consensus']:
        print(f"  • {item['type']}: consensus={item['consensus_score']:.2f}")
        print(f"    Supporting agents: {', '.join(item['supporting_agents'])}")
    
    # Agent statistics
    stats = blackboard.get_agent_statistics()
    print("\nAgent Contributions:")
    for agent, stat in stats.items():
        print(f"  {agent}:")
        print(f"    Posted: {stat['items_posted']}, Supported: {stat['items_supported']}, "
              f"Refined: {stat['items_refined']}")
    
    print("\n" + "="*80)
    print("✓ DEMONSTRATION COMPLETE")
    print("="*80)
