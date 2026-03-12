"""
Body System Agent - MiroFish-Inspired Autonomous Organ/System Agent
Each organ/system is an intelligent agent with state, memory, and personality
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from enum import Enum


class AgentState(Enum):
    """Agent health states"""
    OPTIMAL = "optimal"
    COMPENSATING = "compensating"
    STRESSED = "stressed"
    FAILING = "failing"


@dataclass
class AgentMemory:
    """Long-term memory for agent"""
    timestamp: datetime
    event: str
    state_snapshot: Dict[str, Any]
    impact: float  # -1 to 1, negative is harmful
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'event': self.event,
            'state_snapshot': self.state_snapshot,
            'impact': self.impact
        }


@dataclass
class AgentPersonality:
    """Personality traits that affect agent behavior"""
    resilience: float = 0.5  # 0-1, ability to recover from stress
    reactivity: float = 0.5  # 0-1, how quickly responds to changes
    adaptability: float = 0.5  # 0-1, ability to adapt to new conditions
    cooperation: float = 0.5  # 0-1, willingness to help other agents
    
    def to_dict(self):
        return {
            'resilience': self.resilience,
            'reactivity': self.reactivity,
            'adaptability': self.adaptability,
            'cooperation': self.cooperation
        }


class BodySystemAgent:
    """
    Base class for all body system agents
    Inspired by MiroFish's autonomous agents with personality, memory, and logic
    """
    
    def __init__(
        self,
        name: str,
        initial_state: Dict[str, Any],
        personality: Optional[AgentPersonality] = None,
        memory_capacity: int = 1000
    ):
        self.name = name
        self.state = initial_state
        self.personality = personality or AgentPersonality()
        self.memory: List[AgentMemory] = []
        self.memory_capacity = memory_capacity
        self.environment = None
        self.health_status = AgentState.OPTIMAL
        self.stress_level = 0.0  # 0-1
        self.age_days = 0
        
    def perceive(self, environment: 'InternalMilieu') -> Dict[str, Any]:
        """
        Perceive current internal environment
        Similar to MiroFish agents sensing their world
        """
        perceptions = {
            'timestamp': datetime.now(),
            'environment_state': environment.get_state(),
            'signals_for_me': environment.get_signals_for(self.name),
            'other_agents_state': environment.get_other_agents_state(self.name)
        }
        return perceptions
    
    def decide(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decisions based on current state and perceptions
        Override in subclasses for specific organ logic
        """
        # Base decision logic
        decision = {
            'action': 'maintain',
            'intensity': 1.0,
            'target_agents': [],
            'signals_to_send': {}
        }
        
        # Adjust based on stress level
        if self.stress_level > 0.7:
            decision['action'] = 'emergency_response'
            decision['intensity'] = 1.5
        elif self.stress_level > 0.4:
            decision['action'] = 'compensate'
            decision['intensity'] = 1.2
        
        return decision
    
    def act(self, decision: Dict[str, Any]) -> None:
        """Execute decision and update own state"""
        # Update state based on action
        if decision['action'] == 'emergency_response':
            self.stress_level = min(1.0, self.stress_level + 0.1)
        elif decision['action'] == 'compensate':
            self.stress_level = min(1.0, self.stress_level + 0.05)
        else:
            # Recovery
            self.stress_level = max(0.0, self.stress_level - 0.02 * self.personality.resilience)
        
        # Update health status
        self._update_health_status()
    
    def interact(self, other_agents: List['BodySystemAgent']) -> List[Dict[str, Any]]:
        """
        Interact with other agents
        Key to swarm intelligence emergence
        """
        messages = []
        
        for agent in other_agents:
            if self._should_communicate_with(agent):
                message = self._create_message(agent)
                messages.append(message)
        
        return messages
    
    def update_memory(self, event: str, impact: float) -> None:
        """Store important events in long-term memory"""
        memory = AgentMemory(
            timestamp=datetime.now(),
            event=event,
            state_snapshot=self.state.copy(),
            impact=impact
        )
        
        self.memory.append(memory)
        
        # Prune old memories if capacity exceeded
        if len(self.memory) > self.memory_capacity:
            # Keep most impactful memories
            self.memory.sort(key=lambda m: abs(m.impact), reverse=True)
            self.memory = self.memory[:self.memory_capacity]
    
    def recall_similar_events(self, current_event: str, n: int = 5) -> List[AgentMemory]:
        """Recall similar past events from memory"""
        # Simple keyword matching - in production use embeddings
        relevant_memories = [
            m for m in self.memory
            if any(word in m.event.lower() for word in current_event.lower().split())
        ]
        
        # Sort by recency and impact
        relevant_memories.sort(
            key=lambda m: (m.timestamp, abs(m.impact)),
            reverse=True
        )
        
        return relevant_memories[:n]
    
    def age_one_day(self) -> None:
        """Age the agent by one day"""
        self.age_days += 1
        
        # Aging effects (very simplified)
        aging_factor = 1.0 - (self.age_days / 365000)  # Decline over ~1000 years
        
        # Personality changes with age
        self.personality.resilience *= aging_factor
        self.personality.adaptability *= aging_factor
    
    def respond_to_query(self, question: str, context: Dict[str, Any]) -> str:
        """
        Respond to user queries about agent state
        Enables MiroFish-style deep interaction
        """
        response = f"[{self.name} Agent] "
        
        if 'state' in question.lower() or 'how' in question.lower():
            response += f"Current status: {self.health_status.value}. "
            response += f"Stress level: {self.stress_level:.1%}. "
            response += f"Key metrics: {self._summarize_state()}"
        
        elif 'why' in question.lower():
            recent_events = self.memory[-5:]
            response += "Recent events affecting me: "
            for mem in recent_events:
                response += f"\n- {mem.event} (impact: {mem.impact:+.2f})"
        
        elif 'future' in question.lower() or 'predict' in question.lower():
            response += self._predict_trajectory()
        
        else:
            response += f"I'm the {self.name} system. Ask me about my state, recent events, or future trajectory."
        
        return response
    
    def _update_health_status(self) -> None:
        """Update health status based on stress level"""
        if self.stress_level < 0.2:
            self.health_status = AgentState.OPTIMAL
        elif self.stress_level < 0.5:
            self.health_status = AgentState.COMPENSATING
        elif self.stress_level < 0.8:
            self.health_status = AgentState.STRESSED
        else:
            self.health_status = AgentState.FAILING
    
    def _should_communicate_with(self, other_agent: 'BodySystemAgent') -> bool:
        """Determine if should send message to other agent"""
        # Communicate if stressed or if other agent is stressed
        return (
            self.stress_level > 0.3 or
            other_agent.stress_level > 0.3 or
            self.personality.cooperation > 0.6
        )
    
    def _create_message(self, target_agent: 'BodySystemAgent') -> Dict[str, Any]:
        """Create message to send to another agent"""
        return {
            'from': self.name,
            'to': target_agent.name,
            'timestamp': datetime.now(),
            'type': 'status_update',
            'content': {
                'stress_level': self.stress_level,
                'health_status': self.health_status.value,
                'needs_help': self.stress_level > 0.7
            }
        }
    
    def _summarize_state(self) -> str:
        """Summarize current state for responses"""
        return str({k: v for k, v in list(self.state.items())[:3]})
    
    def _predict_trajectory(self) -> str:
        """Predict future trajectory based on current state"""
        if self.stress_level > 0.7:
            return "If current stress continues, I expect to enter failing state within 30-60 days."
        elif self.stress_level > 0.4:
            return "Currently compensating. Can maintain this for several months, but need support."
        else:
            return "Trajectory looks stable. Expect to maintain optimal function."
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state"""
        return {
            'name': self.name,
            'state': self.state,
            'personality': self.personality.to_dict(),
            'health_status': self.health_status.value,
            'stress_level': self.stress_level,
            'age_days': self.age_days,
            'memory_count': len(self.memory),
            'recent_memories': [m.to_dict() for m in self.memory[-5:]]
        }
