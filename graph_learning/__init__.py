"""
Graph Neural Network Learning Layer
Hybrid mechanistic-learned architecture for organ interactions
"""

from .organ_gnn import OrganGraphNetwork
from .physics_informed_layer import PhysicsInformedGNN

__all__ = ['OrganGraphNetwork', 'PhysicsInformedGNN']
