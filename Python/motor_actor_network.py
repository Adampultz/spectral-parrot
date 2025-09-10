# motor_actor_network.py - Simplified actor for motor control only

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MotorActorNetwork(nn.Module):
    """
    Simplified actor network for discrete motor control.
    Each motor has 3 actions: CCW (0), HOLD (1), CW (2)
    """
    def __init__(self, state_dim, num_motors, hidden_size=64, 
                 use_layernorm=True, dropout_rate=0.1, hold_bias=0.5):
        """
        Initialize the motor actor network.
        
        Args:
            state_dim: Dimension of state (observation) - typically 1 for MSSL loss
            num_motors: Number of motors to control
            hidden_size: Size of hidden layers (can be smaller now)
            use_layernorm: Whether to use layer normalization
            dropout_rate: Dropout for regularization
        """
        super(MotorActorNetwork, self).__init__()
        
        self.num_motors = num_motors
        self.state_dim = state_dim
        self.hold_bias = hold_bias
        
        # Simpler architecture for discrete-only control
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Separate action head for each motor (3 actions each)
        self.motor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 3)  # 3 actions: CCW, HOLD, CW
            ) for _ in range(num_motors)
        ])
        
        # Temperature for exploration control
        self.temperature = 1.0
        
        # Initialize weights to prefer HOLD action initially
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with bias toward HOLD action."""
        for motor_head in self.motor_heads:
            # Get the final linear layer
            final_layer = motor_head[-1]
            
            # Small random weights
            nn.init.normal_(final_layer.weight, std=0.01)
            
            # Bias toward HOLD (action 1)
            nn.init.zeros_(final_layer.bias)
            final_layer.bias.data[1] = self.hold_bias  # Slight preference for HOLD
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            action_dists: List of Categorical distributions for each motor
        """
        # Input validation
        state = torch.nan_to_num(state, nan=0.0)
        state = torch.clamp(state, -10.0, 10.0)
        
        # Extract features
        features = self.feature_extractor(state)
        
        # Generate action distribution for each motor
        action_dists = []
        for motor_head in self.motor_heads:
            logits = motor_head(features)
            
            # Apply temperature scaling for exploration
            if self.training and self.temperature != 1.0:
                logits = logits / self.temperature
            
            action_dists.append(Categorical(logits=logits))
        
        return action_dists
    
    def sample(self, state):
        """
        Sample actions from the policy.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            actions: Tensor of motor actions [batch_size, num_motors]
            log_prob: Combined log probability of all actions
        """
        action_dists = self.forward(state)
        
        # Sample action for each motor
        actions = torch.stack([dist.sample() for dist in action_dists], dim=-1)

        

        # Calculate log probabilities
        log_probs = torch.stack([
            # Is this more correct?
            # dist.log_prob(actions[:, i]) 
            dist.log_prob(actions[i]) 
            for i, dist in enumerate(action_dists)
        ], dim=-1)
        
        # Sum log probabilities across all motors
        total_log_prob = log_probs.sum(dim=-1)
        
        return actions, total_log_prob
    
    def evaluate(self, state, actions):
        """
        Evaluate log probability and entropy of given actions.
        
        Args:
            state: State tensor [batch_size, state_dim]
            actions: Action tensor [batch_size, num_motors]
            
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of the action distributions
        """
        action_dists = self.forward(state)
        
        # Calculate log probabilities for given actions
        log_probs = torch.stack([
            dist.log_prob(actions[:, i]) 
            for i, dist in enumerate(action_dists)
        ], dim=-1)
        
        # Calculate entropy for each distribution
        entropies = torch.stack([dist.entropy() for dist in action_dists], dim=-1)
        
        # Sum across motors
        total_log_prob = log_probs.sum(dim=-1)
        total_entropy = entropies.sum(dim=-1)
        
        return total_log_prob, total_entropy
    
    def get_action_probs(self, state):
        """
        Get action probabilities for visualization/debugging.
        
        Returns:
            probs: [batch_size, num_motors, 3] tensor of action probabilities
        """
        action_dists = self.forward(state)
        probs = torch.stack([dist.probs for dist in action_dists], dim=1)
        return probs
    
    def set_temperature(self, temperature):
        """Set exploration temperature (1.0 = normal, >1 = more random)."""
        self.temperature = max(0.1, temperature)  # Minimum temperature of 0.1