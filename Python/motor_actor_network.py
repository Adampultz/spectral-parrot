# motor_actor_network.py - Simplified actor for motor control only

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import logging
from torch.distributions import Categorical, Beta

logger = logging.getLogger(__name__)

class MotorActorNetwork(nn.Module):
    """
    Simplified actor network for discrete motor control.
    Each motor has 3 actions: CCW (0), HOLD (1), CW (2)
    """
    def __init__(self, 
                 state_dim, 
                 num_motors, 
                 num_actions_per_motor=3,
                 hidden_layers=[64, 64], 
                 use_layernorm=True, 
                 dropout_rate=0.1, 
                 hold_bias=0.5, 
                 step_size_logits_bias=None,
                 activation='relu',
                 min_step_size=10,
                 max_step_size=100,
                 step_size_bias=1.5):
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
        self.num_actions_per_motor = num_actions_per_motor
        self.hold_bias = hold_bias
        self.step_size_logits_bias = step_size_logits_bias
        self.hold_action_index = (num_actions_per_motor - 1) // 2
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.step_size_bias = step_size_bias

        # Select activation function
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'elu':
            self.activation_fn = nn.ELU()
        else:
            self.activation_fn = nn.ReLU()

        # Build feature extractor from list of layer sizes
        layers = []
        in_features = state_dim
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(self.activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)

        final_hidden = hidden_layers[-1] if hidden_layers else state_dim
        # Action heads (use last hidden size)
        self.motor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(final_hidden, final_hidden // 2),
                self.activation_fn,
                nn.Linear(final_hidden // 2, num_actions_per_motor)  # VARIABLE
            ) for _ in range(num_motors)
        ])

        # Magnitude head - Beta distribution parameters per motor
        # Outputs 2 values per motor: alpha and beta for Beta(alpha, beta)
        self.magnitude_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            self.activation_fn,
            nn.Linear(final_hidden // 2, num_motors * 2),
            nn.Softplus()  # Ensures positive parameters
        )
        
        # Temperature for exploration control
        self.temperature = 1.0
        
        # Initialize weights to prefer HOLD action initially
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with bias toward HOLD action."""
        for motor_head in self.motor_heads:
            final_layer = motor_head[-1]
            nn.init.normal_(final_layer.weight, std=0.01)
            nn.init.zeros_(final_layer.bias)
            # Use dynamic hold_action_index instead of hardcoded 1
            final_layer.bias.data[self.hold_action_index] = self.hold_bias

    def _apply_action_biases(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply biases to action logits.
        
        Args:
            logits: Action logits [batch_size, num_motors, num_actions_per_motor]
        
        Returns:
            logits: Biased logits with same shape
        """
        # Hold bias (always applied)
        if self.hold_bias != 0.0:
            logits[:, :, self.hold_action_index] += self.hold_bias
        
        # Step size biases (optional - for exploration)
        if self.step_size_logits_bias is not None:
            num_step_sizes = (self.num_actions_per_motor - 1) // 2
            
            for i, bias in enumerate(self.step_size_logits_bias):
                # Apply to CCW actions
                logits[:, :, i] += bias
                # Apply to CW actions (mirror on other side of HOLD)
                logits[:, :, self.hold_action_index + 1 + i] += bias
        
        return logits
    
    def forward(self, state):
        """Forward pass through the network."""
        # Input validation
        state = torch.nan_to_num(state, nan=0.0)
        state = torch.clamp(state, -10.0, 10.0)
        
        # Extract features
        features = self.feature_extractor(state)
        
        # Generate action distribution for each motor
        action_dists = []
        for motor_head in self.motor_heads:
            logits = motor_head(features)
            
            # Apply action biases (HOLD + optional step size biases)
            logits = logits.unsqueeze(0) if logits.dim() == 1 else logits
            logits = logits.unsqueeze(1) if logits.dim() == 2 else logits
            logits = self._apply_action_biases(logits)
            logits = logits.squeeze(1)
            
            # Apply temperature scaling for exploration
            if self.training and self.temperature != 1.0:
                logits = logits / self.temperature
            
            action_dists.append(Categorical(logits=logits))

        # Generate Beta distribution parameters for magnitudes
            magnitude_params = self.magnitude_head(features)
            # Split into alpha and beta, add small constant to ensure > 0
            alphas = magnitude_params[..., :self.num_motors] + 1.0
            betas = magnitude_params[..., self.num_motors:] + 1.0

        return action_dists, alphas, betas
    
    def sample(self, state):
        """
        Sample actions from the policy.
        
        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
            
        Returns:
            motor_commands: Tensor of actual motor step commands
            directions: Tensor of direction actions (0=CCW, 1=HOLD, 2=CW)
            magnitudes: Tensor of sampled magnitude values [0, 1]
            log_prob: Combined log probability (directions + magnitudes)
        """
        action_dists, alphas, betas = self.forward(state)
        
        # Sample direction for each motor
        directions = torch.stack([dist.sample() for dist in action_dists], dim=-1)
        
        # Calculate log probabilities of directions
        if directions.dim() == 1:
            direction_log_probs = torch.stack([
                dist.log_prob(directions[i]) 
                for i, dist in enumerate(action_dists)
            ], dim=-1)
        else:
            direction_log_probs = torch.stack([
                dist.log_prob(directions[:, i]) 
                for i, dist in enumerate(action_dists)
            ], dim=-1)
        
        # Sample magnitudes from Beta distributions
        if alphas.dim() == 1:
            # Unbatched
            magnitude_dists = [Beta(alphas[i], betas[i]) for i in range(self.num_motors)]
            magnitudes = torch.stack([dist.sample() for dist in magnitude_dists], dim=-1)
            magnitude_log_probs = torch.stack([
                dist.log_prob(magnitudes[i]) 
                for i, dist in enumerate(magnitude_dists)
            ], dim=-1)
        else:
            # Batched
            magnitude_dists = [Beta(alphas[:, i], betas[:, i]) for i in range(self.num_motors)]
            magnitudes = torch.stack([dist.sample() for dist in magnitude_dists], dim=-1)
            magnitude_log_probs = torch.stack([
                dist.log_prob(magnitudes[:, i]) 
                for i, dist in enumerate(magnitude_dists)
            ], dim=-1)
        
        # Apply bias transformation (optional - can still use this)
        magnitude_biased = torch.pow(magnitudes, self.step_size_bias)
        
        # Scale to step sizes
        step_sizes = self.min_step_size + magnitude_biased * (self.max_step_size - self.min_step_size)
        
        # Convert directions to movement values
        direction_map = torch.tensor([-1, 0, 1], device=directions.device, dtype=torch.float32)
        direction_values = direction_map[directions]
        
        # Apply HOLD mask
        hold_mask = (directions == 1)
        step_sizes = step_sizes * (~hold_mask).float()
        magnitude_biased = magnitude_biased * (~hold_mask).float()
        
        # Final motor commands
        motor_commands = direction_values * step_sizes
        
        # Combined log probability (sum over all motors)
        total_log_prob = direction_log_probs.sum(dim=-1) + magnitude_log_probs.sum(dim=-1)
        
        return motor_commands, directions, magnitude_biased, total_log_prob
    
    def evaluate(self, state, directions, magnitudes):
        """
        Evaluate log probability and entropy of given actions.
        
        Args:
            state: State tensor [batch_size, state_dim]
            directions: Direction tensor [batch_size, num_motors]
            magnitudes: Magnitude tensor [batch_size, num_motors]
            
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of the distributions
        """
        action_dists, alphas, betas = self.forward(state)
        
        # Evaluate directions
        direction_log_probs = torch.stack([
            dist.log_prob(directions[:, i]) 
            for i, dist in enumerate(action_dists)
        ], dim=-1)
        
        direction_entropies = torch.stack([
            dist.entropy() for dist in action_dists
        ], dim=-1)
        
        # Evaluate magnitudes (need to reverse bias if applied)
        magnitudes_unbiased = torch.pow(magnitudes, 1.0/self.step_size_bias)
        
        # Clamp to valid range
        magnitudes_unbiased = torch.clamp(magnitudes_unbiased, 1e-6, 1.0 - 1e-6)
        
        # Create Beta distributions and compute log probs
        magnitude_dists = [Beta(alphas[:, i], betas[:, i]) for i in range(self.num_motors)]
        magnitude_log_probs = torch.stack([
            dist.log_prob(magnitudes_unbiased[:, i])
            for i, dist in enumerate(magnitude_dists)
        ], dim=-1)
        
        magnitude_entropies = torch.stack([
            dist.entropy() for dist in magnitude_dists
        ], dim=-1)
        
        # Combined values
        total_log_prob = direction_log_probs.sum(dim=-1) + magnitude_log_probs.sum(dim=-1)
        total_entropy = direction_entropies.sum(dim=-1) + magnitude_entropies.sum(dim=-1)
        
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