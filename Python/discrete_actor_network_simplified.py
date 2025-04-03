import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class DiscreteActorNetwork(nn.Module):
    """
    Actor network for PPO with discrete actions for oscillator frequencies
    and continuous actions for oscillator amplitudes.
    """
    def __init__(self, state_dim, num_oscillators, hidden_size=128, log_std_min=-20, log_std_max=2):
        """
        Initialize the actor network.
        
        Args:
            state_dim: Dimension of state (observation) space
            num_oscillators: Number of oscillators to control
            hidden_size: Size of hidden layers
            log_std_min: Minimum log standard deviation for continuous actions
            log_std_max: Maximum log standard deviation for continuous actions
        """
        super(DiscreteActorNetwork, self).__init__()
        
        self.num_oscillators = num_oscillators
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh()
        )
        
        # Discrete action heads for each oscillator's frequency (3 actions per oscillator)
        # -1 = decrease, 0 = maintain, 1 = increase
        self.freq_action_heads = nn.ModuleList([
            nn.Linear(hidden_size, 3) for _ in range(num_oscillators)
        ])
        
        # Continuous action head for amplitudes
        self.amp_mean = nn.Linear(hidden_size, num_oscillators)
        self.amp_log_std = nn.Linear(hidden_size, num_oscillators)
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            freq_dists: List of Categorical distributions for frequency actions
            amp_dist: Normal distribution for amplitude actions
        """
        state = torch.nan_to_num(state, nan=0.0)
        state = torch.clamp(state, -10.0, 10.0)  # Prevent extreme values

        x = self.features(state)
        
        # Generate discrete frequency action distributions
        freq_dists = [
            Categorical(logits=head(x)) for head in self.freq_action_heads
        ]
        
        # Generate continuous amplitude action distribution
        amp_mean = self.amp_mean(x)
        amp_log_std = self.amp_log_std(x)
        amp_log_std = torch.clamp(amp_log_std, self.log_std_min, self.log_std_max)
        amp_std = amp_log_std.exp()
        
        amp_dist = Normal(amp_mean, amp_std)
        
        return freq_dists, amp_dist
    
    def sample(self, state):
        """
        Sample actions from the policy distribution.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            actions: Dictionary with frequency and amplitude actions
            log_prob: Combined log probability of all actions
        """
        freq_dists, amp_dist = self.forward(state)
        
        # Sample discrete frequency actions
        freq_actions = torch.stack([dist.sample() for dist in freq_dists], dim=-1)
        freq_log_probs = torch.stack([dist.log_prob(freq_actions[:, i]) 
                                     for i, dist in enumerate(freq_dists)], dim=-1)
        
        # Sample continuous amplitude actions
        amp_actions = amp_dist.sample()
        amp_log_probs = amp_dist.log_prob(amp_actions).sum(dim=-1, keepdim=True)
        
        # Combine actions and calculate total log probability
        actions = {
            'freq_actions': freq_actions,
            'amp_actions': amp_actions
        }
        
        # Sum log probabilities across all actions
        log_prob = torch.cat([freq_log_probs, amp_log_probs], dim=-1).sum(dim=-1)
        
        return actions, log_prob
    
    def evaluate(self, state, actions):
        """
        Evaluate log probability of actions given a state.
        
        Args:
            state: State tensor [batch_size, state_dim]
            actions: Dictionary with frequency and amplitude actions
            
        Returns:
            log_prob: Log probability of all actions
            entropy: Combined entropy of all distributions
        """
        freq_dists, amp_dist = self.forward(state)
        
        # Get frequency and amplitude actions
        freq_actions = actions['freq_actions']
        amp_actions = actions['amp_actions']
        
        # Calculate log probabilities for frequency actions
        freq_log_probs = torch.stack([dist.log_prob(freq_actions[:, i]) 
                                     for i, dist in enumerate(freq_dists)], dim=-1)
        
        # Calculate log probabilities for amplitude actions
        amp_log_probs = amp_dist.log_prob(amp_actions).sum(dim=-1, keepdim=True)
        
        # Calculate entropies
        freq_entropies = torch.stack([dist.entropy() for dist in freq_dists], dim=-1)
        amp_entropy = amp_dist.entropy().sum(dim=-1, keepdim=True)
        
        # Combine log probabilities and entropies
        log_prob = torch.cat([freq_log_probs, amp_log_probs], dim=-1).sum(dim=-1)
        entropy = torch.cat([freq_entropies, amp_entropy], dim=-1).sum(dim=-1)
        
        return log_prob, entropy