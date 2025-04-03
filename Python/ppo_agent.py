import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from discrete_actor_network_simplified import DiscreteActorNetwork

logger = logging.getLogger(__name__)

class CriticNetwork(nn.Module):
    """
    Critic network for PPO. Estimates the value of a state.
    """
    def __init__(self, state_dim, hidden_size=128):
        """
        Initialize the critic network.
        
        Args:
            state_dim: Dimension of state (observation) space
            hidden_size: Size of hidden layers
        """
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            value: Estimated value of the state [batch_size, 1]
        """
        return self.network(state)


class PPOHybridMemory:
    """
    Memory buffer for PPO algorithm with hybrid discrete-continuous actions.
    """
    def __init__(self, batch_size):
        self.states = []
        self.freq_actions = []
        self.amp_actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.batch_size = batch_size
        
    def store(self, state, actions, log_prob, reward, value, done):
        """
        Store a transition in memory.
        
        Args:
            state: Current state
            actions: Dictionary containing 'freq_actions' and 'amp_actions'
            log_prob: Log probability of the actions
            reward: Reward received
            value: Estimated value of the state
            done: Whether the episode terminated
        """
        self.states.append(state)
        self.freq_actions.append(actions['freq_actions'])
        self.amp_actions.append(actions['amp_actions'])
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def clear(self):
        """
        Clear the memory after an update.
        """
        self.states = []
        self.freq_actions = []
        self.amp_actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def generate_batches(self):
        """
        Generate training batches.
        
        Returns:
            Generator of training batches.
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches


class HybridPPOAgent:
    """
    PPO agent for controlling oscillators with discrete frequency actions and continuous amplitude actions.
    """
    def __init__(self, state_dim, num_oscillators, device='cpu', hidden_size=128,
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_param=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, batch_size=64):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of state space
            num_oscillators: Number of oscillators to control
            device: Device to run computations on
            hidden_size: Size of hidden layers
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_param: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Batch size for updates
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_oscillators = num_oscillators
        
        # Initialize networks
        self.actor = DiscreteActorNetwork(state_dim, num_oscillators, hidden_size).to(device)
        self.critic = CriticNetwork(state_dim, hidden_size).to(device)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize memory
        self.memory = PPOHybridMemory(batch_size)
        
    def select_action(self, state):
        """
        Select an action according to the policy.
        
        Args:
            state: Current state
            
        Returns:
            actions: Dictionary with discrete frequency actions and continuous amplitude actions
            log_prob: Log probability of the actions
            value: Estimated value of the state
        """
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            actions, log_prob = self.actor.sample(state)
            value = self.critic(state)
            
        # Convert to numpy for environment interaction
        np_actions = {
            'freq_actions': actions['freq_actions'].cpu().numpy().squeeze(),
            'amp_actions': actions['amp_actions'].cpu().numpy().squeeze()
        }
            
        return np_actions, log_prob.cpu().item(), value.cpu().item()
    
    def store_transition(self, state, actions, log_prob, reward, value, done):
        """
        Store a transition in memory.
        
        Args:
            state: Current state
            actions: Dictionary with frequency and amplitude actions
            log_prob: Log probability of the actions
            reward: Reward received
            value: Estimated value of the state
            done: Whether the episode terminated
        """
        self.memory.store(state, actions, log_prob, reward, value, done)
    
    def update(self, next_value, n_epochs=10):
        """
        Update policy and value networks using PPO.
        
        Args:
            next_value: Estimated value of the next state
            n_epochs: Number of epochs to update for
            
        Returns:
            actor_loss, critic_loss: Final losses
        """
        # Compute returns and advantages
        returns = self.compute_gae(next_value)
        
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        freq_actions = torch.LongTensor(np.array(self.memory.freq_actions)).to(self.device)
        amp_actions = torch.FloatTensor(np.array(self.memory.amp_actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.memory.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Update policy for n_epochs
        actor_losses, critic_losses, entropy_losses = [], [], []
        
        for _ in range(n_epochs):
            # Generate batches
            batches = self.memory.generate_batches()
            
            # Process each batch
            for batch_indices in batches:
                # Get batch data
                batch_states = states[batch_indices]
                batch_freq_actions = freq_actions[batch_indices]
                batch_amp_actions = amp_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Combine actions into dictionary format for actor evaluation
                batch_actions = {
                    'freq_actions': batch_freq_actions,
                    'amp_actions': batch_amp_actions
                }
                
                # Evaluate actions
                new_log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                state_values = self.critic(batch_states).squeeze()
                
                # Compute advantage
                advantages = batch_returns - state_values.detach()
                
                # Normalize advantages (helps with training stability)
                if len(advantages) > 1:  # Only normalize if we have multiple samples
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Compute actor loss (PPO clip objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss (MSE)
                value_loss = F.mse_loss(state_values, batch_returns)
                value_loss = self.value_coef * value_loss
                
                # Compute entropy bonus (encourages exploration)
                entropy_loss = -self.entropy_coef * entropy.mean()
                
                # Combine losses
                total_loss = policy_loss + value_loss + entropy_loss
                
                # Perform backpropagation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                # Option 1: Backpropagate combined loss (simpler but less control)
                # total_loss.backward()
                
                # Option 2: Separate backpropagation for actor and critic (more control)
                policy_loss.backward(retain_graph=True)
                value_loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                # Apply updates
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Store losses for logging
                actor_losses.append(policy_loss.item())
                critic_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # Optional: Additional diagnostic information
                with torch.no_grad():
                    approx_kl_divergence = 0.5 * ((new_log_probs - batch_old_log_probs) ** 2).mean().item()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.clip_param).float().mean().item()
                    explained_variance = 1 - (advantages.var() / (batch_returns.var() + 1e-8)).item()
                
                # Log additional information every 10 batches
                if len(actor_losses) % 10 == 0:
                    logger.debug(f"Update stats - KL: {approx_kl_divergence:.4f}, "
                            f"Clip: {clip_fraction:.2f}, "
                            f"Explained var: {explained_variance:.4f}")
        
        # Clear memory after update
        self.memory.clear()
        
        # Return mean losses
        return np.mean(actor_losses), np.mean(critic_losses)

    def compute_gae(self, next_value):
        """
        Compute returns using Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Estimated value of the next state
            
        Returns:
            returns: Computed returns
        """
        rewards = self.memory.rewards
        values = self.memory.values + [next_value]
        dones = self.memory.dones
        
        returns = []
        gae = 0
        
        # Compute GAE
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
            
        return returns
    
    def save(self, path):
        """
        Save the agent's networks.
        
        Args:
            path: Path to save to
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """
        Load the agent's networks.
        
        Args:
            path: Path to load from
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])