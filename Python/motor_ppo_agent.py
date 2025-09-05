# motor_ppo_agent.py - Simplified PPO agent for motor control

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class MotorPPOMemory:
    """Simplified memory buffer for motor-only actions."""
    def __init__(self, batch_size):
        self.states = []
        self.actions = []  # Just motor actions now
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.batch_size = batch_size
        
    def store(self, state, actions, log_prob, reward, value, done):
        """Store a transition in memory."""
        self.states.append(state)
        self.actions.append(actions)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def clear(self):
        """Clear memory after update."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def generate_batches(self):
        """Generate training batches."""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches


class MotorPPOAgent:
    """
    Simplified PPO agent for motor control.
    Only handles discrete actions for motors.
    """
    def __init__(self, state_dim, num_motors, device='cpu', 
                 hidden_size=64, lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_param=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                 batch_size=64):
        """Initialize the motor PPO agent."""
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_motors = num_motors
        
        # Import the motor actor network
        from motor_actor_network import MotorActorNetwork
        
        # Initialize networks
        self.actor = MotorActorNetwork(state_dim, num_motors, hidden_size).to(device)
        
        # Critic can be simpler too
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize memory
        self.memory = MotorPPOMemory(batch_size)
        
        # Exploration temperature schedule
        self.initial_temperature = 2.0  # Up from 1.0
        self.temperature_decay = 0.995 
        self.min_temperature = 0.3
        self.current_temperature = self.initial_temperature
        
    def select_action(self, state):
        """
        Select motor actions according to the policy.
        
        Args:
            state: Current state
            
        Returns:
            actions: Array of motor actions (0=CCW, 1=HOLD, 2=CW)
            log_prob: Log probability of the actions
            value: Estimated value of the state
        """
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            # Get actions from actor
            actions, log_prob = self.actor.sample(state_tensor)
            
            # Get value from critic
            value = self.critic(state_tensor).squeeze()
            
        # Convert to numpy
        actions_np = actions.cpu().numpy().squeeze()
        log_prob_np = log_prob.cpu().item()
        value_np = value.cpu().item()
        
        return actions_np, log_prob_np, value_np
    
    def store_transition(self, state, actions, log_prob, reward, value, done):
        """Store a transition in memory."""
        self.memory.store(state, actions, log_prob, reward, value, done)
    
    def update(self, next_value, n_epochs=10):
        """
        Update policy and value networks using PPO.
        
        Args:
            next_value: Value of the next state for bootstrapping
            n_epochs: Number of epochs to train
            
        Returns:
            actor_loss, critic_loss: Average losses
        """
        # Compute returns and advantages
        returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.memory.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.memory.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Update for multiple epochs
        actor_losses, critic_losses = [], []
        
        for _ in range(n_epochs):
            # Generate batches
            batches = self.memory.generate_batches()
            
            for batch_indices in batches:
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                new_log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                state_values = self.critic(batch_states).squeeze()
                
                # Compute advantages
                advantages = batch_returns - state_values.detach()
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(state_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_actor_loss = actor_loss + self.entropy_coef * entropy_loss
                total_critic_loss = self.value_coef * value_loss
                
                # Update networks
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Record losses
                actor_losses.append(actor_loss.item())
                critic_losses.append(value_loss.item())
        
        # Clear memory
        self.memory.clear()
        
        # Update exploration temperature
        self.current_temperature = max(
            self.min_temperature,
            self.current_temperature * self.temperature_decay
        )
        self.actor.set_temperature(self.current_temperature)
        
        return np.mean(actor_losses), np.mean(critic_losses)
    
    def compute_gae(self, next_value):
        """Compute returns using Generalized Advantage Estimation."""
        rewards = self.memory.rewards
        values = self.memory.values + [next_value]
        dones = self.memory.dones
        
        returns = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
            
        return returns
    
    def save(self, path):
        """Save the agent's networks and state."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'temperature': self.current_temperature,
            'num_motors': self.num_motors
        }, path)
        
    def load(self, path):
        """Load the agent's networks and state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.current_temperature = checkpoint.get('temperature', 1.0)
        self.actor.set_temperature(self.current_temperature)