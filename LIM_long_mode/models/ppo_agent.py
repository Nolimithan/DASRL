"""
PPO agent module.

Implements a PPO-based RL agent for portfolio weight optimization.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class PPOAgent:
    """PPO agent for portfolio optimization."""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_ratio=0.2,
                 value_coef=0.5, entropy_coef=0.01, use_literature_baseline=False):
        """Initialize the PPO agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.use_literature_baseline = use_literature_baseline
        
        # Create actor and critic networks
        self._build_networks()
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # Training metrics
        self.training_metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_rewards': []
        }
    
    def _build_networks(self):
        """Build actor and critic networks."""
        if self.use_literature_baseline:
            # Use literature baseline architecture
            # Actor network
            inputs = tf.keras.layers.Input(shape=(self.state_dim,))
            x = tf.keras.layers.Dense(64, activation='tanh')(inputs)
            x = tf.keras.layers.Dense(32, activation='tanh')(x)
            outputs = tf.keras.layers.Dense(self.action_dim, activation='softplus')(x)
            self.actor = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Critic network
            inputs = tf.keras.layers.Input(shape=(self.state_dim,))
            x = tf.keras.layers.Dense(64, activation='tanh')(inputs)
            x = tf.keras.layers.Dense(32, activation='tanh')(x)
            outputs = tf.keras.layers.Dense(1)(x)
            self.critic = tf.keras.Model(inputs=inputs, outputs=outputs)
        else:
            # Use improved architecture
            # Actor network
            inputs = tf.keras.layers.Input(shape=(self.state_dim,))
            x = tf.keras.layers.Dense(256, activation='relu')(inputs)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            outputs = tf.keras.layers.Dense(self.action_dim, activation='softplus')(x)
            self.actor = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Critic network
            inputs = tf.keras.layers.Input(shape=(self.state_dim,))
            x = tf.keras.layers.Dense(256, activation='relu')(inputs)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            outputs = tf.keras.layers.Dense(1)(x)
            self.critic = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def predict(self, state):
        """Predict action (portfolio weights)."""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state_tensor).numpy()[0]
        
        # Ensure non-negative weights and normalize
        actions = np.maximum(0, actions)
        actions_sum = np.sum(actions)
        
        if actions_sum > 0:
            actions = actions / actions_sum
        else:
            # If all weights are 0, use uniform distribution
            actions = np.ones(self.action_dim) / self.action_dim
            
        return actions
    
    def train(self, env, epochs=100, steps_per_epoch=1000):
        """Train the agent."""
        for epoch in range(epochs):
            # Collect trajectories
            states, actions, rewards, next_states, dones, log_probs, values = self._collect_trajectory(env, steps_per_epoch)
            
            # Compute advantages and returns
            advantages, returns = self._compute_advantages_and_returns(rewards, values, dones)
            
            # Update policy and value function
            actor_loss, critic_loss, entropy = self._update_policy(states, actions, log_probs, advantages, returns)
            
            # Record training metrics
            self.training_metrics['actor_loss'].append(actor_loss)
            self.training_metrics['critic_loss'].append(critic_loss)
            self.training_metrics['entropy'].append(entropy)
            self.training_metrics['total_rewards'].append(np.sum(rewards))
            
            # Print training progress
            print(f"Epoch {epoch+1}/{epochs} | Total reward: {np.sum(rewards):.2f} | "
                  f"Actor loss: {actor_loss:.4f} | Critic loss: {critic_loss:.4f} | Entropy: {entropy:.4f}")
    
    def _collect_trajectory(self, env, max_steps):
        """Collect training trajectories."""
        # Initialize containers
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []
        values = []
        
        # Reset environment
        state = env.reset()
        
        # Validate state
        if np.isnan(state).any():
            print("Warning: initial state has NaNs, replacing with 0")
            state = np.nan_to_num(state, nan=0.0)
        
        for step in range(max_steps):
            # Record current state
            states.append(state)
            
            # Get value estimate
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            value = self.critic(state_tensor).numpy()[0, 0]
            values.append(value)
            
            # Get policy distribution parameters
            mu = self.actor(state_tensor).numpy()[0]
            
            # Check for NaNs in mu
            if np.isnan(mu).any():
                print("Warning: action distribution params have NaNs, using uniform distribution")
                mu = np.ones(self.action_dim) / self.action_dim
            
            # Add exploration noise
            noise_std = 0.1
            noise = np.random.normal(0, noise_std, size=self.action_dim)
            action = np.maximum(0, mu + noise)  # Ensure non-negative weights
            
            # Normalize action to sum to 1
            action_sum = np.sum(action)
            if action_sum > 0:
                action = action / action_sum
            else:
                action = np.ones(self.action_dim) / self.action_dim
            
            # Compute log probability (simplified)
            log_prob = -0.5 * np.sum(np.square((action - mu) / noise_std))
            
            # Interact with environment
            try:
                next_state, reward, done, _ = env.step(action)
                
                # Validate reward
                if np.isnan(reward) or np.isinf(reward):
                    print(f"Warning: invalid reward ({reward}), replacing with 0")
                    reward = 0.0
                
                # Clip reward to avoid extreme values
                reward = np.clip(reward, -10.0, 10.0)
                
                # Validate next state
                if np.isnan(next_state).any():
                    print("Warning: next state has NaNs, using current state")
                    next_state = np.copy(state)
            except Exception as e:
                print(f"Error interacting with environment: {e}")
                # Use safe defaults on error
                next_state = np.copy(state)
                reward = 0.0
                done = True
            
            # Record data
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            log_probs.append(log_prob)
            
            # Update state
            state = next_state
            
            # Reset if environment is done
            if done:
                state = env.reset()
                # Validate reset state
                if np.isnan(state).any():
                    print("Warning: reset state has NaNs, replacing with 0")
                    state = np.nan_to_num(state, nan=0.0)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        log_probs = np.array(log_probs, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        
        # Final data check
        if np.isnan(states).any() or np.isnan(actions).any() or np.isnan(rewards).any() or \
           np.isnan(log_probs).any() or np.isnan(values).any():
            print("Warning: trajectory data contains NaNs, cleaning")
            states = np.nan_to_num(states, nan=0.0)
            actions = np.nan_to_num(actions, nan=1.0/self.action_dim)
            rewards = np.nan_to_num(rewards, nan=0.0)
            log_probs = np.nan_to_num(log_probs, nan=0.0)
            values = np.nan_to_num(values, nan=0.0)
            
            # Ensure actions are normalized
            actions_sum = np.sum(actions, axis=1, keepdims=True)
            mask = (actions_sum > 0)
            actions = np.where(mask, actions / actions_sum, np.ones_like(actions) / self.action_dim)
        
        return states, actions, rewards, next_states, dones, log_probs, values
    
    def _compute_advantages_and_returns(self, rewards, values, dones):
        """Compute advantages and returns (GAE)."""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0
        
        # Check for NaNs in inputs
        if np.isnan(rewards).any() or np.isnan(values).any():
            print("Warning: input data contains NaNs")
            # Replace NaNs with 0
            rewards = np.nan_to_num(rewards, nan=0.0)
            values = np.nan_to_num(values, nan=0.0)
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1 or dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Guard against extreme deltas
            delta = np.clip(delta, -10.0, 10.0)
            
            # GAE computation
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            
            # Guard against extreme GAE
            gae = np.clip(gae, -10.0, 10.0)
            
            advantages[t] = gae
            
            # Compute returns
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages, avoid divide-by-zero
        adv_std = np.std(advantages)
        if adv_std < 1e-10:
            adv_std = 1.0
            
        advantages = (advantages - np.mean(advantages)) / adv_std
        
        # Final check
        if np.isnan(advantages).any() or np.isnan(returns).any():
            print("Warning: advantages or returns contain NaNs, replacing with zeros")
            advantages = np.nan_to_num(advantages, nan=0.0)
            returns = np.nan_to_num(returns, nan=0.0)
        
        return advantages, returns
    
    @tf.function
    def _update_policy(self, states, actions, old_log_probs, advantages, returns):
        """Update policy and value function with PPO."""
        with tf.GradientTape() as tape:
            # Compute new policy distribution
            mu = self.actor(states)
            
            # Compute new log-probabilities (simplified)
            noise_std = 0.1
            new_log_probs = -0.5 * tf.reduce_sum(
                tf.square((actions - mu) / noise_std), axis=1
            )
            
            # Compute ratio
            ratio = tf.exp(new_log_probs - old_log_probs)
            
            # Guard against NaN/inf ratios
            ratio = tf.clip_by_value(ratio, 1e-10, 10.0)
            
            # Compute clipped objective
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            
            # Compute policy loss
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            # Compute entropy (simplified)
            entropy = tf.reduce_mean(tf.math.log(2 * np.pi * noise_std**2) / 2)
            
            # Compute value loss
            values = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))
            
            # Apply gradient clipping to prevent explosion
            actor_loss = tf.clip_by_value(actor_loss, -10.0, 10.0)
            critic_loss = tf.clip_by_value(critic_loss, -10.0, 10.0)
            
            # Total loss
            total_loss = actor_loss - self.entropy_coef * entropy + self.value_coef * critic_loss
        
        # Compute and apply gradients
        variables = self.actor.trainable_variables + self.critic.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        
        # Clip gradients to prevent explosion
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        
        # Check gradients for NaNs
        if tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None]):
            print("Warning: gradients contain NaNs, skipping update")
            return actor_loss, critic_loss, entropy
            
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return actor_loss, critic_loss, entropy
    
    def save(self, path):
        """Save model weights."""
        # Ensure path ends with .weights.h5
        actor_path = f"{path}_actor.weights.h5"
        critic_path = f"{path}_critic.weights.h5"
        
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print(f"Model weights saved to {path}")
        
        # Save training metrics
        np.save(f"{path}_metrics.npy", self.training_metrics)
    
    def load(self, path):
        """Load model weights."""
        try:
            self.actor.load_weights(f"{path}_actor.weights.h5")
            self.critic.load_weights(f"{path}_critic.weights.h5")
            
            # Try loading training metrics
            metrics_path = f"{path}_metrics.npy"
            if os.path.exists(metrics_path):
                self.training_metrics = np.load(metrics_path, allow_pickle=True).item()
                
            print(f"Model loaded: {path}")
        except Exception as e:
            print(f"Failed to load model: {e}")

