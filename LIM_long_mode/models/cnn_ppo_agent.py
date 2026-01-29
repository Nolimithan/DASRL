"""
CNN-PPO agent module.

Uses a CNN to process 2D feature matrices for PPO-based portfolio optimization.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class CNNActor(tf.keras.Model):
    """CNN-based policy network (Actor)."""
    
    def __init__(self, action_dim):
        super(CNNActor, self).__init__()
        
        # CNN backbone
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        
        # Pooling
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        
        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        
        # Action mean (output layer)
        self.action_mean = tf.keras.layers.Dense(action_dim, activation='softplus')
        
        # Action std (learnable log std)
        self.action_logstd = tf.Variable(initial_value=-0.5 * np.ones(action_dim, dtype=np.float32))
        
    def call(self, inputs):
        # Ensure 4D input: [batch, height, width, channels]
        if len(inputs.shape) == 3:
            x = tf.expand_dims(inputs, axis=-1)  # Add channel dimension
        else:
            x = inputs
            
        # CNN layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Pooling layers
        x = self.pool(x)
        
        # Fully connected layers
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Output action distribution parameters
        action_mean = self.action_mean(x)
        action_logstd = tf.broadcast_to(self.action_logstd, tf.shape(action_mean))
        
        return action_mean, action_logstd


class CNNCritic(tf.keras.Model):
    """CNN-based value network (Critic)."""
    
    def __init__(self):
        super(CNNCritic, self).__init__()
        
        # CNN backbone (same architecture as Actor, separate params)
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        
        # Pooling
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        
        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        
        # Value output (single value)
        self.value = tf.keras.layers.Dense(1, activation=None)
        
    def call(self, inputs):
        # Ensure 4D input: [batch, height, width, channels]
        if len(inputs.shape) == 3:
            x = tf.expand_dims(inputs, axis=-1)  # Add channel dimension
        else:
            x = inputs
            
        # CNN layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Pooling layers
        x = self.pool(x)
        
        # Fully connected layers
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Output state value
        value = self.value(x)
        
        return value


class MatrixExperienceBuffer:
    """Replay buffer for matrix states."""
    
    def __init__(self, max_size=10000):
        self.states = []  # Matrix states
        self.actions = []
        self.rewards = []
        self.next_states = []  # Next matrix states
        self.dones = []
        self.log_probs = []  # Log-probabilities
        self.values = []     # Value estimates
        self.max_size = max_size
    
    def add(self, state, action, reward, next_state, done, log_prob=None, value=None):
        """Add one experience."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
        
        # If over capacity, drop oldest
        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            if len(self.log_probs) > 0:
                self.log_probs.pop(0)
            if len(self.values) > 0:
                self.values.pop(0)
    
    def get_all(self):
        """Get all stored experiences."""
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.next_states),
            np.array(self.dones),
            np.array(self.log_probs) if len(self.log_probs) > 0 else None,
            np.array(self.values) if len(self.values) > 0 else None
        )
    
    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def size(self):
        """Return current buffer size."""
        return len(self.states)


class CNNPPOAgent:
    """CNN-based PPO agent for 2D feature matrices."""
    
    def __init__(self, action_dim, lr=0.0003, gamma=0.99, 
                 clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        """Initialize CNN-PPO agent."""
        # Store hyperparameters
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = 64
        
        # Create CNN actor and critic networks
        self.actor = CNNActor(action_dim)
        self.critic = CNNCritic()
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # Initialize replay buffer
        self.buffer = MatrixExperienceBuffer()
        
        # Training metrics
        self.training_metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_rewards': []
        }
    
    def predict(self, state):
        """Predict action from the current state."""
        # Ensure state matches CNN input format
        if len(state.shape) == 2:  # 2D matrix
            # Add batch dimension
            state_tensor = tf.convert_to_tensor(
                state.reshape(1, state.shape[0], state.shape[1]), 
                dtype=tf.float32)
        else:
            # Already in proper format
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        
        # Get action distribution from actor
        action_mean, _ = self.actor(state_tensor)
        
        # Ensure non-negative weights and normalize
        actions = action_mean.numpy()[0]
        actions = np.maximum(0, actions)
        actions_sum = np.sum(actions)
        
        if actions_sum > 0:
            actions = actions / actions_sum
        else:
            # If all weights are 0, use uniform distribution
            actions = np.ones(self.action_dim) / self.action_dim
        
        return actions
    
    def select_action(self, state):
        """Select a stochastic action during training."""
        # Ensure state matches CNN input format
        if len(state.shape) == 2:  # 2D matrix
            state_tensor = tf.convert_to_tensor(
                state.reshape(1, state.shape[0], state.shape[1]), 
                dtype=tf.float32)
        else:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        
        # Get action distribution
        action_mean, action_logstd = self.actor(state_tensor)
        action_std = tf.exp(action_logstd)
        
        # Sample action from normal distribution
        normal_dist = tfp.distributions.Normal(action_mean, action_std)
        action = normal_dist.sample()
        
        # Ensure non-negative action
        action = tf.maximum(action, 0)
        
        # Normalize action
        action_sum = tf.reduce_sum(action, axis=-1, keepdims=True)
        action = tf.where(action_sum > 0, action / action_sum, 
                          tf.ones_like(action) / self.action_dim)
        
        # Compute log-probability of action
        log_prob = normal_dist.log_prob(action)
        log_prob = tf.reduce_sum(log_prob, axis=-1)
        
        # Get state value estimate
        value = self.critic(state_tensor)
        value = tf.squeeze(value)
        
        return action.numpy()[0], log_prob.numpy()[0], value.numpy()
    
    def _compute_advantages_and_returns(self, rewards, values, dones):
        """Compute advantages and returns (GAE)."""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0
        
        # Handle potential NaNs
        rewards = np.nan_to_num(rewards, nan=0.0)
        values = np.nan_to_num(values, nan=0.0)
        
        # Compute last value estimate
        if len(values) < len(rewards):
            last_value = 0
        else:
            last_value = values[-1]
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
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
        returns = advantages + values
        
        # Normalize advantages
        adv_std = np.std(advantages)
        if adv_std < 1e-10:
            adv_std = 1.0
        advantages = (advantages - np.mean(advantages)) / adv_std
        
        return advantages, returns
    
    def update(self, epochs=10):
        """Update policy and value networks."""
        # Get all experiences from buffer
        states, actions, rewards, next_states, dones, old_log_probs, values = self.buffer.get_all()
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns(rewards, values, dones)
        
        # Convert NumPy arrays to TensorFlow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Track losses
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        # Multiple epochs of updates
        for _ in range(epochs):
            # Mini-batch processing
            indices = np.random.permutation(len(states))
            batch_size = min(self.batch_size, len(states))
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = tf.gather(states, batch_indices)
                batch_actions = tf.gather(actions, batch_indices)
                batch_old_log_probs = tf.gather(old_log_probs, batch_indices)
                batch_advantages = tf.gather(advantages, batch_indices)
                batch_returns = tf.gather(returns, batch_indices)
                
                with tf.GradientTape() as tape:
                    # Compute new policy distribution
                    action_mean, action_logstd = self.actor(batch_states)
                    action_std = tf.exp(action_logstd)
                    
                    # Use normal distribution
                    dist = tfp.distributions.Normal(action_mean, action_std)
                    
                    # Compute new log-probabilities
                    new_log_probs = dist.log_prob(batch_actions)
                    new_log_probs = tf.reduce_sum(new_log_probs, axis=1)
                    
                    # Compute ratio
                    ratio = tf.exp(new_log_probs - batch_old_log_probs)
                    
                    # Clip ratio
                    clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    
                    # Compute policy loss
                    surrogate1 = ratio * batch_advantages
                    surrogate2 = clipped_ratio * batch_advantages
                    actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                    
                    # Compute entropy
                    entropy = tf.reduce_mean(dist.entropy())
                    
                    # Compute value loss
                    values_pred = self.critic(batch_states)
                    values_pred = tf.squeeze(values_pred, -1)
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - values_pred))
                    
                    # Total loss
                    total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Compute and apply gradients
                variables = self.actor.trainable_variables + self.critic.trainable_variables
                gradients = tape.gradient(total_loss, variables)
                
                # Apply gradient clipping
                gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
                self.optimizer.apply_gradients(zip(gradients, variables))
                
                # Record loss
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
                entropy_losses.append(entropy.numpy())
        
        # Compute average losses
        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        avg_entropy = np.mean(entropy_losses)
        
        # Update training metrics
        self.training_metrics['actor_loss'].append(avg_actor_loss)
        self.training_metrics['critic_loss'].append(avg_critic_loss)
        self.training_metrics['entropy'].append(avg_entropy)
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy
        }
    
    def train(self, env, epochs=100, steps_per_epoch=1000, update_epochs=10):
        """Train the agent."""
        for epoch in range(epochs):
            # Reset environment
            state = env.reset()
            
            # Ensure state is matrix-form
            if len(state.shape) == 1:  # 1D vector
                # Convert 1D vector to matrix
                state = self._reshape_to_matrix(state)
            
            epoch_rewards = []
            
            for step in range(steps_per_epoch):
                # Select action
                action, log_prob, value = self.select_action(state)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Ensure next state is matrix-form
                if len(next_state.shape) == 1:  # 1D vector
                    next_state = self._reshape_to_matrix(next_state)
                
                # Store experience
                self.buffer.add(state, action, reward, next_state, done, log_prob, value)
                
                # Update state
                state = next_state
                
                # Accumulate reward
                epoch_rewards.append(reward)
                
                # Break if episode ends
                if done:
                    break
            
            # Update networks
            if self.buffer.size() > 0:
                update_info = self.update(epochs=update_epochs)
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Total Reward: {sum(epoch_rewards):.2f}, "
                      f"Actor Loss: {update_info['actor_loss']:.4f}, "
                      f"Critic Loss: {update_info['critic_loss']:.4f}, "
                      f"Entropy: {update_info['entropy']:.4f}")
            
            # Record total reward
            self.training_metrics['total_rewards'].append(sum(epoch_rewards))
    
    def _reshape_to_matrix(self, state_vector):
        """Reshape 1D state vector to 2D matrix."""
        # Compute near-square dimensions
        dim = int(np.ceil(np.sqrt(len(state_vector))))
        
        # Create matrix, pad with zeros
        matrix = np.zeros((dim, dim), dtype=np.float32)
        
        # Fill matrix
        for i in range(min(len(state_vector), dim * dim)):
            row = i // dim
            col = i % dim
            matrix[row, col] = state_vector[i]
        
        return matrix
    
    def save(self, path):
        """Save model weights."""
        # Create directory
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save weights - ensure filename ends with .weights.h5
        actor_path = f"{path}_cnn_actor.weights.h5"
        critic_path = f"{path}_cnn_critic.weights.h5"
        
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
        print(f"CNN-PPO model saved to: {path}")
    
    def load(self, path):
        """Load model weights."""
        actor_path = f"{path}_cnn_actor.weights.h5"
        critic_path = f"{path}_cnn_critic.weights.h5"
        
        # Force weight creation
        dummy_state = np.zeros((1, 10, 10, 1), dtype=np.float32)  # Assume 10x10 matrix
        self.actor(dummy_state)  
        self.critic(dummy_state)
        
        # Load weights
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        
        print(f"CNN-PPO model loaded from: {path}") 

