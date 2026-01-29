"""
Train deep learning models for OLPS strategies.

This script trains models used in online portfolio selection (OLPS),
including the CVaR-based RL model (RLCVAR) and the differential Sharpe ratio
dynamic asset network (DNA-S).
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
from datetime import datetime

# Import strategy-related models
from strategy_comparison import ResidualBlock, PolicyNetwork, DynamicAssetNetwork, DifferentialSharpeRatio
from utils.data_loader_factory import create_data_loader
from config.config import get_config

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ModelTrainer:
    """Model trainer."""
    
    def __init__(self, config):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_avaSIMble() else 'cpu')
        
        # Create model output directory
        os.makedirs('models', exist_ok=True)
        
        # Load data
        print("Loading data...")
        self.data_loader = create_data_loader(config)
        self.data_loader.load_data()
        
        # Prepare training data
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare training data."""
        print("Preparing training data...")
        
        # Get all stock data
        self.all_stock_data = self.data_loader.all_stock_data
        
        # Determine number of assets
        self.n_assets = len(self.all_stock_data)
        print(f"Data contains {self.n_assets} stocks")
        
        # Set feature dimensions based on model requirements
        self.rlcvar_input_dim = 30 * 4  # RLCVAR window size * feature count
        self.dnas_feature_dim = 7  # DNA-S per-asset feature dimension
        self.hist_len = 60  # DNA-S historical window length
        
    def train_rlcvar_model(self, epochs=50, batch_size=64, learning_rate=0.001):
        """Train the RLCVAR model."""
        print("Starting RLCVAR training...")
        
        # Create model
        model = PolicyNetwork(input_dim=self.rlcvar_input_dim, num_assets=self.n_assets).to(self.device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Define loss (MSE as a simplification of CVaR optimization)
        criterion = nn.MSELoss()
        
        # Prepare training data (use synthetic data)
        n_samples = 1000
        window_size = 30
        
        # Generate feature matrix [n_samples, window_size, 4*n_assets]
        X = np.random.randn(n_samples, window_size * 4)
        
        # Normalize to simulate real features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Generate labels (simulate ideal portfolio weights)
        # Assume market-cap-weighted targets
        Y = np.abs(np.random.randn(n_samples, self.n_assets))
        Y = Y / Y.sum(axis=1, keepdims=True)  # Normalize to sum to 1
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        Y_tensor = torch.FloatTensor(Y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, Y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_X, batch_Y in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                
                # Backprop and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print average loss per epoch
            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save model
        torch.save(model.state_dict(), 'models/rl_cvar_model.pth')
        print("RLCVAR training complete and saved")
    
    def train_dnas_model(self, epochs=50, batch_size=64, learning_rate=0.001):
        """Train the DNA-S model."""
        print("Starting DNA-S training...")
        
        # Create a simplified DNA-S model
        class SimpleDNASModel(nn.Module):
            def __init__(self, input_size, n_assets, hidden_size=32):
                super().__init__()
                self.feature_encoder = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
                
                # Weight generator
                self.weight_generator = nn.Sequential(
                    nn.Linear(hidden_size + 1, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
                
                # Value estimator
                self.value_estimator = nn.Sequential(
                    nn.Linear(hidden_size, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1)
                )
                
            def forward(self, features, current_weights):
                """
                Args:
                    features: [batch_size, n_assets, feature_dim]
                    current_weights: [batch_size, n_assets]
                """
                batch_size, n_assets, feature_dim = features.shape
                
                # Encode features
                encoded_features = self.feature_encoder(features.view(-1, feature_dim))
                encoded_features = encoded_features.view(batch_size, n_assets, -1)
                
                # Combine with current weights
                weights_expanded = current_weights.unsqueeze(-1)
                combined = torch.cat([encoded_features, weights_expanded], dim=2)
                
                # Generate new weights
                new_weights = self.weight_generator(combined).squeeze(-1)
                new_weights = torch.softmax(new_weights, dim=1)  # Normalize weights
                
                # Estimate value
                pooled_features = torch.mean(encoded_features, dim=1)  # [batch_size, hidden_size]
                value = self.value_estimator(pooled_features)
                
                return new_weights, value
        
        # Use simplified model
        model = SimpleDNASModel(
            input_size=self.dnas_feature_dim * self.hist_len,
            n_assets=self.n_assets,
            hidden_size=32
        ).to(self.device)
        
        print("Training with simplified DNA-S model")
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Ensure all parameters require gradients
        for name, param in model.named_parameters():
            param.requires_grad = True
            print(f"Param {name}: requires_grad={param.requires_grad}, shape={param.shape}")
        
        # Define loss (multi-task: weight prediction + value prediction)
        def multi_task_loss(action_probs, target_weights, value_pred, target_value):
            # Weight prediction uses MSE
            weight_loss = nn.MSELoss()(action_probs, target_weights)
            
            # Value prediction uses MSE
            value_loss = nn.MSELoss()(value_pred, target_value)
            
            # Total loss is weighted sum of tasks
            return weight_loss + 0.5 * value_loss
        
        # Prepare training data (synthetic)
        n_samples = 1000
        hist_len = 60  # Historical window length
        
        # Use flattened feature representation
        # Each sample/asset feature is a flat vector [hist_len * feature_dim]
        feature_dim = self.dnas_feature_dim * hist_len
        
        # Generate features [n_samples, n_assets, feature_dim]
        X_features = np.random.randn(n_samples, self.n_assets, feature_dim)
        # Standardize features
        X_features = (X_features - X_features.mean(axis=(0, 1), keepdims=True)) / (X_features.std(axis=(0, 1), keepdims=True) + 1e-8)
        
        # Generate current weights [n_samples, n_assets]
        X_weights = np.random.rand(n_samples, self.n_assets)
        X_weights = X_weights / X_weights.sum(axis=1, keepdims=True)
        
        # Generate target weights and values (simulated)
        Y_weights = np.random.rand(n_samples, self.n_assets)
        Y_weights = Y_weights / Y_weights.sum(axis=1, keepdims=True)
        Y_values = np.random.randn(n_samples, 1) * 0.1 + 1.0  # Values around 1.0
        
        # Convert to PyTorch tensors, matching model dtype
        X_features_tensor = torch.FloatTensor(X_features).to(self.device)
        X_weights_tensor = torch.FloatTensor(X_weights).to(self.device)
        Y_weights_tensor = torch.FloatTensor(Y_weights).to(self.device)
        Y_values_tensor = torch.FloatTensor(Y_values).to(self.device)
        
        # Print shapes to verify dimensions
        print(f"Feature tensor shape: {X_features_tensor.shape}")
        print(f"Weight tensor shape: {X_weights_tensor.shape}")
        print(f"Target weight shape: {Y_weights_tensor.shape}")
        print(f"Target value shape: {Y_values_tensor.shape}")
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_features_tensor, X_weights_tensor, Y_weights_tensor, Y_values_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Validate dimensions for the first batch
        for batch_X_features, batch_X_weights, batch_Y_weights, batch_Y_values in data_loader:
            print(f"Batch feature shape: {batch_X_features.shape}")
            print(f"Batch weight shape: {batch_X_weights.shape}")
            
            # Validate a forward pass
            with torch.no_grad():
                try:
                    output_weights, output_values = model(batch_X_features, batch_X_weights)
                    print(f"Output weight shape: {output_weights.shape}")
                    print(f"Output value shape: {output_values.shape}")
                    print("Forward pass test succeeded!")
                except Exception as e:
                    print(f"Forward pass test failed: {e}")
            break
        
        # Training loop
        for epoch in range(epochs):
            model.train()  # Ensure training mode
            total_loss = 0
            
            for batch_X_features, batch_X_weights, batch_Y_weights, batch_Y_values in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Clear previous gradients
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    action_probs, values = model(batch_X_features, batch_X_weights)
                    
                    # Compute multi-task loss
                    loss = multi_task_loss(action_probs, batch_Y_weights, values, batch_Y_values)
                    
                    # Backprop
                    loss.backward()
                    
                    # Optimizer step
                    optimizer.step()
                    
                    total_loss += loss.item()
                except Exception as e:
                    print(f"Error during training: {str(e)}")
                    # Print batch tensor shapes
                    print(f"batch_X_features.shape = {batch_X_features.shape}")
                    print(f"batch_X_weights.shape = {batch_X_weights.shape}")
                    raise e
            
            # Print average loss per epoch
            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save model
        torch.save(model.state_dict(), 'models/dna_s_model.pth')
        print("DNA-S training complete and saved")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train deep learning models for OLPS strategies')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Config file path')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_rlcvar', action='store_true',
                        help='Train RLCVAR model')
    parser.add_argument('--train_dnas', action='store_true',
                        help='Train DNA-S model')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        # Try adding LIMPPO_CNN prefix
        alt_config_path = os.path.join('LIMPPO_CNN', config_path)
        if os.path.exists(alt_config_path):
            config_path = alt_config_path
            print(f"Using config file: {config_path}")
        else:
            print(f"Config file not found: {config_path} or {alt_config_path}")
            raise FileNotFoundError(f"Config not found: {config_path} or {alt_config_path}")
    
    config = get_config(config_path)
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    # Train all models if none specified
    train_all = not (args.train_rlcvar or args.train_dnas)
    
    # Train models
    if args.train_rlcvar or train_all:
        trainer.train_rlcvar_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    
    if args.train_dnas or train_all:
        trainer.train_dnas_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    
    print("All model training complete!")

if __name__ == "__main__":
    main() 

