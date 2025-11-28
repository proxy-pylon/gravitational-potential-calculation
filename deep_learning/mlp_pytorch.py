"""
mlp_pytorch.py - PyTorch implementation for MLP gravitational potential prediction

This implements a Multi-Layer Perceptron to predict gravitational potential energy U
from N-body particle configurations using aggregated features for generalization
across different N values.

USAGE:
    # Train the model
    python mlp_pytorch.py --train --input ../outputs/deep_learning/training_data.npz --epochs 100

    # Make predictions
    python mlp_pytorch.py --predict --input ../plummer_data.npz --model ../outputs/deep_learning/model.pt

FEATURES:
- Permutation-invariant feature extraction (works with any N)
- 80/20 train/validation split
- Early stopping to prevent overfitting
- Feature normalization for stable training
- CUDA acceleration
"""

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
from datetime import datetime


def extract_features(positions, masses, n_distance_samples=10000, seed=None):
    """
    Extract permutation-invariant aggregated features from N-body configuration.

    This creates a fixed-size feature vector regardless of N, allowing the model
    to generalize to different numbers of particles.

    Parameters:
    -----------
    positions : ndarray (n_bodies, 3)
        Particle positions
    masses : ndarray (n_bodies,)
        Particle masses
    n_distance_samples : int
        Number of pairwise distances to sample (default: 10000)
    seed : int, optional
        Random seed for distance sampling

    Returns:
    --------
    features : ndarray (21,)
        Fixed-size feature vector containing:
        - Spatial statistics (11 features)
        - Pairwise distance statistics (5 features)
        - Mass distribution (2 features)
        - System properties (3 features)
    """
    if seed is not None:
        np.random.seed(seed)

    n_bodies = positions.shape[0]
    features = []

    # 1. Spatial statistics (11 features)
    # Mean position (center of mass for equal masses)
    features.extend(positions.mean(axis=0))  # 3 features: mean_x, mean_y, mean_z

    # Position standard deviation (spread)
    features.extend(positions.std(axis=0))   # 3 features: std_x, std_y, std_z

    # Position range
    features.extend(positions.max(axis=0) - positions.min(axis=0))  # 3 features

    # Radial statistics
    radii = np.sqrt((positions**2).sum(axis=1))
    features.append(radii.mean())  # 1 feature: mean radius from origin
    features.append(radii.std())   # 1 feature: std radius

    # 2. Pairwise distance statistics (5 features)
    # Sample distances to avoid O(N²) memory usage
    n_sample = min(n_distance_samples, n_bodies * (n_bodies - 1) // 2)

    if n_bodies > 1:
        idx1 = np.random.randint(0, n_bodies, n_sample)
        idx2 = np.random.randint(0, n_bodies, n_sample)
        # Ensure no self-pairs
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        if len(idx1) > 0:
            dists = np.sqrt(((positions[idx1] - positions[idx2])**2).sum(axis=1))
            features.append(dists.mean())          # 1 feature: mean distance
            features.append(dists.std())           # 1 feature: std distance
            features.append(dists.min())           # 1 feature: min distance
            features.append(np.median(dists))      # 1 feature: median distance
            features.append(np.percentile(dists, 75))  # 1 feature: 75th percentile
        else:
            features.extend([0.0] * 5)
    else:
        features.extend([0.0] * 5)

    # 3. Mass distribution (2 features)
    features.append(masses.sum())  # 1 feature: total mass (should be ~1.0)
    features.append(masses.std())  # 1 feature: mass variance

    # 4. System properties (3 features)
    features.append(float(n_bodies))  # 1 feature: number of bodies

    # Moment of inertia (related to virial theorem)
    I = (masses * radii**2).sum()
    features.append(I)  # 1 feature: moment of inertia

    # Core density proxy (fraction of particles within scale radius)
    core_count = (radii < 1.0).sum() / n_bodies
    features.append(core_count)  # 1 feature: normalized core density

    return np.array(features, dtype=np.float32)


class GravitationalPotentialMLP(nn.Module):
    """
    Multi-Layer Perceptron for predicting gravitational potential energy U.

    Architecture:
    - Input: Fixed-size feature vector (21 features by default)
    - Hidden layers: [128, 256, 128, 64] with ReLU activation and dropout
    - Output: Single scalar U value
    """

    def __init__(self, input_dim=21, hidden_dims=[128, 256, 128, 64], dropout_rate=0.1):
        """
        Initialize the MLP.

        Parameters:
        -----------
        input_dim : int
            Number of input features (default: 21)
        hidden_dims : list
            Sizes of hidden layers (default: [128, 256, 128, 64])
        dropout_rate : float
            Dropout probability for regularization (default: 0.1)
        """
        super(GravitationalPotentialMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer (single scalar)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor (batch_size, input_dim)
            Input features

        Returns:
        --------
        output : torch.Tensor (batch_size,)
            Predicted U values
        """
        return self.network(x).squeeze(-1)


class NBodyDataset(Dataset):
    """
    Dataset for N-body gravitational potential prediction.

    Converts raw positions/masses to aggregated features and handles normalization.
    """

    def __init__(self, positions, masses, U_values, feature_mean=None, feature_std=None):
        """
        Initialize dataset.

        Parameters:
        -----------
        positions : ndarray (n_samples, n_bodies, 3) or (n_bodies, 3)
            Particle positions
        masses : ndarray (n_samples, n_bodies) or (n_bodies,)
            Particle masses
        U_values : ndarray (n_samples,) or float
            Ground truth U values
        feature_mean : ndarray, optional
            Pre-computed feature means for normalization
        feature_std : ndarray, optional
            Pre-computed feature stds for normalization
        """
        # Handle single configuration case
        if positions.ndim == 2:
            positions = positions[np.newaxis, ...]
            masses = masses[np.newaxis, ...]
            U_values = np.array([U_values])

        self.n_samples = positions.shape[0]

        # Extract features for all samples
        print(f"Extracting features from {self.n_samples} samples...")
        features_list = []
        for i in range(self.n_samples):
            feat = extract_features(positions[i], masses[i])
            features_list.append(feat)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{self.n_samples} samples")

        features = np.array(features_list, dtype=np.float32)

        # Normalize features
        if feature_mean is None or feature_std is None:
            self.feature_mean = features.mean(axis=0)
            self.feature_std = features.std(axis=0) + 1e-8  # Avoid division by zero
        else:
            self.feature_mean = feature_mean
            self.feature_std = feature_std

        features_normalized = (features - self.feature_mean) / self.feature_std

        # Convert to tensors
        self.features = torch.FloatTensor(features_normalized)
        self.U_values = torch.FloatTensor(U_values)

        print(f"Feature extraction complete. Feature shape: {self.features.shape}")

    def __len__(self):
        """Return the size of the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """Get a single sample."""
        return self.features[idx], self.U_values[idx]


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda',
                early_stop_threshold=1e-4, patience=10):
    """
    Train the gravitational potential MLP.

    Parameters:
    -----------
    model : GravitationalPotentialMLP
        The model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    epochs : int
        Maximum number of epochs (default: 100)
    lr : float
        Learning rate (default: 0.001)
    device : str
        Device to use ('cuda' or 'cpu')
    early_stop_threshold : float
        Stop if validation loss drops below this (default: 1e-4)
    patience : int
        Number of epochs to wait for improvement before stopping (default: 10)

    Returns:
    --------
    model : GravitationalPotentialMLP
        Trained model
    history : dict
        Training history (train_loss, val_loss)
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"\nTraining on {device}...")
    print(f"Epochs: {epochs}, Learning rate: {lr}")
    print(f"Early stopping threshold: {early_stop_threshold}")
    print("-" * 60)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * features.size(0)

        val_loss /= len(val_loader.dataset)

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.6e}, '
                  f'Val Loss: {val_loss:.6e}')

        # Early stopping check
        if val_loss < early_stop_threshold:
            print(f"\nEarly stopping: validation loss {val_loss:.6e} < {early_stop_threshold}")
            break

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs")
                break

    print("-" * 60)
    print(f"Training complete. Best validation loss: {best_val_loss:.6e}")

    return model, history


def predict_potential(model, positions, masses, feature_mean, feature_std,
                     device='cuda', batch_size=1024):
    """
    Predict gravitational potential U using trained model.

    Parameters:
    -----------
    model : GravitationalPotentialMLP
        Trained model
    positions : ndarray (n_bodies, 3) or (n_samples, n_bodies, 3)
        Particle positions
    masses : ndarray (n_bodies,) or (n_samples, n_bodies)
        Particle masses
    feature_mean : ndarray
        Feature normalization means
    feature_std : ndarray
        Feature normalization stds
    device : str
        Device to use ('cuda' or 'cpu')
    batch_size : int
        Batch size for inference (default: 1024)

    Returns:
    --------
    predictions : ndarray
        Predicted U values
    """
    model = model.to(device)
    model.eval()

    # Create dataset with dummy U values (not used for prediction)
    dummy_U = np.zeros(positions.shape[0] if positions.ndim == 3 else 1)
    dataset = NBodyDataset(positions, masses, dummy_U, feature_mean, feature_std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Make predictions
    predictions = []
    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            outputs = model(features)
            predictions.append(outputs.cpu().numpy())

    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser(description='MLP for gravitational potential')
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--predict', action='store_true', help='Prediction mode')
    parser.add_argument('--input', type=str, required=True, help='Input data file (.npz)')
    parser.add_argument('--model', type=str, default='../outputs/deep_learning/model.pt',
                        help='Model file path (default: ../outputs/deep_learning/model.pt)')
    parser.add_argument('--output', type=str, default='../outputs/deep_learning/predictions.npy',
                        help='Output file for predictions (default: ../outputs/deep_learning/predictions.npy)')
    parser.add_argument('--json_output', type=str, default=None,
                        help='JSON file for metrics (default: None)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.train:
        print("\n" + "="*60)
        print("TRAINING MODE")
        print("="*60)

        # Load training data
        print(f"\nLoading training data from {args.input}...")
        data = np.load(args.input)
        positions = data['positions']
        masses = data['masses']
        U_values = data['U']

        print(f"  Loaded {len(positions)} samples")
        print(f"  Positions shape: {positions.shape}")
        print(f"  U range: [{U_values.min():.6f}, {U_values.max():.6f}]")

        # Create train/validation split
        n_samples = len(positions)
        n_val = int(n_samples * args.val_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        print(f"\nSplitting data: {n_train} train, {n_val} validation")

        # Create datasets
        train_dataset = NBodyDataset(
            positions[train_idx],
            masses[train_idx],
            U_values[train_idx]
        )
        val_dataset = NBodyDataset(
            positions[val_idx],
            masses[val_idx],
            U_values[val_idx],
            feature_mean=train_dataset.feature_mean,
            feature_std=train_dataset.feature_std
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Create model
        input_dim = train_dataset.features.shape[1]
        print(f"\nInitializing MLP with input dimension: {input_dim}")
        model = GravitationalPotentialMLP(input_dim=input_dim)

        # Train model
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device
        )

        # Save model and normalization parameters
        model_dir = os.path.dirname(args.model)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_mean': train_dataset.feature_mean,
            'feature_std': train_dataset.feature_std,
            'input_dim': input_dim,
            'history': history
        }, args.model)

        print(f"\nModel saved to {args.model}")

        # Compute final metrics
        model.eval()
        with torch.no_grad():
            train_features = train_dataset.features.to(device)
            train_targets = train_dataset.U_values.to(device)
            train_preds = model(train_features).cpu().numpy()
            train_true = train_targets.cpu().numpy()

            val_features = val_dataset.features.to(device)
            val_targets = val_dataset.U_values.to(device)
            val_preds = model(val_features).cpu().numpy()
            val_true = val_targets.cpu().numpy()

        train_mse = np.mean((train_preds - train_true)**2)
        val_mse = np.mean((val_preds - val_true)**2)

        # Analytical value for Plummer sphere
        U_analytical = -0.294524
        train_rel_error = np.abs((train_preds - U_analytical) / U_analytical).mean()
        val_rel_error = np.abs((val_preds - U_analytical) / U_analytical).mean()

        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        print(f"Train MSE: {train_mse:.6e}")
        print(f"Val MSE:   {val_mse:.6e}")
        print(f"Train relative error (vs analytical): {train_rel_error:.4%}")
        print(f"Val relative error (vs analytical):   {val_rel_error:.4%}")
        print(f"Analytical U: {U_analytical:.6f}")
        print(f"Predicted U mean (val): {val_preds.mean():.6f}")

    elif args.predict:
        print("\n" + "="*60)
        print("PREDICTION MODE")
        print("="*60)

        # Load model
        print(f"\nLoading model from {args.model}...")
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)

        input_dim = checkpoint['input_dim']
        feature_mean = checkpoint['feature_mean']
        feature_std = checkpoint['feature_std']

        model = GravitationalPotentialMLP(input_dim=input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        print("Model loaded successfully")

        # Load data
        print(f"\nLoading data from {args.input}...")
        data = np.load(args.input)
        positions = data['positions']
        masses = data['masses']

        print(f"  Loaded data with shape: {positions.shape}")

        # Make predictions with timing
        print("\nMaking predictions...")
        start_time = time.perf_counter()
        predictions = predict_potential(
            model, positions, masses,
            feature_mean, feature_std,
            device=device
        )
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        # Save predictions
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        np.save(args.output, predictions)

        # Calculate metrics
        U_analytical = -0.294524
        n_predictions = len(predictions)

        if n_predictions == 1:
            U_computed = float(predictions[0])
            rel_error = abs((U_computed - U_analytical) / U_analytical)

            # Prepare JSON results
            results = {
                "method": "deep_learning",
                "implementation": "pytorch",
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "n_bodies": int(positions.shape[0] if positions.ndim == 2 else positions.shape[1]),
                    "model_file": args.model,
                    "input_dim": int(input_dim)
                },
                "timing": {
                    "inference_time_seconds": float(inference_time),
                    "n_predictions": int(n_predictions)
                },
                "results": {
                    "U_computed": float(U_computed),
                    "U_analytical": float(U_analytical),
                    "relative_error": float(rel_error),
                    "relative_error_percent": float(rel_error * 100)
                },
                "files": {
                    "input": args.input,
                    "model": args.model,
                    "output": args.output
                }
            }

            # Save JSON if requested
            if args.json_output:
                json_dir = os.path.dirname(args.json_output)
                if json_dir and not os.path.exists(json_dir):
                    os.makedirs(json_dir, exist_ok=True)
                with open(args.json_output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nJSON results saved to {args.json_output}")

            print(f"\nPredictions saved to {args.output}")
            print(f"  Number of predictions: {n_predictions}")
            print(f"  Predicted U: {U_computed:.6f}")
            print(f"  Analytical U (reference): {U_analytical:.6f}")
            print(f"  Relative error: {rel_error:.6e} ({rel_error*100:.4f}%)")
            print(f"  Inference time: {inference_time:.6f} seconds")
        else:
            print(f"\nPredictions saved to {args.output}")
            print(f"  Number of predictions: {n_predictions}")
            print(f"  Predicted U range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            print(f"  Mean: {predictions.mean():.6f}")
            print(f"  Inference time: {inference_time:.6f} seconds")

    else:
        print("Please specify --train or --predict")
        parser.print_help()


if __name__ == "__main__":
    main()
