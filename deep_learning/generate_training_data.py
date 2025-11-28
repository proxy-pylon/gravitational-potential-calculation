"""
generate_training_data.py - Generate labeled training data for MLP

This script generates multiple small N-body configurations and computes
ground truth gravitational potential U for each using CUDA direct summation.

The output is a labeled dataset suitable for training a neural network to
predict gravitational potential energy.

USAGE:
    # Generate training data
    python generate_training_data.py --n_samples 1000 --n_bodies 5000 --output training_data.npz

    # With custom parameters
    python generate_training_data.py --n_samples 500 --n_bodies 10000 --seed 42 --output train.npz

REQUIREMENTS:
    - CUDA direct summation must be compiled: nvcc -o build/direct_cuda direct_cuda.cu
    - The compiled executable should be at: ../build/direct_cuda

WORKFLOW:
    1. Generate n_samples different Plummer configurations
    2. For each configuration, compute ground truth U using CUDA
    3. Save all configurations with their labels to npz file
"""

import numpy as np
import argparse
import subprocess
import os
import sys
from pathlib import Path

# Import Plummer generation from parent directory
sys.path.append(str(Path(__file__).parent.parent))
from data_gen import generate_plummer_positions


def generate_single_configuration(n_bodies, scale_radius=1.0, seed=None):
    """
    Generate a single N-body configuration.

    Parameters:
    -----------
    n_bodies : int
        Number of bodies in the configuration
    scale_radius : float
        Scale radius of the Plummer model
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    positions : ndarray (n_bodies, 3)
        Particle positions
    masses : ndarray (n_bodies,)
        Particle masses (uniform, sum to 1.0)
    """
    positions, masses = generate_plummer_positions(n_bodies, scale_radius, seed)
    return positions, masses


def save_binary_for_cuda(filename, positions):
    """
    Save positions in binary format for CUDA program.

    Format: Interleaved positions (x1,y1,z1,x2,y2,z2,...)
    """
    n_bodies = positions.shape[0]
    # Interleave positions
    interleaved = positions.flatten().astype(np.float32)

    with open(filename, 'wb') as f:
        interleaved.tofile(f)


def compute_ground_truth_cuda(positions, masses, cuda_executable="../build/direct_cuda"):
    """
    Compute ground truth gravitational potential U using CUDA direct summation.

    Parameters:
    -----------
    positions : ndarray (n_bodies, 3)
        Particle positions
    masses : ndarray (n_bodies,)
        Particle masses
    cuda_executable : str
        Path to compiled CUDA executable

    Returns:
    --------
    U : float
        Total gravitational potential energy
    """
    # Create temporary files
    temp_input = "/tmp/temp_plummer.bin"
    temp_output = "/tmp/temp_potentials.bin"

    # Save positions to binary file
    save_binary_for_cuda(temp_input, positions)

    # Run CUDA program
    try:
        result = subprocess.run(
            [cuda_executable, "--input", temp_input, "--output", temp_output, "--G", "1.0"],
            capture_output=True,
            text=True,
            check=True
        )

        # Read output potentials
        potentials = np.fromfile(temp_output, dtype=np.float32)

        # Calculate total U = 0.5 * sum(m_i * phi_i)
        U = 0.5 * np.sum(masses * potentials)

        # Clean up temp files
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(temp_output):
            os.remove(temp_output)

        return U

    except subprocess.CalledProcessError as e:
        print(f"Error running CUDA program: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


def generate_training_dataset(n_samples, n_bodies, scale_radius=1.0, base_seed=42,
                              cuda_executable="../build/direct_cuda"):
    """
    Generate a complete training dataset with ground truth labels.

    Parameters:
    -----------
    n_samples : int
        Number of different configurations to generate
    n_bodies : int
        Number of bodies per configuration
    scale_radius : float
        Scale radius for Plummer model
    base_seed : int
        Base random seed (each sample gets base_seed + i)
    cuda_executable : str
        Path to compiled CUDA executable

    Returns:
    --------
    all_positions : ndarray (n_samples, n_bodies, 3)
        All particle positions
    all_masses : ndarray (n_samples, n_bodies)
        All particle masses
    all_U : ndarray (n_samples,)
        Ground truth U values for each configuration
    """
    all_positions = []
    all_masses = []
    all_U = []

    print(f"\nGenerating {n_samples} training samples with {n_bodies} bodies each...")
    print(f"Using CUDA executable: {cuda_executable}")
    print(f"Base seed: {base_seed}")
    print(f"Scale radius: {scale_radius}")
    print("-" * 60)

    for i in range(n_samples):
        seed = base_seed + i

        # Generate configuration
        positions, masses = generate_single_configuration(n_bodies, scale_radius, seed)

        # Compute ground truth U using CUDA
        U = compute_ground_truth_cuda(positions, masses, cuda_executable)

        # Store results
        all_positions.append(positions)
        all_masses.append(masses)
        all_U.append(U)

        # Progress report
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Sample {i+1}/{n_samples}: N={n_bodies}, U={U:.6f}")

    # Convert to arrays
    all_positions = np.array(all_positions, dtype=np.float32)
    all_masses = np.array(all_masses, dtype=np.float32)
    all_U = np.array(all_U, dtype=np.float32)

    print("-" * 60)
    print(f"\nDataset generation complete!")
    print(f"  Total samples: {n_samples}")
    print(f"  Bodies per sample: {n_bodies}")
    print(f"  U statistics:")
    print(f"    Mean: {all_U.mean():.6f}")
    print(f"    Std:  {all_U.std():.6f}")
    print(f"    Min:  {all_U.min():.6f}")
    print(f"    Max:  {all_U.max():.6f}")
    print(f"  Expected analytical U: -0.294524")

    return all_positions, all_masses, all_U


def main():
    parser = argparse.ArgumentParser(description='Generate labeled training data for MLP')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of training samples to generate (default: 1000)')
    parser.add_argument('--n_bodies', type=int, default=5000,
                        help='Number of bodies per sample (default: 5000)')
    parser.add_argument('--scale_radius', type=float, default=1.0,
                        help='Scale radius of Plummer model (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--output', type=str, default='../outputs/deep_learning/training_data.npz',
                        help='Output npz file (default: ../outputs/deep_learning/training_data.npz)')
    parser.add_argument('--cuda_executable', type=str, default='../build/direct_cuda',
                        help='Path to compiled CUDA executable (default: ../build/direct_cuda)')

    args = parser.parse_args()

    # Check if CUDA executable exists
    if not os.path.exists(args.cuda_executable):
        print(f"ERROR: CUDA executable not found at: {args.cuda_executable}")
        print("\nPlease compile the CUDA program first:")
        print("  cd direct_summation")
        print("  nvcc -o ../build/direct_cuda direct_cuda.cu")
        sys.exit(1)

    # Generate dataset
    positions, masses, U_values = generate_training_dataset(
        n_samples=args.n_samples,
        n_bodies=args.n_bodies,
        scale_radius=args.scale_radius,
        base_seed=args.seed,
        cuda_executable=args.cuda_executable
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Save to npz file
    print(f"\nSaving dataset to {args.output}...")
    np.savez_compressed(
        args.output,
        positions=positions,
        masses=masses,
        U=U_values,
        n_samples=args.n_samples,
        n_bodies=args.n_bodies,
        scale_radius=args.scale_radius,
        seed=args.seed
    )

    print(f"Dataset saved successfully!")
    print(f"\nFile contents:")
    print(f"  positions: {positions.shape} (n_samples, n_bodies, 3)")
    print(f"  masses:    {masses.shape} (n_samples, n_bodies)")
    print(f"  U:         {U_values.shape} (n_samples,)")
    print(f"\nYou can now train the MLP with:")
    print(f"  python mlp_pytorch.py --train --input {args.output} --epochs 100")


if __name__ == "__main__":
    main()
