"""
mc_cupy.py - CuPy implementation skeleton for Monte Carlo approximation

TODO: Implement Monte Carlo approximation using CuPy.

Instead of summing over all particles, randomly sample a subset.
The potential at position i is estimated as:
    Φ_i ≈ -(G * N / n_samples) * Σ(k=1 to n_samples) m_k / |r_i - r_k|

INSTRUCTIONS:
1. Complete the compute_potential_monte_carlo function
2. Use random sampling to select particles
3. Scale the result appropriately

USAGE:
    python mc_cupy.py --input plummer_data.npz --output potentials_mc.npy --n_samples 100

HINTS:
- Use cp.random.randint() to generate random indices
- Sample shape should be (n_bodies, n_samples)
- Use array indexing to get sampled positions and masses
- Scale by (N-1)/n_samples factor
"""

import numpy as np
import argparse
import time
import json
import os
from datetime import datetime

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("ERROR: CuPy not installed. Install with: pip install cupy-cuda11x")
    exit(1)


def compute_potential_monte_carlo(positions, masses, n_samples, G=1.0, softening=0.01, seed=None, batch_size=10000, body_batch_size=None):
    """
    Calculate gravitational potential using Monte Carlo approximation on GPU.

    Uses double-batched processing to avoid memory issues with large datasets.
    Batches both the number of samples AND the number of bodies.

    Parameters:
    -----------
    positions : cupy.ndarray or numpy.ndarray
        Array of shape (n_bodies, 3) containing x, y, z positions
    masses : cupy.ndarray or numpy.ndarray
        Array of shape (n_bodies,) containing masses
    n_samples : int
        Number of particles to sample for each potential calculation
    G : float
        Gravitational constant (default: 1.0)
    softening : float
        Softening parameter to avoid singularities (default: 0.01)
    seed : int, optional
        Random seed for reproducibility
    batch_size : int
        Number of samples to process at once (default: 10000)
    body_batch_size : int, optional
        Number of bodies to process at once (default: auto-calculated)
        Limits memory usage to O(body_batch_size * batch_size)

    Returns:
    --------
    potentials : cupy.ndarray
        Array of shape (n_bodies,) containing estimated gravitational potentials
    """

    # Convert to CuPy arrays if needed
    if not isinstance(positions, cp.ndarray):
        positions = cp.asarray(positions)
    if not isinstance(masses, cp.ndarray):
        masses = cp.asarray(masses)

    n_bodies = positions.shape[0]

    # Auto-calculate body_batch_size based on available memory
    # Target: keep memory usage under 2GB per batch
    # Memory per batch ≈ body_batch_size * batch_size * 104 bytes
    if body_batch_size is None:
        target_memory_gb = 2.0  # 2GB target
        bytes_per_element = 104
        body_batch_size = int((target_memory_gb * 1e9) / (batch_size * bytes_per_element))
        body_batch_size = max(1000, min(body_batch_size, n_bodies))  # Clamp between 1000 and n_bodies

    # Set random seed if provided
    if seed is not None:
        cp.random.seed(seed)

    # Initialize potentials array
    potentials = cp.zeros(n_bodies, dtype=cp.float64)

    # Calculate number of body batches
    n_body_batches = (n_bodies + body_batch_size - 1) // body_batch_size

    # Process bodies in batches
    for body_batch_idx in range(n_body_batches):
        # Calculate body batch range
        body_start = body_batch_idx * body_batch_size
        body_end = min(body_start + body_batch_size, n_bodies)
        current_body_batch_size = body_end - body_start

        # Get positions for this body batch
        positions_batch = positions[body_start:body_end]

        # Initialize potentials for this body batch
        potentials_batch = cp.zeros(current_body_batch_size, dtype=cp.float64)

        # Process samples in batches for this body batch
        n_sample_batches = (n_samples + batch_size - 1) // batch_size

        for sample_batch_idx in range(n_sample_batches):
            # Calculate sample batch size
            sample_start = sample_batch_idx * batch_size
            sample_end = min(sample_start + batch_size, n_samples)
            current_sample_batch_size = sample_end - sample_start

            # Generate random sample indices for this batch, excluding self-interactions
            # Use a deterministic approach: sample from range [0, n_bodies-1), then adjust indices >= body_idx
            # This avoids the slow rejection sampling loop

            # Shape: (current_body_batch_size, current_sample_batch_size)
            # Sample from [0, n_bodies-1) to exclude one position per body
            sample_indices = cp.random.randint(0, n_bodies - 1, size=(current_body_batch_size, current_sample_batch_size))

            # Create body indices for this batch (the "i" in the pair (i,j))
            body_indices = cp.arange(body_start, body_end)[:, cp.newaxis]

            # Adjust indices: if sample_index >= body_index, increment by 1
            # This effectively excludes the self-interaction without rejection sampling
            sample_indices = cp.where(sample_indices >= body_indices, sample_indices + 1, sample_indices)

            # Get sampled positions and masses
            # sampled_positions shape: (current_body_batch_size, current_sample_batch_size, 3)
            # sampled_masses shape: (current_body_batch_size, current_sample_batch_size)
            sampled_positions = positions[sample_indices]
            sampled_masses = masses[sample_indices]

            # Expand dimensions for broadcasting
            # positions_i shape: (current_body_batch_size, 1, 3)
            positions_i = positions_batch[:, cp.newaxis, :]

            # Calculate distances from each particle to its samples
            # dr shape: (current_body_batch_size, current_sample_batch_size, 3)
            dr = sampled_positions - positions_i

            # Distance with softening
            # r shape: (current_body_batch_size, current_sample_batch_size)
            r = cp.sqrt(cp.sum(dr**2, axis=2) + softening**2)

            # Calculate potential contributions
            # For each particle i, sum -G * m_j / r_ij over sampled particles j
            potential_contributions = -G * sampled_masses / r

            # Sum contributions for this batch
            potentials_batch += cp.sum(potential_contributions, axis=1)

        # Store results for this body batch
        potentials[body_start:body_end] = potentials_batch

    # Scale factor accounts for sampling only a subset of particles
    # The factor (n_bodies - 1) / n_samples scales the sample average to the full sum
    scale_factor = (n_bodies - 1) / n_samples
    potentials *= scale_factor

    return potentials


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo approximation with CuPy')
    parser.add_argument('--input', type=str, default='plummer_data.npz',
                        help='Input data file (default: plummer_data.npz)')
    parser.add_argument('--output', type=str, default='outputs/monte_carlo/potentials_mc.npy',
                        help='Output file for potentials (default: outputs/monte_carlo/potentials_mc.npy)')
    parser.add_argument('--json_output', type=str, default=None,
                        help='JSON file for metrics (default: None)')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples per particle (default: 100)')
    parser.add_argument('--G', type=float, default=1.0,
                        help='Gravitational constant (default: 1.0)')
    parser.add_argument('--softening', type=float, default=0.01,
                        help='Softening parameter (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size for processing samples (default: 10000)')
    parser.add_argument('--body_batch_size', type=int, default=None,
                        help='Batch size for processing bodies (default: auto-calculated)')

    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = np.load(args.input)
    positions = data['positions']
    masses = data['masses']
    n_bodies = len(positions)
    print(f"  Number of bodies: {n_bodies}")

    # Warmup runs
    print("Warming up (3 runs)...")
    for _ in range(3):
        _ = compute_potential_monte_carlo(
            positions, masses, args.n_samples, G=args.G,
            softening=args.softening, seed=args.seed, batch_size=args.batch_size,
            body_batch_size=args.body_batch_size
        )

    # Timed runs
    print(f"Computing potentials using Monte Carlo (n_samples={args.n_samples}) - 10 timed runs...")
    times = []
    for i in range(10):
        start = time.perf_counter()
        potentials = compute_potential_monte_carlo(
            positions, masses, args.n_samples, G=args.G,
            softening=args.softening, seed=args.seed, batch_size=args.batch_size,
            body_batch_size=args.body_batch_size
        )
        cp.cuda.Stream.null.synchronize()  # Ensure GPU completion
        end = time.perf_counter()
        times.append(end - start)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/10 runs")

    mean_time = np.mean(times)
    std_time = np.std(times)

    # Convert back to NumPy and save
    potentials_cpu = cp.asnumpy(potentials)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    np.save(args.output, potentials_cpu)

    # Calculate total Plummer potential U = (1/2) * sum(m_i * phi_i)
    U_computed = 0.5 * np.sum(masses * potentials_cpu)
    U_analytical = -0.294524
    relative_error = abs(U_computed - U_analytical) / abs(U_analytical)

    # Prepare results dictionary
    results = {
        "method": "monte_carlo",
        "implementation": "cupy",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n_bodies": int(n_bodies),
            "n_samples": int(args.n_samples),
            "G": float(args.G),
            "softening": float(args.softening),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "body_batch_size": int(args.body_batch_size) if args.body_batch_size is not None else "auto"
        },
        "timing": {
            "mean_time_seconds": float(mean_time),
            "std_time_seconds": float(std_time),
            "min_time_seconds": float(np.min(times)),
            "max_time_seconds": float(np.max(times)),
            "n_trials": len(times),
            "warmup_runs": 3
        },
        "results": {
            "U_computed": float(U_computed),
            "U_analytical": float(U_analytical),
            "relative_error": float(relative_error),
            "relative_error_percent": float(relative_error * 100)
        },
        "potential_statistics": {
            "mean": float(potentials_cpu.mean()),
            "std": float(potentials_cpu.std()),
            "min": float(potentials_cpu.min()),
            "max": float(potentials_cpu.max())
        },
        "files": {
            "input": args.input,
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

    print(f"\nResults saved to {args.output}")
    print(f"\n--- Particle Potential Statistics ---")
    print(f"  Mean potential: {potentials_cpu.mean():.6f}")
    print(f"  Std potential: {potentials_cpu.std():.6f}")
    print(f"  Min potential: {potentials_cpu.min():.6f}")
    print(f"  Max potential: {potentials_cpu.max():.6f}")
    print(f"\n--- Total Plummer Potential ---")
    print(f"  Computed U: {U_computed:.6f}")
    print(f"  Analytical U: {U_analytical:.6f}")
    print(f"  Relative error: {relative_error:.6e} ({relative_error*100:.4f}%)")
    print(f"\n--- Timing ---")
    print(f"  Mean time: {mean_time:.6f} ± {std_time:.6f} seconds")
    print(f"  Min time: {np.min(times):.6f} seconds")
    print(f"  Max time: {np.max(times):.6f} seconds")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
