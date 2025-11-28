"""
direct_cupy.py - CuPy implementation skeleton for direct summation

TODO: Implement the direct summation method using CuPy (GPU-accelerated NumPy).

The potential at position i is:
    Φ_i = -G * Σ(j≠i) m_j / |r_i - r_j|

INSTRUCTIONS:
1. Complete the compute_potential_direct function
2. Use CuPy array operations for GPU acceleration
3. Leverage broadcasting to compute all pairwise distances at once
4. Use a softening parameter to avoid singularities

USAGE:
    python direct_cupy.py --input plummer_data.npz --output potentials.npy

HINTS:
- Use cp.newaxis to add dimensions for broadcasting
- Calculate pairwise distances with broadcasting
- Use cp.fill_diagonal() to handle self-interactions
- Result should be a 1D array of potentials, one per particle
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


def compute_potential_direct(positions, masses, G=1.0, softening=0.01, chunk_size=2**15):
    """
    Calculate gravitational potential using direct summation on GPU.

    Uses batching/chunking to avoid out-of-memory errors for large N.

    Parameters:
    -----------
    positions : cupy.ndarray or numpy.ndarray
        Array of shape (n_bodies, 3) containing x, y, z positions
    masses : cupy.ndarray or numpy.ndarray
        Array of shape (n_bodies,) containing masses
    G : float
        Gravitational constant (default: 1.0)
    softening : float
        Softening parameter to avoid singularities (default: 0.01)
    chunk_size : int
        Number of particles to process at a time (default: 1024)

    Returns:
    --------
    potentials : cupy.ndarray
        Array of shape (n_bodies,) containing gravitational potentials
    """
    # Convert to CuPy arrays if needed
    if not isinstance(positions, cp.ndarray):
        positions = cp.asarray(positions)
    if not isinstance(masses, cp.ndarray):
        masses = cp.asarray(masses)

    n_bodies = positions.shape[0]
    softening_sq = softening ** 2

    # Initialize potentials array
    potentials = cp.zeros(n_bodies, dtype=positions.dtype)

    # Process in chunks to avoid OOM
    for i_start in range(0, n_bodies, chunk_size):
        i_end = min(i_start + chunk_size, n_bodies)
        
        # Get positions for current chunk of particles (targets)
        pos_i = positions[i_start:i_end]  # Shape: (chunk_i, 3)
        # Initialize potential contributions for this chunk
        chunk_potentials = cp.zeros(i_end - i_start, dtype=positions.dtype)

        # Compute interactions with all other particles in chunks
        for j_start in range(0, n_bodies, chunk_size):
            j_end = min(j_start + chunk_size, n_bodies)

            # Get positions and masses for source particles
            pos_j = positions[j_start:j_end]  # Shape: (chunk_j, 3)
            mass_j = masses[j_start:j_end]    # Shape: (chunk_j,)

            dx = pos_i[:, None, 0] - pos_j[None, :, 0]
            dy = pos_i[:, None, 1] - pos_j[None, :, 1]
            dz = pos_i[:, None, 2] - pos_j[None, :, 2]

            r = cp.sqrt(dx*dx + dy*dy + dz*dz + softening_sq)

            # Calculate potential contributions
            # Shape: (chunk_i, chunk_j)
            potential_matrix = -G * mass_j / r
            # Handle self-interactions (when i and j chunks overlap)
            if i_start == j_start:
                cp.fill_diagonal(potential_matrix, 0)
            # Sum contributions from this j-chunk
            chunk_potentials += cp.sum(potential_matrix, axis=1)

        # Store results for this i-chunk
        potentials[i_start:i_end] = chunk_potentials

    return potentials


def main():
    parser = argparse.ArgumentParser(description='Direct summation with CuPy')
    parser.add_argument('--input', type=str, default='plummer_data.npz',
                        help='Input data file (default: plummer_data.npz)')
    parser.add_argument('--output', type=str, default='outputs/direct_summation/potentials_direct.npy',
                        help='Output file for potentials (default: outputs/direct_summation/potentials_direct.npy)')
    parser.add_argument('--json_output', type=str, default=None,
                        help='JSON file for metrics (default: None)')
    parser.add_argument('--G', type=float, default=1.0,
                        help='Gravitational constant (default: 1.0)')
    parser.add_argument('--softening', type=float, default=0.01,
                        help='Softening parameter (default: 0.01)')
    parser.add_argument('--chunk_size', type=int, default=2**15,
                        help='Chunk size for processing (default: 32768)')

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
        _ = compute_potential_direct(positions, masses, G=args.G,
                                     softening=args.softening, chunk_size=args.chunk_size)

    # Timed runs
    print("Computing potentials using direct summation (CuPy) - 10 timed runs...")
    times = []
    for i in range(10):
        start = time.perf_counter()
        potentials = compute_potential_direct(positions, masses, G=args.G,
                                             softening=args.softening, chunk_size=args.chunk_size)
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
        "method": "direct_summation",
        "implementation": "cupy",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n_bodies": int(n_bodies),
            "G": float(args.G),
            "softening": float(args.softening),
            "chunk_size": int(args.chunk_size)
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
    start_time = time.perf_counter() # for quick timing checks
    main()
    end_time = time.perf_counter()  # or time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
