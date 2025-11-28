"""
data_gen.py - Plummer Model Data Generation

Generates N-body particle positions and masses following the Plummer distribution.
The Plummer model is a density distribution commonly used in stellar dynamics:

    ρ(r) = (3M / 4πa³) * (1 + r²/a²)^(-5/2)

where M is total mass and a is the scale radius.

Usage:
    python data_gen.py --n_bodies 1000 --scale_radius 1.0 --output data.npz
"""

import numpy as np
import argparse


def generate_plummer_positions(n_bodies, scale_radius=1.0, seed=None):
    """
    Generate positions for N bodies following a Plummer distribution.
    
    Parameters:
    -----------
    n_bodies : int
        Number of bodies to generate
    scale_radius : float
        Scale radius of the Plummer model (default: 1.0)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    positions : ndarray
        Array of shape (n_bodies, 3) containing x, y, z positions
    masses : ndarray
        Array of shape (n_bodies,) containing masses (uniform for simplicity)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate radii using inverse CDF method for Plummer distribution
    # For Plummer: M(r) = r³ / (r² + a²)^(3/2)
    # Inverse: r = a / sqrt(u^(-2/3) - 1) where u ~ U(0,1)
    u = np.random.uniform(0, 1, n_bodies)
    radii = scale_radius / np.sqrt(u**(-2/3) - 1)
    
    # Generate random directions (uniform on sphere)
    theta = np.random.uniform(0, 2 * np.pi, n_bodies)
    phi = np.arccos(2 * np.random.uniform(0, 1, n_bodies) - 1)
    
    # Convert to Cartesian coordinates
    x = radii * np.sin(phi) * np.cos(theta)
    y = radii * np.sin(phi) * np.sin(theta)
    z = radii * np.cos(phi)
    
    positions = np.column_stack([x, y, z]).astype(np.float32)
    
    # Uniform masses (can be modified for more realistic distributions)
    masses = np.ones(n_bodies, dtype=np.float32) / n_bodies
    
    return positions, masses

def save_binary(filename, positions, masses):
    """Save data in raw binary format for C/Python compatibility.
    
    Format: [n_bodies:int32][x:float32*n][y:float32*n][z:float32*n][masses:float32*n]
    """
    n_bodies = len(masses)
    with open(filename, 'wb') as f:
        np.array([n_bodies], dtype=np.int32).tofile(f)
        positions[:, 0].astype(np.float32).tofile(f)
        positions[:, 1].astype(np.float32).tofile(f)
        positions[:, 2].astype(np.float32).tofile(f)
        masses.astype(np.float32).tofile(f)
    print(f"Binary data saved to {filename} ({n_bodies} bodies)")

def main():
    parser = argparse.ArgumentParser(description='Generate Plummer model N-body data')
    parser.add_argument('--n_bodies', type=int, default=1000,
                        help='Number of bodies to generate (default: 1000)')
    parser.add_argument('--scale_radius', type=float, default=1.0,
                        help='Scale radius of Plummer model (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='plummer_data.npz',
                        help='Output file name (default: plummer_data.npz)')
    
    args = parser.parse_args()
    
    print(f"Generating Plummer distribution with {args.n_bodies} bodies...")
    print(f"  Scale radius: {args.scale_radius}")
    print(f"  Random seed: {args.seed}")
    
    positions, masses = generate_plummer_positions(
        args.n_bodies, 
        scale_radius=args.scale_radius, 
        seed=args.seed
    )
    
    # Save to file
    print(f"Saving npz data to {args.output}...")
    np.savez(args.output, positions=positions, masses=masses)
    print(f"Data saved to {args.output}")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Masses shape: {masses.shape}")
    print(f"  Position range: [{positions.min():.3f}, {positions.max():.3f}]")

    # Save binary file with same name as npz but .bin extension
    bin_output = args.output.replace('.npz', '.bin')
    print(f"Saving binary data to {bin_output}...")
    save_binary(bin_output, positions, masses)
    print(f"Binary data saved to {bin_output}")

if __name__ == "__main__":
    main()
