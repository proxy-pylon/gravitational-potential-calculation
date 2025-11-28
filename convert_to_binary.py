#!/usr/bin/env python3
"""
Convert .npz data to binary format for CUDA C program.

Usage:
    python convert_to_binary.py --input plummer_data.npz --output plummer_data.bin
"""

import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert .npz to binary format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .npz file')
    parser.add_argument('--output', type=str, default='plummer_data.bin',
                        help='Output binary file for positions (default: plummer_data.bin)')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    data = np.load(args.input)
    positions = data['positions'].astype(np.float32)

    print(f"  Number of bodies: {len(positions)}")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Note: Masses will be computed as equal masses (m_i = 1/N) in CUDA code")

    # Save positions (interleaved x, y, z)
    positions.tofile(args.output)
    print(f"\nPositions saved to {args.output}")
    print(f"  File size: {positions.nbytes} bytes")

    print("\nConversion complete!")

if __name__ == "__main__":
    main()
