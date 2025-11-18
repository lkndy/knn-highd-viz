#!/usr/bin/env python3
"""Generate MNIST test data for HKNN embedding."""

import numpy as np
import struct
import argparse
from sklearn.datasets import fetch_openml


def write_fvecs(filename, data):
    """Write data in .fvecs format (FAISS convention)."""
    with open(filename, 'wb') as f:
        for vec in data:
            d = len(vec)
            f.write(struct.pack('i', d))
            f.write(struct.pack('f' * d, *vec))


def write_raw_float32(filename, data):
    """Write data as raw float32 (row-major)."""
    data.astype(np.float32).tofile(filename)


def load_mnist(n_samples=10000, random_state=42):
    """Load MNIST dataset from OpenML."""
    print("Loading MNIST dataset from OpenML...")
    mnist = fetch_openml(name='mnist_784', version=1, as_frame=False, parser='liac-arff')
    
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    # Shuffle and take subset
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    X = X[indices[:n_samples]]
    y = y[indices[:n_samples]]
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} dimensions")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description='Generate MNIST test data for HKNN embedding')
    parser.add_argument('--N', type=int, default=10000, help='Number of samples (default: 10000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='mnist_data', help='Output base name')
    
    args = parser.parse_args()
    
    # Load MNIST
    data, labels = load_mnist(n_samples=args.N, random_state=args.seed)
    
    # Write data files
    write_fvecs(f'{args.output}.fvecs', data)
    write_raw_float32(f'{args.output}.f32', data)
    
    # Save ground truth labels for visualization
    np.save(f'{args.output}_colors.npy', labels)
    
    print(f"\nGenerated {len(data)} points in {data.shape[1]} dimensions")
    print(f"Saved as {args.output}.fvecs and {args.output}.f32")
    print(f"Ground truth labels saved as {args.output}_colors.npy")


if __name__ == '__main__':
    main()
