#!/usr/bin/env python3
"""Visualize HKNN embedding output."""

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

# Try to import sklearn for clustering, but make it optional
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def load_embedding(csv_path):
    """Load embedding from CSV file."""
    ids = []
    coords = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row['id']))
            if 'z' in row:  # 3D
                coords.append([float(row['x']), float(row['y']), float(row['z'])])
            else:  # 2D
                coords.append([float(row['x']), float(row['y'])])
    
    return np.array(ids), np.array(coords)

def load_ground_truth(base_path):
    """Load ground truth colors/labels if available."""
    try:
        colors = np.load(f'{base_path}_colors.npy')
        return colors
    except FileNotFoundError:
        return None

def visualize_2d(ids, coords, output_path=None, ground_truth=None):
    """Visualize 2D embedding."""
    if ground_truth is not None:
        # Three plots: ID, Ground Truth, Clusters
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
        
        # Left plot: Color by ID (sequential)
        scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], 
                              c=ids, cmap='viridis', 
                              alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_title('HKNN Embedding - Colored by Point ID', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Point ID', fontsize=10)
        
        # Middle plot: Color by ground truth
        if ground_truth.dtype == np.float64 or ground_truth.dtype == np.float32:
            # Continuous color (e.g., Swiss roll parameter)
            scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], 
                                  c=ground_truth, cmap='plasma', 
                                  alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
            ax2.set_title('HKNN Embedding - Ground Truth Colors', fontsize=14, fontweight='bold')
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('Ground Truth', fontsize=10)
        else:
            # Discrete labels (e.g., circles, clusters)
            scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], 
                                  c=ground_truth, cmap='Set1', 
                                  alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
            ax2.set_title('HKNN Embedding - Ground Truth Labels', fontsize=14, fontweight='bold')
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('Label', fontsize=10)
        
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Y', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        # Right plot: Color by cluster (K-means)
        if HAS_SKLEARN:
            try:
                n_clusters = min(4, len(coords) // 20)
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(coords)
                    scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], 
                                          c=cluster_labels, cmap='Set1', 
                                          alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
                    ax3.set_title(f'HKNN Embedding - {n_clusters} Clusters (K-means)', fontsize=14, fontweight='bold')
                else:
                    scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], 
                                          c=ids, cmap='tab20', 
                                          alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
                    ax3.set_title('HKNN Embedding - Colored by Point ID', fontsize=14, fontweight='bold')
            except:
                scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], 
                                      c=ids, cmap='tab20', 
                                      alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
                ax3.set_title('HKNN Embedding - Colored by Point ID', fontsize=14, fontweight='bold')
        else:
            scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], 
                                  c=ids, cmap='tab20', 
                                  alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
            ax3.set_title('HKNN Embedding - Colored by Point ID', fontsize=14, fontweight='bold')
        
        ax3.set_xlabel('X', fontsize=12)
        ax3.set_ylabel('Y', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
    else:
        # Two plots: ID and Clusters (original behavior)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left plot: Color by ID (sequential)
        scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], 
                              c=ids, cmap='viridis', 
                              alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_title('HKNN Embedding - Colored by Point ID', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Point ID', fontsize=10)
        
        # Right plot: Color by cluster (using K-means to identify clusters)
        if HAS_SKLEARN:
            try:
                n_clusters = min(4, len(coords) // 20)
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(coords)
                    scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], 
                                          c=cluster_labels, cmap='Set1', 
                                          alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
                    ax2.set_title(f'HKNN Embedding - {n_clusters} Clusters', fontsize=14, fontweight='bold')
                else:
                    scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], 
                                          c=ids, cmap='tab20', 
                                          alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
                    ax2.set_title('HKNN Embedding - Colored by Point ID', fontsize=14, fontweight='bold')
            except:
                scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], 
                                      c=ids, cmap='tab20', 
                                      alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
                ax2.set_title('HKNN Embedding - Colored by Point ID', fontsize=14, fontweight='bold')
        else:
            scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], 
                                  c=ids, cmap='tab20', 
                                  alpha=0.8, s=60, edgecolors='black', linewidth=0.3)
            ax2.set_title('HKNN Embedding - Colored by Point ID', fontsize=14, fontweight='bold')
        
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Y', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

def visualize_3d(ids, coords, output_path=None):
    """Visualize 3D embedding."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=ids, cmap='tab20',
                        alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('HKNN Embedding Visualization (3D)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Point ID', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <embedding.csv> [output.png]")
        print("  embedding.csv: CSV file with columns 'id,x,y' (2D) or 'id,x,y,z' (3D)")
        print("  output.png: Optional output file (if not specified, displays interactively)")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Loading embedding from {csv_path}...")
    ids, coords = load_embedding(csv_path)
    
    print(f"Loaded {len(ids)} points in {coords.shape[1]}D")
    
    # Try to load ground truth colors
    base_path = csv_path.replace('.csv', '')
    ground_truth = load_ground_truth(base_path)
    if ground_truth is not None:
        print(f"Loaded ground truth colors/labels from {base_path}_colors.npy")
    
    if coords.shape[1] == 2:
        print("Visualizing 2D embedding...")
        visualize_2d(ids, coords, output_path, ground_truth)
    elif coords.shape[1] == 3:
        print("Visualizing 3D embedding...")
        visualize_3d(ids, coords, output_path)
    else:
        print(f"Error: Unsupported dimensionality {coords.shape[1]}")
        sys.exit(1)

if __name__ == '__main__':
    main()

