#!/usr/bin/env python3
"""
Enhanced metrics collection for HKNN embeddings.
Collects comprehensive metrics for paper analysis.
"""

import sys
import json
import csv
import numpy as np
import argparse
from pathlib import Path

try:
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.spatial.distance import pdist, squareform, cdist
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_embedding(csv_path):
    """Load embedding from CSV file."""
    coords = []
    ids = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                coords.append([float(row['x']), float(row['y'])])
                ids.append(int(row['id']))
        return np.array(ids), np.array(coords)
    except Exception as e:
        print(f"Error loading embedding: {e}", file=sys.stderr)
        return None, None


def load_ground_truth(base_path):
    """Load ground truth labels if available."""
    try:
        colors = np.load(f'{base_path}_colors.npy')
        return colors
    except FileNotFoundError:
        return None


def compute_knn_preservation(coords, data, K=30):
    """Compute KNN preservation metric."""
    if not HAS_SCIPY:
        return None
    
    try:
        N = len(coords)
        D = data.shape[1] if len(data.shape) > 1 else len(data) // N
        
        if len(data.shape) == 1:
            data = data.reshape(N, D)
        
        # Compute KNN in original space
        orig_dist = cdist(data, data, metric='euclidean')
        orig_knn = np.argsort(orig_dist, axis=1)[:, 1:K+1]
        
        # Compute KNN in embedding space
        embed_dist = cdist(coords, coords, metric='euclidean')
        embed_knn = np.argsort(embed_dist, axis=1)[:, 1:K+1]
        
        # Compute overlap
        overlaps = []
        for i in range(N):
            orig_set = set(orig_knn[i])
            embed_set = set(embed_knn[i])
            overlap = len(orig_set & embed_set)
            overlaps.append(overlap)
        
        overlap_array = np.array(overlaps)
        return {
            'knn_overlap_mean': float(overlap_array.mean()),
            'knn_overlap_median': float(np.median(overlap_array)),
            'knn_overlap_std': float(overlap_array.std()),
            'knn_overlap_min': int(overlap_array.min()),
            'knn_overlap_max': int(overlap_array.max()),
            'knn_overlap_pct': float(100 * overlap_array.mean() / K)
        }
    except Exception as e:
        print(f"Error computing KNN preservation: {e}", file=sys.stderr)
        return None


def compute_cluster_metrics(coords, n_clusters_list=[3, 5, 10]):
    """Compute cluster quality metrics."""
    if not HAS_SKLEARN:
        return {}
    
    metrics = {}
    
    for n_clusters in n_clusters_list:
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coords)
            
            # Silhouette score
            sil_score = silhouette_score(coords, cluster_labels)
            metrics[f'silhouette_{n_clusters}'] = float(sil_score)
            
            # Cluster balance
            unique, counts = np.unique(cluster_labels, return_counts=True)
            cluster_sizes = counts
            max_cluster_pct = float(100 * cluster_sizes.max() / len(coords))
            min_cluster_size = int(cluster_sizes.min())
            cluster_std = float(np.std(cluster_sizes))
            
            metrics[f'max_cluster_pct_{n_clusters}'] = max_cluster_pct
            metrics[f'min_cluster_size_{n_clusters}'] = min_cluster_size
            metrics[f'cluster_std_{n_clusters}'] = cluster_std
            metrics[f'cluster_imbalance_{n_clusters}'] = float(cluster_sizes.max() / min_cluster_size if min_cluster_size > 0 else float('inf'))
            
        except Exception as e:
            print(f"Error computing cluster metrics for {n_clusters} clusters: {e}", file=sys.stderr)
    
    return metrics


def compute_separation_metrics(coords):
    """Compute point separation metrics."""
    if not HAS_SCIPY:
        return {}
    
    try:
        dist_matrix = squareform(pdist(coords))
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_distances = np.min(dist_matrix, axis=1)
        
        # Check for duplicates
        unique_coords = len(np.unique(coords, axis=0))
        duplicates = len(coords) - unique_coords
        
        return {
            'min_dist_mean': float(min_distances.mean()),
            'min_dist_median': float(np.median(min_distances)),
            'min_dist_std': float(min_distances.std()),
            'min_dist_min': float(min_distances.min()),
            'min_dist_max': float(min_distances.max()),
            'unique_coords': int(unique_coords),
            'duplicate_coords': int(duplicates),
            'duplicate_pct': float(100 * duplicates / len(coords))
        }
    except Exception as e:
        print(f"Error computing separation metrics: {e}", file=sys.stderr)
        return {}


def compute_spread_metrics(coords):
    """Compute embedding spread metrics."""
    try:
        x_span = float(coords[:, 0].max() - coords[:, 0].min())
        y_span = float(coords[:, 1].max() - coords[:, 1].min())
        x_std = float(coords[:, 0].std())
        y_std = float(coords[:, 1].std())
        
        dist_from_origin = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        
        return {
            'x_span': x_span,
            'y_span': y_span,
            'x_std': x_std,
            'y_std': y_std,
            'mean_dist_from_origin': float(dist_from_origin.mean()),
            'max_dist_from_origin': float(dist_from_origin.max())
        }
    except Exception as e:
        print(f"Error computing spread metrics: {e}", file=sys.stderr)
        return {}


def compute_local_density(coords, k=10):
    """Compute local density metrics."""
    if not HAS_SKLEARN:
        return {}
    
    try:
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        local_density = distances[:, 1:].mean(axis=1)
        
        return {
            'local_density_mean': float(local_density.mean()),
            'local_density_median': float(np.median(local_density)),
            'local_density_std': float(local_density.std()),
            'local_density_min': float(local_density.min()),
            'local_density_max': float(local_density.max())
        }
    except Exception as e:
        print(f"Error computing local density: {e}", file=sys.stderr)
        return {}


def compute_ground_truth_correlation(coords, ground_truth):
    """Compute correlation with ground truth if available."""
    if ground_truth is None or not HAS_SCIPY:
        return {}
    
    try:
        # For discrete labels, compute ARI if possible
        if ground_truth.dtype in [np.int32, np.int64]:
            # Try clustering and compare
            if HAS_SKLEARN:
                from sklearn.cluster import KMeans
                n_clusters = len(np.unique(ground_truth))
                if n_clusters > 1 and n_clusters < len(coords) // 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(coords)
                    ari = adjusted_rand_score(ground_truth, cluster_labels)
                    return {'ari_score': float(ari)}
        
        # For continuous, compute spatial correlation
        elif ground_truth.dtype in [np.float32, np.float64]:
            embedding_dist = pdist(coords)
            gt_dist = np.abs(pdist(ground_truth.reshape(-1, 1)))
            corr, p_value = spearmanr(embedding_dist, gt_dist)
            return {
                'spatial_correlation': float(corr),
                'correlation_pvalue': float(p_value)
            }
    except Exception as e:
        print(f"Error computing ground truth correlation: {e}", file=sys.stderr)
    
    return {}


def main():
    parser = argparse.ArgumentParser(description='Collect comprehensive metrics for embedding')
    parser.add_argument('--embedding', required=True, help='Embedding CSV file')
    parser.add_argument('--data', help='Original data file (.f32) for KNN analysis')
    parser.add_argument('--output', required=True, help='Output JSON file for metrics')
    parser.add_argument('--k', type=int, default=30, help='K for KNN preservation')
    
    args = parser.parse_args()
    
    # Load embedding
    ids, coords = load_embedding(args.embedding)
    if ids is None or coords is None:
        print("Error: Could not load embedding", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(coords)} points")
    
    # Collect all metrics
    all_metrics = {}
    
    # Basic metrics
    all_metrics['n_points'] = int(len(coords))
    all_metrics['dim'] = int(coords.shape[1])
    
    # Separation metrics
    print("Computing separation metrics...")
    all_metrics.update(compute_separation_metrics(coords))
    
    # Spread metrics
    print("Computing spread metrics...")
    all_metrics.update(compute_spread_metrics(coords))
    
    # Cluster metrics
    print("Computing cluster metrics...")
    all_metrics.update(compute_cluster_metrics(coords))
    
    # Local density
    print("Computing local density...")
    all_metrics.update(compute_local_density(coords))
    
    # KNN preservation
    if args.data:
        print("Computing KNN preservation...")
        try:
            data = np.fromfile(args.data.replace('.csv', '.f32'), dtype=np.float32)
            N = len(coords)
            D = len(data) // N
            data = data.reshape(N, D)
            knn_metrics = compute_knn_preservation(coords, data, args.k)
            if knn_metrics:
                all_metrics.update(knn_metrics)
        except Exception as e:
            print(f"Warning: Could not compute KNN preservation: {e}", file=sys.stderr)
    
    # Ground truth correlation
    base_path = args.embedding.replace('.csv', '')
    ground_truth = load_ground_truth(base_path)
    if ground_truth is not None:
        print("Computing ground truth correlation...")
        all_metrics.update(compute_ground_truth_correlation(coords, ground_truth))
    
    # Save metrics
    with open(args.output, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Metrics saved to {args.output}")
    print(f"Collected {len(all_metrics)} metrics")


if __name__ == '__main__':
    main()

