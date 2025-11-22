#!/usr/bin/env python3
"""
Comprehensive embedding quality analysis script.
Analyzes KNN graph correctness, embedding quality, cluster structure, and optimization metrics.
"""

import sys
import csv
import numpy as np
from collections import Counter
import argparse

try:
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available. Some metrics will be skipped.")

try:
    from scipy.spatial.distance import pdist, squareform, cdist
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Some metrics will be skipped.")


def load_embedding(csv_path):
    """Load embedding from CSV file."""
    coords = []
    ids = []
    for r in csv.DictReader(open(csv_path)):
        coords.append([float(r['x']), float(r['y'])])
        ids.append(int(r['id']))
    return np.array(ids), np.array(coords)


def load_ground_truth(base_path):
    """Load ground truth colors/labels if available."""
    try:
        colors = np.load(f'{base_path}_colors.npy')
        return colors
    except FileNotFoundError:
        return None


def analyze_point_separation(coords):
    """Analyze point-to-point distances."""
    print("\n" + "="*70)
    print("POINT SEPARATION ANALYSIS")
    print("="*70)
    
    if not HAS_SCIPY:
        print("  Skipped (scipy not available)")
        return
    
    dist_matrix = squareform(pdist(coords))
    np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances
    
    min_distances = np.min(dist_matrix, axis=1)
    print(f"\nMin distance to nearest neighbor:")
    print(f"  Mean:   {min_distances.mean():.6f}")
    print(f"  Median: {np.median(min_distances):.6f}")
    print(f"  Min:    {min_distances.min():.6f}")
    print(f"  Max:    {min_distances.max():.6f}")
    print(f"  Std:    {min_distances.std():.6f}")
    
    # Distribution analysis
    very_close = np.sum(min_distances < 0.1)
    close = np.sum((min_distances >= 0.1) & (min_distances < 1.0))
    medium = np.sum((min_distances >= 1.0) & (min_distances < 10.0))
    far = np.sum(min_distances >= 10.0)
    
    print(f"\nDistance distribution:")
    print(f"  Very close (<0.1):     {very_close:4d} ({100*very_close/len(coords):5.1f}%)")
    print(f"  Close (0.1-1.0):       {close:4d} ({100*close/len(coords):5.1f}%)")
    print(f"  Medium (1.0-10.0):     {medium:4d} ({100*medium/len(coords):5.1f}%)")
    print(f"  Far (>10.0):           {far:4d} ({100*far/len(coords):5.1f}%)")
    
    # Check for duplicate coordinates
    unique_coords = len(np.unique(coords, axis=0))
    duplicates = len(coords) - unique_coords
    print(f"\nCoordinate uniqueness:")
    print(f"  Unique pairs: {unique_coords}/{len(coords)} ({100*unique_coords/len(coords):.1f}%)")
    if duplicates > 0:
        print(f"  ⚠️  WARNING: {duplicates} duplicate coordinate pairs detected!")
    
    return {
        'min_dist_mean': min_distances.mean(),
        'min_dist_median': np.median(min_distances),
        'min_dist_std': min_distances.std(),
        'very_close_pct': 100*very_close/len(coords),
        'duplicates': duplicates
    }


def analyze_spread(coords):
    """Analyze embedding spread and scale."""
    print("\n" + "="*70)
    print("EMBEDDING SPREAD ANALYSIS")
    print("="*70)
    
    print(f"\nCoordinate ranges:")
    print(f"  X: [{coords[:,0].min():8.2f}, {coords[:,0].max():8.2f}] (span: {coords[:,0].max()-coords[:,0].min():.2f})")
    print(f"  Y: [{coords[:,1].min():8.2f}, {coords[:,1].max():8.2f}] (span: {coords[:,1].max()-coords[:,1].min():.2f})")
    
    print(f"\nStandard deviations:")
    print(f"  X std: {coords[:,0].std():8.4f}")
    print(f"  Y std: {coords[:,1].std():8.4f}")
    print(f"  Combined: {np.sqrt(coords[:,0].std()**2 + coords[:,1].std()**2):8.4f}")
    
    # Distance from origin
    dist_from_origin = np.sqrt(coords[:,0]**2 + coords[:,1]**2)
    print(f"\nDistance from origin:")
    print(f"  Mean: {dist_from_origin.mean():.4f}")
    print(f"  Median: {np.median(dist_from_origin):.4f}")
    print(f"  Max: {dist_from_origin.max():.4f}")
    
    return {
        'x_span': coords[:,0].max() - coords[:,0].min(),
        'y_span': coords[:,1].max() - coords[:,1].min(),
        'x_std': coords[:,0].std(),
        'y_std': coords[:,1].std(),
        'mean_dist_from_origin': dist_from_origin.mean()
    }


def analyze_clusters(coords, n_clusters_list=[3, 4, 5, 6, 10]):
    """Analyze cluster structure using K-means."""
    print("\n" + "="*70)
    print("CLUSTER STRUCTURE ANALYSIS")
    print("="*70)
    
    if not HAS_SKLEARN:
        print("  Skipped (sklearn not available)")
        return {}
    
    results = {}
    
    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        cluster_counts = Counter(cluster_labels)
        sizes = list(cluster_counts.values())
        
        print(f"\n{n_clusters} clusters:")
        print(f"  Size range: [{min(sizes)}, {max(sizes)}]")
        print(f"  Size std: {np.std(sizes):.1f}")
        print(f"  Size ratio (max/min): {max(sizes)/min(sizes) if min(sizes) > 0 else float('inf'):.1f}")
        
        # Cluster distribution
        for c in sorted(cluster_counts.keys()):
            count = cluster_counts[c]
            pct = 100 * count / len(coords)
            if pct > 10 or count < 5:  # Highlight imbalanced clusters
                marker = " ⚠️ " if (pct > 50 or count < 5) else "   "
                print(f"{marker}Cluster {c}: {count:4d} points ({pct:5.1f}%)")
        
        # Silhouette score
        try:
            sil_score = silhouette_score(coords, cluster_labels)
            print(f"  Silhouette score: {sil_score:.4f}")
            results[f'silhouette_{n_clusters}'] = sil_score
        except:
            pass
        
        # Check for imbalanced clusters
        max_cluster_pct = max(sizes) / len(coords) * 100
        min_cluster_size = min(sizes)
        if max_cluster_pct > 70:
            print(f"  ⚠️  WARNING: Largest cluster contains {max_cluster_pct:.1f}% of points!")
        if min_cluster_size == 1:
            print(f"  ⚠️  WARNING: {sum(1 for s in sizes if s == 1)} singleton clusters detected!")
        
        results[f'max_cluster_pct_{n_clusters}'] = max_cluster_pct
        results[f'min_cluster_size_{n_clusters}'] = min_cluster_size
        results[f'cluster_std_{n_clusters}'] = np.std(sizes)
    
    return results


def analyze_local_density(coords, k=10):
    """Analyze local point density."""
    print("\n" + "="*70)
    print("LOCAL DENSITY ANALYSIS")
    print("="*70)
    
    if not HAS_SKLEARN:
        print("  Skipped (sklearn not available)")
        return {}
    
    nn = NearestNeighbors(n_neighbors=k+1)  # k neighbors + self
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    local_density = distances[:, 1:].mean(axis=1)  # Exclude self
    
    print(f"\nLocal density (mean distance to {k} nearest neighbors):")
    print(f"  Mean:   {local_density.mean():.6f}")
    print(f"  Median: {np.median(local_density):.6f}")
    print(f"  Min:    {local_density.min():.6f}")
    print(f"  Max:    {local_density.max():.6f}")
    print(f"  Std:    {local_density.std():.6f}")
    
    # Density distribution
    tight = np.sum(local_density < 1.0)
    medium = np.sum((local_density >= 1.0) & (local_density < 10.0))
    sparse = np.sum(local_density >= 10.0)
    
    print(f"\nDensity distribution:")
    print(f"  Tight (<1.0):    {tight:4d} ({100*tight/len(coords):5.1f}%)")
    print(f"  Medium (1-10):   {medium:4d} ({100*medium/len(coords):5.1f}%)")
    print(f"  Sparse (>10):    {sparse:4d} ({100*sparse/len(coords):5.1f}%)")
    
    return {
        'local_density_mean': local_density.mean(),
        'local_density_median': np.median(local_density),
        'local_density_std': local_density.std(),
        'tight_clusters_pct': 100*tight/len(coords)
    }


def analyze_ground_truth_correlation(coords, ground_truth):
    """Analyze correlation with ground truth if available."""
    print("\n" + "="*70)
    print("GROUND TRUTH CORRELATION")
    print("="*70)
    
    if ground_truth is None:
        print("  No ground truth available")
        return {}
    
    print(f"\nGround truth statistics:")
    print(f"  Range: [{ground_truth.min():.2f}, {ground_truth.max():.2f}]")
    print(f"  Mean: {ground_truth.mean():.2f}, Std: {ground_truth.std():.2f}")
    print(f"  Unique values: {len(np.unique(ground_truth))}")
    
    # Check if ground truth is continuous or discrete
    is_continuous = ground_truth.dtype in [np.float32, np.float64]
    
    if is_continuous:
        # For continuous (e.g., Swiss roll parameter), check spatial correlation
        if HAS_SCIPY:
            # Compute distance in embedding space vs distance in ground truth
            embedding_dist = pdist(coords)
            gt_dist = np.abs(pdist(ground_truth.reshape(-1, 1)))
            
            # Spearman correlation
            corr, p_value = spearmanr(embedding_dist, gt_dist)
            print(f"\nSpatial correlation (Spearman):")
            print(f"  Correlation: {corr:.4f}")
            print(f"  P-value: {p_value:.2e}")
            
            if corr > 0.5:
                print(f"  ✓ Good correlation with ground truth structure")
            elif corr > 0.3:
                print(f"  ⚠️  Moderate correlation")
            else:
                print(f"  ⚠️  WARNING: Low correlation with ground truth")
            
            return {'spatial_correlation': corr, 'correlation_pvalue': p_value}
    else:
        # For discrete labels, use ARI if we have clustering
        print(f"  Ground truth is discrete (labels)")
        return {}
    
    return {}


def analyze_knn_preservation(coords, data_path, K=30):
    """Check if KNN structure is preserved in embedding."""
    print("\n" + "="*70)
    print("KNN PRESERVATION ANALYSIS")
    print("="*70)
    
    try:
        # Load original data
        data = np.fromfile(data_path.replace('.csv', '.f32'), dtype=np.float32)
        N = len(coords)
        D = len(data) // N
        data = data.reshape(N, D)
        
        # Compute KNN in original space
        if HAS_SCIPY:
            orig_dist = cdist(data, data, metric='euclidean')
            orig_knn = np.argsort(orig_dist, axis=1)[:, 1:K+1]  # Skip self
            
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
            print(f"\nKNN preservation (K={K}):")
            print(f"  Mean overlap: {overlap_array.mean():.2f}/{K} ({100*overlap_array.mean()/K:.1f}%)")
            print(f"  Median overlap: {np.median(overlap_array):.2f}/{K} ({100*np.median(overlap_array)/K:.1f}%)")
            print(f"  Min overlap: {overlap_array.min()}/{K} ({100*overlap_array.min()/K:.1f}%)")
            print(f"  Max overlap: {overlap_array.max()}/{K} ({100*overlap_array.max()/K:.1f}%)")
            
            if overlap_array.mean() / K > 0.5:
                print(f"  ✓ Good KNN preservation")
            elif overlap_array.mean() / K > 0.3:
                print(f"  ⚠️  Moderate KNN preservation")
            else:
                print(f"  ⚠️  WARNING: Poor KNN preservation")
            
            return {
                'knn_overlap_mean': overlap_array.mean(),
                'knn_overlap_pct': 100*overlap_array.mean()/K
            }
    except Exception as e:
        print(f"  Could not analyze KNN preservation: {e}")
        return {}
    
    return {}


def generate_summary(all_results):
    """Generate summary and recommendations."""
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    issues = []
    warnings = []
    good = []
    
    # Check for collapse
    if 'very_close_pct' in all_results and all_results['very_close_pct'] > 50:
        issues.append(f"Embedding collapse: {all_results['very_close_pct']:.1f}% of points have nearest neighbor < 0.1")
    
    if 'duplicates' in all_results and all_results['duplicates'] > 0:
        issues.append(f"Duplicate coordinates: {all_results['duplicates']} pairs")
    
    # Check cluster balance
    for key in all_results:
        if 'max_cluster_pct' in key:
            pct = all_results[key]
            if pct > 70:
                issues.append(f"Imbalanced clusters: largest cluster contains {pct:.1f}% of points ({key})")
            elif pct > 50:
                warnings.append(f"Moderately imbalanced: largest cluster contains {pct:.1f}% ({key})")
    
    # Check spread
    if 'x_span' in all_results:
        span = max(all_results['x_span'], all_results['y_span'])
        if span < 10:
            warnings.append(f"Small embedding span: {span:.2f}")
        elif span > 1000:
            warnings.append(f"Very large embedding span: {span:.2f}")
    
    # Check KNN preservation
    if 'knn_overlap_pct' in all_results:
        pct = all_results['knn_overlap_pct']
        if pct < 30:
            issues.append(f"Poor KNN preservation: only {pct:.1f}% overlap")
        elif pct < 50:
            warnings.append(f"Moderate KNN preservation: {pct:.1f}% overlap")
        else:
            good.append(f"Good KNN preservation: {pct:.1f}% overlap")
    
    # Check spatial correlation
    if 'spatial_correlation' in all_results:
        corr = all_results['spatial_correlation']
        if corr > 0.5:
            good.append(f"Good ground truth correlation: {corr:.3f}")
        elif corr < 0.3:
            warnings.append(f"Low ground truth correlation: {corr:.3f}")
    
    if issues:
        print("\n❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"  • {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
    
    if good:
        print("\n✓ POSITIVE INDICATORS:")
        for item in good:
            print(f"  • {item}")
    
    if not issues and not warnings:
        print("\n✓ No major issues detected. Embedding looks good!")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive embedding quality analysis')
    parser.add_argument('embedding_csv', help='Path to embedding CSV file')
    parser.add_argument('--data', help='Path to original data file (for KNN analysis)', default=None)
    parser.add_argument('--k', type=int, default=30, help='K for KNN preservation analysis')
    args = parser.parse_args()
    
    print("="*70)
    print("EMBEDDING QUALITY ANALYSIS")
    print("="*70)
    print(f"\nLoading embedding from: {args.embedding_csv}")
    
    ids, coords = load_embedding(args.embedding_csv)
    print(f"Loaded {len(ids)} points in {coords.shape[1]}D")
    
    # Try to load ground truth
    base_path = args.embedding_csv.replace('.csv', '')
    ground_truth = load_ground_truth(base_path)
    if ground_truth is not None:
        print(f"Loaded ground truth from: {base_path}_colors.npy")
    
    # Data path for KNN analysis
    data_path = args.data if args.data else args.embedding_csv
    
    # Run all analyses
    all_results = {}
    
    all_results.update(analyze_point_separation(coords))
    all_results.update(analyze_spread(coords))
    all_results.update(analyze_clusters(coords))
    all_results.update(analyze_local_density(coords))
    
    if ground_truth is not None:
        all_results.update(analyze_ground_truth_correlation(coords, ground_truth))
    
    all_results.update(analyze_knn_preservation(coords, data_path, args.k))
    
    # Generate summary
    generate_summary(all_results)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()


