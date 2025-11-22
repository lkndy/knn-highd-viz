#!/usr/bin/env python3
"""
Analyze and visualize test suite results.
Generates comprehensive analysis for paper.
"""

import json
import csv
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.")


def load_results(results_dir):
    """Load test results."""
    results_file = os.path.join(results_dir, 'results.json')
    summary_file = os.path.join(results_dir, 'results_summary.csv')
    
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        return df
    elif os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        # Convert to DataFrame
        rows = []
        for r in results:
            if r.get('success'):
                row = {}
                config = r.get('config', {})
                row.update({f'config_{k}': v for k, v in config.items()})
                row.update({k: v for k, v in r.items() if k not in ['config', 'test_id']})
                row['test_id'] = r.get('test_id', '')
                rows.append(row)
        return pd.DataFrame(rows)
    else:
        raise FileNotFoundError(f"No results found in {results_dir}")


def analyze_parameter_sensitivity(df, param_name, metric_name='knn_overlap_pct'):
    """Analyze sensitivity to a parameter."""
    if param_name not in df.columns:
        return None
    
    # Filter successful runs
    df_success = df[df['success'] == 1].copy()
    if len(df_success) == 0:
        return None
    
    # Group by parameter value
    grouped = df_success.groupby(param_name)[metric_name].agg(['mean', 'std', 'count'])
    
    return {
        'parameter': param_name,
        'metric': metric_name,
        'values': grouped.index.tolist(),
        'means': grouped['mean'].tolist(),
        'stds': grouped['std'].tolist(),
        'counts': grouped['count'].tolist()
    }


def generate_parameter_sweep_plots(df, output_dir):
    """Generate plots for parameter sweeps."""
    if not HAS_PLOTTING:
        print("Skipping plots (matplotlib not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters to analyze
    params = ['K', 'perplexity', 'lr', 'epochs_coarse', 'epochs_fine', 'mneg', 'gamma']
    metrics = ['knn_overlap_pct', 'silhouette_10', 'elapsed_time', 'duplicate_pct']
    
    for param in params:
        param_col = f'config_{param}' if f'config_{param}' in df.columns else param
        if param_col not in df.columns:
            continue
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            # Filter successful runs
            df_success = df[df['success'] == 1].copy()
            if len(df_success) == 0:
                continue
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot with trend line
            x = df_success[param_col].values
            y = df_success[metric].values
            
            ax.scatter(x, y, alpha=0.6, s=50)
            
            # Trend line
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend')
            
            ax.set_xlabel(param)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} vs {param}')
            ax.grid(True, alpha=0.3)
            if len(x) > 1:
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param}_{metric}.png'), dpi=150)
            plt.close()
    
    print(f"Generated plots in {output_dir}")


def generate_embedding_visualizations(results_dir, output_dir):
    """Generate scatter plots for all embedding CSVs in results_dir."""
    if not HAS_PLOTTING:
        print("Skipping embedding visualizations (matplotlib not available)")
        return
        
    print(f"Generating embedding visualizations in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find ground truth colors
    ground_truth = None
    
    # Check standard locations for test_mnist_colors.npy
    potential_gt_paths = [
        os.path.join(results_dir, 'test_mnist_colors.npy'),
        os.path.join(os.path.dirname(results_dir), 'test_mnist_colors.npy'), # Parent dir
        'test_mnist_colors.npy', # CWD
        'test_data_colors.npy',
        'mnist_data_colors.npy'
    ]
    
    for path in potential_gt_paths:
        if os.path.exists(path):
            try:
                ground_truth = np.load(path)
                print(f"Loaded ground truth from {path}")
                break
            except:
                pass
    
    # Find all CSV files that look like embeddings
    csv_files = []
    for f in os.listdir(results_dir):
        if f.endswith('.csv') and f.startswith('test_') and 'summary' not in f:
            csv_files.append(os.path.join(results_dir, f))
            
    if not csv_files:
        print("No embedding CSV files found to visualize.")
        return

    for csv_path in csv_files:
        try:
            # Load embedding
            ids = []
            coords = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ids.append(int(row['id']))
                    if 'z' in row:
                        coords.append([float(row['x']), float(row['y']), float(row['z'])])
                    else:
                        coords.append([float(row['x']), float(row['y'])])
            
            ids = np.array(ids)
            coords = np.array(coords)
            
            if len(coords) == 0:
                continue
                
            # Output path
            basename = os.path.basename(csv_path).replace('.csv', '.png')
            out_path = os.path.join(output_dir, basename)
            
            # Plot
            plt.figure(figsize=(10, 8))
            
            if ground_truth is not None and len(ground_truth) == len(coords):
                 # Plot with ground truth colors
                plt.scatter(coords[:, 0], coords[:, 1], c=ground_truth, cmap='tab10', 
                           alpha=0.6, s=15, edgecolors='none')
                plt.colorbar(label='Ground Truth Label')
                plt.title(f"Embedding: {basename[:-4]} (Ground Truth Colors)")
            else:
                # Plot with ID colors
                plt.scatter(coords[:, 0], coords[:, 1], c=ids, cmap='viridis', 
                           alpha=0.6, s=15, edgecolors='none')
                plt.colorbar(label='Point ID')
                plt.title(f"Embedding: {basename[:-4]}")
                
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"  Generated {basename}")
            
        except Exception as e:
            print(f"  Failed to visualize {csv_path}: {e}")


def generate_summary_report(df, output_file):
    """Generate text summary report."""
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HKNN ALGORITHM ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*70 + "\n")
        total = len(df)
        successful = len(df[df['success'] == 1])
        f.write(f"Total tests: {total}\n")
        f.write(f"Successful: {successful} ({100*successful/total:.1f}%)\n")
        f.write(f"Failed: {total - successful} ({100*(total-successful)/total:.1f}%)\n\n")
        
        if successful > 0:
            df_success = df[df['success'] == 1].copy()
            
            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*70 + "\n")
            if 'elapsed_time' in df_success.columns:
                f.write(f"Mean runtime: {df_success['elapsed_time'].mean():.2f}s\n")
                f.write(f"Median runtime: {df_success['elapsed_time'].median():.2f}s\n")
                f.write(f"Min runtime: {df_success['elapsed_time'].min():.2f}s\n")
                f.write(f"Max runtime: {df_success['elapsed_time'].max():.2f}s\n\n")
            
            # Quality metrics
            f.write("QUALITY METRICS\n")
            f.write("-"*70 + "\n")
            
            if 'knn_overlap_pct' in df_success.columns:
                f.write(f"KNN Preservation:\n")
                f.write(f"  Mean: {df_success['knn_overlap_pct'].mean():.2f}%\n")
                f.write(f"  Median: {df_success['knn_overlap_pct'].median():.2f}%\n")
                f.write(f"  Best: {df_success['knn_overlap_pct'].max():.2f}%\n")
                f.write(f"  Worst: {df_success['knn_overlap_pct'].min():.2f}%\n\n")
            
            if 'silhouette_10' in df_success.columns:
                f.write(f"Cluster Quality (10 clusters):\n")
                f.write(f"  Mean silhouette: {df_success['silhouette_10'].mean():.4f}\n")
                f.write(f"  Best: {df_success['silhouette_10'].max():.4f}\n\n")
            
            if 'duplicate_pct' in df_success.columns:
                f.write(f"Coordinate Uniqueness:\n")
                f.write(f"  Mean duplicate %: {df_success['duplicate_pct'].mean():.2f}%\n")
                f.write(f"  Best (lowest duplicates): {df_success['duplicate_pct'].min():.2f}%\n\n")
            
            # Parameter analysis
            f.write("PARAMETER SENSITIVITY\n")
            f.write("-"*70 + "\n")
            
            params = ['K', 'perplexity', 'lr', 'epochs_coarse', 'epochs_fine', 'mneg', 'gamma']
            for param in params:
                param_col = f'config_{param}' if f'config_{param}' in df.columns else param
                if param_col in df_success.columns and 'knn_overlap_pct' in df_success.columns:
                    grouped = df_success.groupby(param_col)['knn_overlap_pct'].agg(['mean', 'std'])
                    f.write(f"\n{param}:\n")
                    for val, row in grouped.iterrows():
                        f.write(f"  {val}: {row['mean']:.2f}% ± {row['std']:.2f}%\n")
            
            # Best configurations
            f.write("\nBEST CONFIGURATIONS\n")
            f.write("-"*70 + "\n")
            
            if 'knn_overlap_pct' in df_success.columns:
                best_knn = df_success.nlargest(5, 'knn_overlap_pct')
                f.write("\nTop 5 by KNN Preservation:\n")
                for idx, row in best_knn.iterrows():
                    f.write(f"  {row.get('test_id', 'unknown')}: {row['knn_overlap_pct']:.2f}%\n")
                    if 'config_K' in row:
                        f.write(f"    K={row.get('config_K', '?')}, "
                               f"perp={row.get('config_perplexity', '?')}, "
                               f"lr={row.get('config_lr', '?')}\n")
            
            if 'silhouette_10' in df_success.columns:
                best_sil = df_success.nlargest(5, 'silhouette_10')
                f.write("\nTop 5 by Cluster Quality:\n")
                for idx, row in best_sil.iterrows():
                    f.write(f"  {row.get('test_id', 'unknown')}: {row['silhouette_10']:.4f}\n")
    
    print(f"Summary report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze test suite results')
    parser.add_argument('--results-dir', required=True, help='Results directory')
    parser.add_argument('--output-dir', default=None, help='Output directory for analysis')
    parser.add_argument('--generate-plots', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} test results")
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate summary report
    report_file = os.path.join(args.output_dir, 'summary_report.txt')
    generate_summary_report(df, report_file)
    
    # Generate plots
    if args.generate_plots and HAS_PLOTTING:
        plots_dir = os.path.join(args.output_dir, 'plots')
        generate_parameter_sweep_plots(df, plots_dir)
        
        # Generate embedding visualizations
        viz_dir = os.path.join(args.output_dir, 'viz')
        generate_embedding_visualizations(args.results_dir, viz_dir)
    
    # Export detailed CSV
    csv_file = os.path.join(args.output_dir, 'detailed_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"Detailed results saved to {csv_file}")
    
    print(f"\nAnalysis complete! Results in {args.output_dir}")


if __name__ == '__main__':
    main()

