#!/usr/bin/env python3
"""
Comprehensive test suite for HKNN algorithm analysis.
Runs multiple configurations and collects metrics for paper analysis.
"""

import subprocess
import json
import csv
import time
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product
import numpy as np

# Test configurations
TEST_CONFIGS = {
    # Parameter sweeps
    'K_values': [30, 50, 100, 150],
    'perplexity_values': [30, 50, 75, 100],
    'learning_rate_values': [50, 100, 200, 400],
    'epochs_coarse_values': [10, 25, 50, 100],
    'epochs_fine_values': [20, 50, 100, 200],
    'mneg_values': [3, 5, 10],
    'gamma_values': [1.0, 3.0, 5.0, 7.0],
    'kml_values': [2, 3, 4, 5],
    'rho_values': [0.6, 0.7, 0.8, 0.9],
    
    # Dataset sizes
    'dataset_sizes': [500, 1000, 2000, 5000],
    
    # Fixed baseline
    'baseline': {
        'K': 100,
        'perplexity': 50,
        'lr': 200,
        'epochs_coarse': 50,
        'epochs_fine': 100,
        'mneg': 5,
        'gamma': 7.0,
        'kml': 3,
        'rho': 0.8,
    }
}

def run_embedding(config, data_path, N, D, output_dir, seed=123):
    """Run embedding with given configuration."""
    output_name = f"test_{config['test_id']}"
    output_path = os.path.join(output_dir, output_name)
    
    cmd = [
        './build/embed_hknn',
        '--input', data_path,
        '--N', str(N),
        '--D', str(D),
        '--out', output_path,
        '--K', str(config['K']),
        '--perplexity', str(config['perplexity']),
        '--lr', str(config['lr']),
        '--epochs_coarse', str(config['epochs_coarse']),
        '--epochs_fine', str(config['epochs_fine']),
        '--mneg', str(config['mneg']),
        '--gamma', str(config['gamma']),
        '--kml', str(config['kml']),
        '--rho', str(config['rho']),
        '--dim', '2',
        '--seed', str(seed),
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': result.stderr,
                'test_id': config['test_id'],
                'elapsed_time': elapsed_time
            }
        
        # Parse output for timing info
        output_lines = result.stdout.split('\n')
        timing_info = {}
        for line in output_lines:
            if 'ms' in line or 'seconds' in line or 'Total time' in line:
                timing_info['log_line'] = line.strip()
        
        return {
            'success': True,
            'test_id': config['test_id'],
            'output_path': f"{output_path}.csv",
            'elapsed_time': elapsed_time,
            'timing_info': timing_info
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Timeout (1 hour)',
            'test_id': config['test_id'],
            'elapsed_time': 3600
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'test_id': config['test_id'],
            'elapsed_time': time.time() - start_time
        }

def collect_metrics(embedding_csv, data_path, output_dir, test_id):
    """Collect comprehensive metrics for an embedding."""
    metrics_script = 'collect_metrics.py'
    cmd = [
        'uv', 'run', 'python', metrics_script,
        '--embedding', embedding_csv,
        '--data', data_path,
        '--output', os.path.join(output_dir, f'metrics_{test_id}.json')
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Load the metrics
            metrics_path = os.path.join(output_dir, f'metrics_{test_id}.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    return json.load(f)
        return None
    except Exception as e:
        print(f"Warning: Could not collect metrics for {test_id}: {e}")
        return None

def generate_test_configs(test_type='baseline'):
    """Generate test configurations."""
    configs = []
    test_id = 0
    
    baseline = TEST_CONFIGS['baseline']
    
    if test_type == 'baseline':
        configs.append({
            'test_id': f'baseline_{test_id}',
            **baseline
        })
    
    elif test_type == 'K_sweep':
        for K in TEST_CONFIGS['K_values']:
            configs.append({
                'test_id': f'K_{K}_{test_id}',
                **{**baseline, 'K': K}
            })
            test_id += 1
    
    elif test_type == 'perplexity_sweep':
        for perp in TEST_CONFIGS['perplexity_values']:
            configs.append({
                'test_id': f'perp_{perp}_{test_id}',
                **{**baseline, 'perplexity': perp}
            })
            test_id += 1
    
    elif test_type == 'lr_sweep':
        for lr in TEST_CONFIGS['learning_rate_values']:
            configs.append({
                'test_id': f'lr_{lr}_{test_id}',
                **{**baseline, 'lr': lr}
            })
            test_id += 1
    
    elif test_type == 'epochs_sweep':
        for ec, ef in product(TEST_CONFIGS['epochs_coarse_values'], 
                              TEST_CONFIGS['epochs_fine_values']):
            configs.append({
                'test_id': f'ep_{ec}_{ef}_{test_id}',
                **{**baseline, 'epochs_coarse': ec, 'epochs_fine': ef}
            })
            test_id += 1
    
    elif test_type == 'mneg_sweep':
        for mneg in TEST_CONFIGS['mneg_values']:
            configs.append({
                'test_id': f'mneg_{mneg}_{test_id}',
                **{**baseline, 'mneg': mneg}
            })
            test_id += 1
    
    elif test_type == 'gamma_sweep':
        for gamma in TEST_CONFIGS['gamma_values']:
            configs.append({
                'test_id': f'gamma_{gamma}_{test_id}',
                **{**baseline, 'gamma': gamma}
            })
            test_id += 1
    
    elif test_type == 'coarsening_sweep':
        for kml, rho in product(TEST_CONFIGS['kml_values'], 
                                TEST_CONFIGS['rho_values']):
            configs.append({
                'test_id': f'coarse_{kml}_{rho}_{test_id}',
                **{**baseline, 'kml': kml, 'rho': rho}
            })
            test_id += 1
    
    elif test_type == 'comprehensive':
        # Run all single-parameter sweeps
        for K in TEST_CONFIGS['K_values']:
            configs.append({
                'test_id': f'K_{K}',
                **{**baseline, 'K': K}
            })
        for perp in TEST_CONFIGS['perplexity_values']:
            configs.append({
                'test_id': f'perp_{perp}',
                **{**baseline, 'perplexity': perp}
            })
        for lr in TEST_CONFIGS['learning_rate_values']:
            configs.append({
                'test_id': f'lr_{lr}',
                **{**baseline, 'lr': lr}
            })
        for ec, ef in [(25, 50), (50, 100), (100, 200)]:
            configs.append({
                'test_id': f'ep_{ec}_{ef}',
                **{**baseline, 'epochs_coarse': ec, 'epochs_fine': ef}
            })
        for mneg in TEST_CONFIGS['mneg_values']:
            configs.append({
                'test_id': f'mneg_{mneg}',
                **{**baseline, 'mneg': mneg}
            })
        for gamma in [3.0, 5.0, 7.0]:
            configs.append({
                'test_id': f'gamma_{gamma}',
                **{**baseline, 'gamma': gamma}
            })
    
    elif test_type == 'custom':
        # User-defined custom test
        configs.append({
            'test_id': 'custom_0',
            **baseline
        })
    
    return configs

def main():
    parser = argparse.ArgumentParser(description='Comprehensive HKNN test suite')
    parser.add_argument('--data', required=True, help='Input data file (.fvecs)')
    parser.add_argument('--N', type=int, required=True, help='Number of points')
    parser.add_argument('--D', type=int, required=True, help='Dimension')
    parser.add_argument('--test-type', default='baseline',
                       choices=['baseline', 'K_sweep', 'perplexity_sweep', 'lr_sweep',
                               'epochs_sweep', 'mneg_sweep', 'gamma_sweep', 
                               'coarsening_sweep', 'comprehensive', 'custom'],
                       help='Type of test to run')
    parser.add_argument('--output-dir', default='test_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--skip-embedding', action='store_true',
                       help='Skip embedding generation, only analyze existing results')
    parser.add_argument('--max-tests', type=int, default=None,
                       help='Maximum number of tests to run (for quick testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save test configuration
    config_file = os.path.join(results_dir, 'test_config.json')
    with open(config_file, 'w') as f:
        json.dump({
            'test_type': args.test_type,
            'data_path': args.data,
            'N': args.N,
            'D': args.D,
            'seed': args.seed,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"Test Suite: {args.test_type}")
    print(f"Results directory: {results_dir}")
    print("="*70)
    
    # Generate test configurations
    configs = generate_test_configs(args.test_type)
    if args.max_tests:
        configs = configs[:args.max_tests]
    
    print(f"Running {len(configs)} test configurations...")
    
    results = []
    successful = 0
    failed = 0
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running test: {config['test_id']}")
        print(f"  Config: K={config['K']}, perp={config['perplexity']}, "
              f"lr={config['lr']}, epochs=({config['epochs_coarse']}, {config['epochs_fine']})")
        
        if not args.skip_embedding:
            # Run embedding
            result = run_embedding(config, args.data, args.N, args.D, results_dir, args.seed)
            
            if result['success']:
                print(f"  ✓ Embedding completed in {result['elapsed_time']:.2f}s")
                
                # Collect metrics
                print(f"  Collecting metrics...")
                metrics = collect_metrics(
                    result['output_path'],
                    args.data,
                    results_dir,
                    config['test_id']
                )
                
                if metrics:
                    result.update(metrics)
                    print(f"  ✓ Metrics collected")
                else:
                    print(f"  ⚠️  Could not collect metrics")
                
                successful += 1
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                failed += 1
        else:
            # Just try to load existing results
            embedding_path = os.path.join(results_dir, f"test_{config['test_id']}.csv")
            if os.path.exists(embedding_path):
                metrics = collect_metrics(embedding_path, args.data, results_dir, config['test_id'])
                result = {'success': True, 'test_id': config['test_id'], 'output_path': embedding_path}
                if metrics:
                    result.update(metrics)
                successful += 1
            else:
                print(f"  ⚠️  Embedding file not found: {embedding_path}")
                failed += 1
                result = {'success': False, 'test_id': config['test_id']}
        
        result['config'] = config
        results.append(result)
        
        # Save intermediate results
        results_file = os.path.join(results_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Generate summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    print(f"Total tests: {len(configs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {results_dir}")
    print(f"  - results.json: Full results")
    print(f"  - test_config.json: Test configuration")
    print(f"  - metrics_*.json: Individual metric files")
    print(f"  - test_*.csv: Embedding outputs")
    
    # Generate CSV summary for easy analysis
    csv_file = os.path.join(results_dir, 'results_summary.csv')
    generate_summary_csv(results, csv_file)
    print(f"  - results_summary.csv: CSV summary for analysis")
    
    print("\nRun 'python results_analyzer.py' to analyze results.")

def generate_summary_csv(results, output_file):
    """Generate CSV summary of all results."""
    if not results:
        return
    
    # Extract all unique metric keys
    all_keys = set()
    for r in results:
        if r.get('success'):
            all_keys.update(r.keys())
    
    # Remove non-metric keys
    exclude_keys = {'success', 'test_id', 'config', 'output_path', 'error', 'timing_info'}
    metric_keys = sorted([k for k in all_keys if k not in exclude_keys])
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header: config params + metrics
        header = [
            'test_id', 'K', 'perplexity', 'lr', 'epochs_coarse', 'epochs_fine',
            'mneg', 'gamma', 'kml', 'rho', 'elapsed_time', 'success'
        ] + metric_keys
        writer.writerow(header)
        
        # Rows
        for r in results:
            config = r.get('config', {})
            row = [
                r.get('test_id', ''),
                config.get('K', ''),
                config.get('perplexity', ''),
                config.get('lr', ''),
                config.get('epochs_coarse', ''),
                config.get('epochs_fine', ''),
                config.get('mneg', ''),
                config.get('gamma', ''),
                config.get('kml', ''),
                config.get('rho', ''),
                r.get('elapsed_time', ''),
                '1' if r.get('success') else '0'
            ] + [r.get(k, '') for k in metric_keys]
            writer.writerow(row)

if __name__ == '__main__':
    main()

