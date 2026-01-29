"""
Run transaction cost analysis.

Automatically extract baseline weights and run the analysis.
"""

import os
import argparse
import subprocess
import sys

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run transaction cost analysis')
    parser.add_argument('--results_path', type=str, default='0419_main_experiment_results',
                       help='Results directory used to extract baseline weights')
    parser.add_argument('--data_path', type=str, default='data',
                      help='Data path')
    parser.add_argument('--output_dir', type=str, default='transaction_cost_analysis_results',
                      help='Output directory')
    parser.add_argument('--start_date', type=str, default='2024-04-06',
                      help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-04-06',
                      help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial_capital', type=float, default=1000000,
                      help='Initial capital')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                      help='Config file path')
    args = parser.parse_args()
    
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure results path is correct (avoid duplication)
    if args.results_path.startswith('LIMPPO_CNN/'):
        args.results_path = args.results_path[len('LIMPPO_CNN/'):]
    
    # Convert relative path to absolute
    results_path = os.path.join(script_dir, args.results_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: extract baseline weights
    print("Step 1: Extract baseline weights...")
    weights_path = os.path.join(args.output_dir, 'baseline_weights.csv')
    
    # Build script path
    create_weights_script = os.path.join(script_dir, 'create_baseline_weights.py')
    print(f"Weight extraction script: {create_weights_script}")
    print(f"Results directory: {results_path}")
    
    extract_cmd = [
        sys.executable,
        create_weights_script,
        '--results_path', results_path,
        '--output_path', weights_path
    ]
    
    try:
        subprocess.run(extract_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract baseline weights: {str(e)}")
        print("Using uniform allocation as fallback...")
    
    # Step 2: run transaction cost analysis
    print("\nStep 2: Run transaction cost analysis...")
    
    # Ensure config directory exists
    config_dir = os.path.join(script_dir, 'config')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    
    # Update config to use extracted weights
    config_path = os.path.join(config_dir, 'transaction_cost_config.yaml')
    if not os.path.exists(config_path):
        # Copy default config
        try:
            import yaml
            
            # Read base config
            default_config_path = os.path.join(script_dir, args.config)
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add baseline config
            if 'baseline' not in config:
                config['baseline'] = {}
            
            # Only add if weights file exists
            if os.path.exists(weights_path):
                config['baseline']['weights_file'] = weights_path
            
            # Save new config
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                
            args.config = config_path
        except Exception as e:
            print(f"Failed to create transaction cost config: {str(e)}")
            # Continue with base config
    
    # Ensure analysis script path is correct
    analysis_script = os.path.join(script_dir, 'transaction_cost_analysis.py')
    print(f"Analysis script path: {analysis_script}")
    
    analysis_cmd = [
        sys.executable,
        analysis_script,
        '--data_path', args.data_path,
        '--output_dir', args.output_dir,
        '--start_date', args.start_date,
        '--end_date', args.end_date,
        '--initial_capital', str(args.initial_capital),
        '--config', args.config
    ]
    
    try:
        subprocess.run(analysis_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Transaction cost analysis failed: {str(e)}")
        return
    
    print(f"\nTransaction cost analysis complete! Results in: {args.output_dir}")
    print(f"- Final wealth data: {args.output_dir}/transaction_cost_analysis_final_wealth.csv")
    print(f"- Transaction cost data: {args.output_dir}/transaction_cost_analysis_transaction_cost.csv")
    print(f"- Full metrics data: {args.output_dir}/transaction_cost_analysis_all_metrics.csv")
    print(f"- Visualizations: {args.output_dir}/transaction_cost_vs_final_wealth.png")
    print("You can use these outputs to plot custom charts of cost impact.")

if __name__ == '__main__':
    main() 

