#!/usr/bin/env python3
"""
Batch Experiment Runner for PoSyg Simulations
Author: Daniil Krizhanovskyi
Date: June 2025

This script runs multiple simulation experiments in parallel and aggregates results.
"""

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import argparse
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import using a relative import approach
import importlib.util
import sys

# Dynamically load the module with hyphen in filename
spec = importlib.util.spec_from_file_location("posyg_simulation", 
    os.path.join(os.path.dirname(__file__), "posyg-simulation.py"))
posyg_simulation = importlib.util.module_from_spec(spec)
sys.modules["posyg_simulation"] = posyg_simulation
spec.loader.exec_module(posyg_simulation)

# Import the required classes
PoSygSimulation = posyg_simulation.PoSygSimulation
SimulationConfig = posyg_simulation.SimulationConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Manages and runs simulation experiments"""
    
    def __init__(self, config_file: str, output_dir: str = "experiment_results"):
        self.config_file = config_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load experiment configurations
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            
        # Handle different config formats
        if 'experiments' in config_data:
            # Multi-experiment format
            self.experiments = config_data['experiments']
        elif 'config' in config_data:
            # Single experiment format
            experiment_name = Path(config_file).stem
            self.experiments = {experiment_name: config_data['config']}
        else:
            # Assume the entire file is a single experiment config
            experiment_name = Path(config_file).stem
            self.experiments = {experiment_name: config_data}
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
    def run_single_experiment(self, name: str, config_dict: Dict) -> Tuple[str, Dict]:
        """Run a single experiment and return results"""
        logger.info(f"Starting experiment: {name}")
        start_time = time.time()
        
        try:
            # Create simulation config
            config = SimulationConfig(**config_dict)
            
            # Run simulation
            sim = PoSygSimulation(config)
            metrics_df = sim.run_simulation()
            
            # Export results
            experiment_dir = self.run_dir / name
            experiment_dir.mkdir(exist_ok=True)
            sim.export_results(str(experiment_dir))
            
            # Calculate summary statistics
            summary = {
                "experiment": name,
                "status": "success",
                "duration": time.time() - start_time,
                "epochs_completed": len(metrics_df),
                "final_active_validators": metrics_df['active_validators'].iloc[-1],
                "avg_finality_rate": metrics_df['finality_rate'].mean(),
                "std_finality_rate": metrics_df['finality_rate'].std(),
                "final_gini": metrics_df['gini_coefficient'].iloc[-1],
                "max_gini": metrics_df['gini_coefficient'].max(),
                "total_slashing": metrics_df['slashing_events'].iloc[-1],
                "avg_governance_participation": metrics_df['governance_participation'].mean(),
                "final_byzantine_control": metrics_df['byzantine_stake_ratio'].iloc[-1],
                "final_cartel_control": metrics_df['cartel_control'].iloc[-1]
            }
            
            logger.info(f"Completed experiment: {name} in {summary['duration']:.2f}s")
            return name, summary
            
        except Exception as e:
            logger.error(f"Failed experiment: {name} - {str(e)}")
            return name, {
                "experiment": name,
                "status": "failed",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def run_experiments_parallel(self, max_workers: int = None):
        """Run all experiments in parallel"""
        if max_workers is None:
            max_workers = mp.cpu_count() - 1
        
        logger.info(f"Running experiments with {max_workers} workers")
        
        # Prepare experiment list
        experiment_list = []
        for name, exp_config in self.experiments.items():
            if 'variants' in exp_config:
                # Handle variants
                for variant in exp_config['variants']:
                    variant_name = f"{name}_{variant['name']}"
                    # Merge base config with variant config
                    base_config = self.experiments.get('baseline', {}).get('config', {})
                    variant_config = {**base_config, **variant['config']}
                    experiment_list.append((variant_name, variant_config))
            else:
                # Regular experiment
                experiment_list.append((name, exp_config['config']))
        
        # Run experiments in parallel
        with mp.Pool(max_workers) as pool:
            results = pool.starmap(self.run_single_experiment, experiment_list)
        
        # Aggregate results
        summary_df = pd.DataFrame([result[1] for result in results])
        summary_df.to_csv(self.run_dir / "experiment_summary.csv", index=False)
        
        # Generate report
        self.generate_report(summary_df)
        
        return summary_df
    
    def generate_report(self, summary_df: pd.DataFrame):
        """Generate a comprehensive report of all experiments"""
        report_path = self.run_dir / "experiment_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# PoSyg Experiment Results Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- Total experiments: {len(summary_df)}\n")
            f.write(f"- Successful: {(summary_df['status'] == 'success').sum()}\n")
            f.write(f"- Failed: {(summary_df['status'] == 'failed').sum()}\n")
            f.write(f"- Total runtime: {summary_df['duration'].sum():.2f} seconds\n\n")
            
            # Results table
            f.write("## Experiment Results\n\n")
            
            # Filter columns for display
            display_cols = [
                'experiment', 'status', 'epochs_completed',
                'final_active_validators', 'avg_finality_rate',
                'final_gini', 'total_slashing'
            ]
            
            # Convert to markdown table
            success_df = summary_df[summary_df['status'] == 'success'][display_cols]
            f.write(success_df.to_markdown(index=False, floatfmt='.3f'))
            f.write("\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Find best/worst performers
            if len(success_df) > 0:
                best_finality = success_df.loc[success_df['avg_finality_rate'].idxmax()]
                worst_finality = success_df.loc[success_df['avg_finality_rate'].idxmin()]
                
                f.write(f"### Best Finality Rate\n")
                f.write(f"- Experiment: {best_finality['experiment']}\n")
                f.write(f"- Rate: {best_finality['avg_finality_rate']:.3f}\n\n")
                
                f.write(f"### Worst Finality Rate\n")
                f.write(f"- Experiment: {worst_finality['experiment']}\n")
                f.write(f"- Rate: {worst_finality['avg_finality_rate']:.3f}\n\n")
                
                # Centralization analysis
                high_gini = success_df[success_df['final_gini'] > 0.8]
                if len(high_gini) > 0:
                    f.write(f"### High Centralization Risk\n")
                    f.write(f"Experiments with Gini > 0.8:\n")
                    for _, exp in high_gini.iterrows():
                        f.write(f"- {exp['experiment']}: {exp['final_gini']:.3f}\n")
                    f.write("\n")
            
            # Failed experiments
            failed_df = summary_df[summary_df['status'] == 'failed']
            if len(failed_df) > 0:
                f.write("## Failed Experiments\n\n")
                for _, exp in failed_df.iterrows():
                    f.write(f"- {exp['experiment']}: {exp['error']}\n")
        
        logger.info(f"Report generated: {report_path}")
    
    def analyze_parameter_sensitivity(self):
        """Analyze sensitivity to parameter changes"""
        # Load all successful experiments
        results = []
        for exp_dir in self.run_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name != "analysis":
                try:
                    metrics_path = exp_dir / "metrics_history.csv"
                    config_path = exp_dir / "simulation_config.json"
                    
                    if metrics_path.exists() and config_path.exists():
                        metrics = pd.read_csv(metrics_path)
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        # Add experiment name and config to metrics
                        summary = {
                            "experiment": exp_dir.name,
                            **config,
                            "avg_finality": metrics['finality_rate'].mean(),
                            "final_gini": metrics['gini_coefficient'].iloc[-1],
                            "total_slashing": metrics['slashing_events'].iloc[-1]
                        }
                        results.append(summary)
                except Exception as e:
                    logger.warning(f"Could not load {exp_dir.name}: {e}")
        
        if not results:
            logger.warning("No results to analyze")
            return
        
        # Create analysis dataframe
        analysis_df = pd.DataFrame(results)
        
        # Perform correlation analysis
        param_cols = ['stake_weight', 'activity_weight', 'governance_weight',
                      'byzantine_ratio', 'cartel_size', 'slashing_rate']
        metric_cols = ['avg_finality', 'final_gini', 'total_slashing']
        
        # Filter available columns
        param_cols = [c for c in param_cols if c in analysis_df.columns]
        metric_cols = [c for c in metric_cols if c in analysis_df.columns]
        
        if param_cols and metric_cols:
            correlations = analysis_df[param_cols + metric_cols].corr()
            
            # Save correlation matrix
            correlations.to_csv(self.run_dir / "parameter_correlations.csv")
            
            logger.info("Parameter sensitivity analysis completed")
            
            # Find strongest correlations
            strong_corr = []
            for param in param_cols:
                for metric in metric_cols:
                    corr = correlations.loc[param, metric]
                    if abs(corr) > 0.5:
                        strong_corr.append((param, metric, corr))
            
            if strong_corr:
                logger.info("Strong correlations found:")
                for param, metric, corr in strong_corr:
                    logger.info(f"  {param} <-> {metric}: {corr:.3f}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run PoSyg simulation experiments")
    parser.add_argument(
        "--config",
        default="experiment_config.json",
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--output",
        default="experiment_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run parameter sensitivity analysis"
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = ExperimentRunner(args.config, args.output)
    
    # Run experiments
    logger.info("Starting experiment batch")
    summary_df = runner.run_experiments_parallel(args.workers)
    
    # Print summary
    print("\nExperiment Summary:")
    print("=" * 80)
    print(f"Total experiments: {len(summary_df)}")
    print(f"Successful: {(summary_df['status'] == 'success').sum()}")
    print(f"Failed: {(summary_df['status'] == 'failed').sum()}")
    print(f"Results saved to: {runner.run_dir}")
    
    # Run analysis if requested
    if args.analyze:
        logger.info("Running parameter sensitivity analysis")
        runner.analyze_parameter_sensitivity()
    
    print("\nExperiment batch completed!")

if __name__ == "__main__":
    main()