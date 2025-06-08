# Proof of Synergy (PoSyg) Usage Guide

This guide provides comprehensive instructions for running simulations, analyzing results, and working with the Proof of Synergy consensus protocol research framework.

## Project Structure

The PoSyg project is organized into the following directories:

- `/simulations`: Core simulation code and configuration files
- `/analysis`: Data analysis scripts and notebooks
- `/specifications`: TLA+ formal specifications
- `/results`: Simulation output data (generated during runs)
- `/docs`: Documentation files

## Prerequisites

### Required Software

- Python 3.8+
- TLA+ Toolbox (for formal verification)
- Jupyter Notebook (for analysis)

### Python Dependencies

Install required packages:

```bash
pip install numpy pandas matplotlib seaborn scipy networkx jupyter
```

## Running Simulations

### Configuration Files

Simulation parameters are defined in JSON configuration files located in `/simulations/config/`:

- `baseline.json`: Standard protocol parameters
- `attacks.json`: Various attack scenarios
- `stress_test.json`: Large-scale network testing
- `adaptive_weights.json`: Different synergy score weight configurations
- `economic_scenarios.json`: Economic parameter variations

### Using the Experiment Runner

The `run_experiments.sh` script provides a user-friendly interface for running simulations:

```bash
cd simulations
./run_experiments.sh [options] [experiment]
```

Options:
- `-c, --config-dir DIR`: Directory containing experiment configs (default: `config/`)
- `-o, --output-dir DIR`: Output directory for results (default: `../experiment_results/`)
- `-w, --workers NUM`: Number of parallel workers (default: CPU cores - 1)
- `-a, --analyze`: Run parameter sensitivity analysis after experiments
- `-h, --help`: Show help message

Available experiments:
- `baseline`: Baseline scenario with honest majority
- `attacks`: All attack scenarios (cartel, byzantine, sybil)
- `stress`: Large-scale network stress test
- `weights`: Different weight configurations
- `economic`: Economic parameter variations
- `all`: Run all experiment configurations

Example:
```bash
./run_experiments.sh -w 4 -o ../results attacks
```

### Creating Custom Configurations

To create a custom simulation configuration:

1. Copy an existing configuration file (e.g., `baseline.json`)
2. Modify parameters as needed
3. Save with a descriptive name in the `config` directory
4. Run using the `-c` option to specify your configuration

Example custom configuration:
```json
{
  "num_validators": 200,
  "initial_stake_distribution": "pareto",
  "block_time": 6.0,
  "epoch_length": 32,
  "finality_threshold": 0.67,
  "stake_weight": 0.4,
  "activity_weight": 0.4,
  "governance_weight": 0.2,
  "base_reward": 10.0,
  "slashing_rate": 0.01,
  "max_inflation": 0.05,
  "byzantine_ratio": 0.05,
  "cartel_size": 0.0,
  "sybil_nodes": 0,
  "simulation_epochs": 1000,
  "random_seed": 42
}
```

## Analyzing Results

### Using the Analysis Notebook

The Jupyter notebook in `/analysis/posyg_analysis.ipynb` provides comprehensive analysis tools:

1. Start Jupyter Notebook:
   ```bash
   cd analysis
   jupyter notebook
   ```

2. Open `posyg_analysis.ipynb`

3. Update the scenario paths if needed:
   ```python
   scenarios = ["baseline", "cartel_attack", "high_byzantine"]
   ```

4. Run all cells to generate visualizations and analysis

### Key Analysis Sections

The notebook is organized into sections:

1. **Network Health Analysis**: Active validators, finality rate, decentralization metrics
2. **Validator Strategy Analysis**: Performance comparison of different validator strategies
3. **Synergy Score Dynamics**: Distribution and evolution of synergy scores
4. **Attack Resistance Analysis**: Protocol behavior under various attack scenarios
5. **Economic Analysis**: Reward distribution and economic incentives
6. **Statistical Analysis**: Hypothesis testing and parameter correlations
7. **Summary and Recommendations**: Key findings and protocol improvement suggestions

### Customizing Analysis

To analyze specific aspects:

1. Modify the `load_simulation_data()` function to point to your result directories
2. Comment out or add visualization sections as needed
3. Adjust statistical tests for your specific research questions

## Formal Verification

### Using TLA+ Specifications

The TLA+ specifications in `/specifications/` formally define the protocol:

1. Install TLA+ Toolbox from [lamport.azurewebsites.net/tla/toolbox.html](https://lamport.azurewebsites.net/tla/toolbox.html)
2. Open the Toolbox and create a new specification pointing to `posyg.tla`
3. Create a model with parameters from `PosygMC.tla`
4. Run the model checker to verify properties

See the [TLA+ Guide](tla_guide.md) for detailed instructions.

## Advanced Usage

### Parameter Sensitivity Analysis

To understand how parameter changes affect protocol behavior:

1. Run the experiment with the `-a` flag:
   ```bash
   ./run_experiments.sh -a baseline
   ```

2. Review sensitivity analysis in the results directory

### Custom Validator Strategies

To implement custom validator strategies:

1. Add a new strategy to the `AgentStrategy` enum in `posyg-simulation.py`
2. Implement the strategy behavior in the relevant action methods
3. Update the strategy assignment logic in `_assign_strategies()`

### Batch Processing Results

For large-scale result processing:

```python
import os
import pandas as pd
import glob

# Process all results in a directory
result_dirs = glob.glob("results_*")
combined_data = {}

for dir_name in result_dirs:
    scenario = dir_name.replace("results_", "")
    metrics_file = os.path.join(dir_name, "metrics_history.csv")
    if os.path.exists(metrics_file):
        combined_data[scenario] = pd.read_csv(metrics_file)

# Perform cross-scenario analysis
# ...
```

## Troubleshooting

### Common Issues

1. **Missing Results**:
   - Check that the output directory exists and is writable
   - Verify that the simulation completed successfully

2. **Performance Issues**:
   - Reduce the number of validators or epochs
   - Increase the number of workers for parallel processing
   - Use a more powerful machine for large simulations

3. **Visualization Errors**:
   - Ensure all required Python packages are installed
   - Check that result files have the expected format

### Logging

Simulation logs are stored in the output directory. To increase log verbosity:

```python
# In posyg-simulation.py
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Contributing

To contribute to the PoSyg project:

1. Follow the modular structure
2. Document new parameters and metrics in the [Metrics Glossary](metrics_glossary.md)
3. Add tests for new functionality
4. Update documentation to reflect changes

## References

- [Proof of Synergy Whitepaper](https://example.com/posyg-whitepaper)
- [TLA+ Documentation](https://lamport.azurewebsites.net/tla/tla.html)
- [Agent-Based Modeling Best Practices](https://example.com/abm-practices)