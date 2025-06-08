# PoSyg Consensus Research Data


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15617401.svg)](https://doi.org/10.5281/zenodo.15617401)


## Overview

This repository contains the research data, simulation code, and analysis tools for the Proof of Synergy (PoSyg) consensus protocol doctoral research.

## Directory Structure

```
posyg-research/
├── simulations/
│   ├── posyg_simulation.py      # Main Python simulation framework
│   ├── run_experiments.sh       # Batch experiment runner
│   └── config/
│       ├── baseline.json        # Baseline scenario configuration
│       ├── attacks.json         # Attack scenario configurations
│       └── stress_test.json     # Stress test parameters
├── specifications/
│   ├── PoSyg.tla               # TLA+ formal specification
│   ├── PoSygMC.tla             # Model checking configuration
│   └── invariants.tla          # Additional invariants
├── analysis/
│   ├── posyg_analysis.ipynb    # Main analysis notebook
│   ├── attack_analysis.ipynb   # Attack resistance analysis
│   └── economic_analysis.ipynb # Economic model analysis
├── results/
│   ├── baseline/
│   │   ├── metrics_history.csv
│   │   ├── final_validator_states.csv
│   │   └── simulation_config.json
│   ├── cartel_attack/
│   │   └── [simulation output files]
│   ├── high_byzantine/
│   │   └── [simulation output files]
│   └── summary_statistics.csv
├── docs/
│   ├── usage_guide.md          # How to run simulations
│   ├── metrics_glossary.md     # Explanation of metrics
│   └── tla_guide.md           # TLA+ specification guide
└── requirements.txt            # Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- networkx>=2.6
- scipy>=1.7.0
- jupyter>=1.0.0

### 2. Run Basic Simulation

```bash
python simulations/posyg_simulation.py
```

This will run three default scenarios:
- **baseline**: Normal network conditions
- **cartel_attack**: 30% cartel coordination
- **high_byzantine**: 33% Byzantine validators

### 3. Analyze Results

```bash
jupyter notebook analysis/posyg_analysis.ipynb
```

## Simulation Parameters

### Key Configuration Options

```python
SimulationConfig(
    # Network parameters
    num_validators=100,              # Number of validators
    initial_stake_distribution="pareto",  # Stake distribution type
    
    # Consensus parameters
    block_time=6.0,                  # Seconds per block
    epoch_length=32,                 # Blocks per epoch
    finality_threshold=0.67,         # 2/3 + 1 Byzantine threshold
    
    # Synergy score weights
    stake_weight=0.4,                # Weight for stake component
    activity_weight=0.4,             # Weight for activity component
    governance_weight=0.2,           # Weight for governance component
    
    # Attack parameters
    byzantine_ratio=0.05,            # Fraction of Byzantine validators
    cartel_size=0.0,                 # Fraction in coordinated cartel
    sybil_nodes=0,                   # Number of Sybil identities
    
    # Simulation settings
    simulation_epochs=1000,          # Duration of simulation
    random_seed=42                   # For reproducibility
)
```

### Validator Strategies

1. **HONEST**: Always follows protocol rules
2. **BYZANTINE**: Actively tries to disrupt consensus
3. **LAZY**: Minimal participation, occasional attestations
4. **OPPORTUNISTIC**: Defects when profitable
5. **CARTEL_MEMBER**: Coordinates with other cartel members
6. **ADAPTIVE**: Learns and adapts behavior based on network state

## Output Data Format

### metrics_history.csv

Time-series data with per-epoch metrics:

| Column | Description |
|--------|-------------|
| epoch | Current epoch number |
| height | Current blockchain height |
| active_validators | Number of active validators |
| total_stake | Sum of all active validator stakes |
| gini_coefficient | Stake distribution inequality (0-1) |
| avg_synergy_score | Mean synergy score across validators |
| finality_rate | Recent block finalization rate |
| slashing_events | Cumulative slashing count |
| governance_participation | Fraction participating in governance |
| cartel_control | Stake fraction controlled by cartel |
| byzantine_stake_ratio | Stake fraction held by Byzantine validators |

### final_validator_states.csv

Final state of each validator:

| Column | Description |
|--------|-------------|
| id | Validator identifier |
| final_stake | Remaining stake after simulation |
| final_synergy_score | Final synergy score (0-1000) |
| blocks_proposed | Total blocks proposed |
| blocks_attested | Total blocks attested |
| governance_votes | Governance participation count |
| slashing_count | Times slashed for violations |
| strategy | Validator behavior strategy |
| is_active | Whether validator is still active |
| reputation_mean | Average reputation over time |
| reputation_std | Reputation volatility |

## TLA+ Specification

### Running Model Checker

1. Install TLA+ Toolbox
2. Open `specifications/PoSyg.tla`
3. Create model with configuration from `PoSygMC.tla`
4. Check properties:
   - **TypeInvariant**: Type correctness
   - **SafetyInvariant**: No conflicting finalized blocks
   - **ByzantineFaultTolerance**: < 1/3 Byzantine control
   - **DecentralizationInvariant**: No validator > 33% power
   - **CartelResistance**: No small group majority

### Key Invariants

```tla
\* No two finalized blocks at same height
SafetyInvariant ==
    \A i, j \in 1..Len(blockchain) :
        (i # j /\ blockchain[i].height = blockchain[j].height) =>
        ~(blockchain[i].status = "finalized" /\ 
          blockchain[j].status = "finalized")

\* Byzantine fault tolerance maintained
ByzantineFaultTolerance ==
    byzantineStake < TotalStake * (1 - FinalizationThreshold)
```

## Analysis Notebooks

### 1. posyg_analysis.ipynb
- Network health metrics
- Validator strategy performance
- Synergy score dynamics
- Attack resistance analysis
- Economic analysis
- Statistical tests

### 2. attack_analysis.ipynb (TODO)
- Sybil attack scenarios
- Cartel coordination patterns
- Nothing-at-stake analysis
- Long-range attack resistance

### 3. economic_analysis.ipynb (TODO)
- Token distribution evolution
- Reward optimization
- Slashing economics
- Stability fund modeling

## Extending the Simulation

### Adding New Validator Strategies

```python
def _make_attestation_decision(self, validator: ValidatorState, block: Block) -> bool:
    if validator.strategy == AgentStrategy.YOUR_STRATEGY:
        # Implement custom decision logic
        return your_decision_logic(validator, block)
```

### Custom Attack Scenarios

```python
config = SimulationConfig(
    num_validators=200,
    byzantine_ratio=0.2,
    cartel_size=0.15,
    # Custom parameters
    your_parameter=value
)
```

### New Metrics

Add to `collect_metrics()` method:

```python
metrics = {
    # ... existing metrics ...
    "your_metric": calculate_your_metric(),
}
```

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `simulation_epochs` or `num_validators`
2. **Slow Simulation**: Use multiprocessing or reduce network size
3. **Missing Results**: Check file paths and permissions

### Performance Tips

- Use PyPy for 2-3x speedup
- Parallelize scenarios with multiprocessing
- Profile with cProfile for bottlenecks

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{krizhanovskyi2025posyg,
  title={A Multi-Dimensional Analysis of Proof of Synergy},
  author={Krizhanovskyi, Daniil},
  year={2025},
  institution={Independent Research}
}
```

## License

This research code is provided for academic use. See LICENSE file for details.

## Contact

- Author: Daniil Krizhanovskyi
- Email: daniil.krizhanovskyi@protonmail.ch
- Area of research: Distributed Systems, Blockchain Technology
