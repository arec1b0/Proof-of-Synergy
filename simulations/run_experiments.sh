#!/bin/bash

# PoSyg Experiment Runner Script
# This script provides a user-friendly interface to run different PoSyg simulation experiments

# Default values
CONFIG_DIR="config"
OUTPUT_DIR="../experiment_results"
WORKERS=$(($(nproc) - 1))
ANALYZE=false

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage information
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c, --config-dir DIR    Directory containing experiment configs (default: config/)"
    echo "  -o, --output-dir DIR    Output directory for results (default: ../experiment_results/)"
    echo "  -w, --workers NUM       Number of parallel workers (default: CPU cores - 1)"
    echo "  -a, --analyze           Run parameter sensitivity analysis after experiments"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Available experiment configurations:"
    echo "  baseline              Baseline scenario with honest majority"
    echo "  attacks               All attack scenarios (cartel, byzantine, sybil)"
    echo "  stress                Large-scale network stress test"
    echo "  weights               Different weight configurations"
    echo "  economic              Economic parameter variations"
    echo "  all                   Run all experiment configurations"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -a|--analyze)
            ANALYZE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            EXPERIMENT="$1"
            shift
            ;;
    esac
done

# Check if experiment is specified
if [ -z "$EXPERIMENT" ]; then
    echo -e "${YELLOW}No experiment specified. Running all experiments.${NC}"
    EXPERIMENT="all"
fi

# Function to run a specific experiment
run_experiment() {
    local exp_name="$1"
    local config_file="$2"
    
    echo -e "\n${GREEN}=== Running $exp_name experiment ===${NC}"
    echo "Config: $config_file"
    echo "Workers: $WORKERS"
    echo "Output dir: $OUTPUT_DIR"
    
    python3 posyg-batch-runner.py \
        --config "$config_file" \
        --output "$OUTPUT_DIR" \
        --workers "$WORKERS" \
        $([ "$ANALYZE" = true ] && echo "--analyze")
}

# Main execution
cd "$(dirname "$0")"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

case $EXPERIMENT in
    baseline)
        run_experiment "Baseline" "$CONFIG_DIR/baseline.json"
        ;;
        
    attacks)
        run_experiment "Attack Scenarios" "$CONFIG_DIR/attacks.json"
        ;;
        
    stress)
        run_experiment "Stress Test" "$CONFIG_DIR/stress_test.json"
        ;;
        
    weights)
        run_experiment "Weight Configurations" "$CONFIG_DIR/adaptive_weights.json"
        ;;
        
    economic)
        run_experiment "Economic Scenarios" "$CONFIG_DIR/economic_scenarios.json"
        ;;
        
    all)
        # Run all experiments in sequence
        run_experiment "Baseline" "$CONFIG_DIR/baseline.json"
        run_experiment "Attack Scenarios" "$CONFIG_DIR/attacks.json"
        run_experiment "Stress Test" "$CONFIG_DIR/stress_test.json"
        run_experiment "Weight Configurations" "$CONFIG_DIR/adaptive_weights.json"
        run_experiment "Economic Scenarios" "$CONFIG_DIR/economic_scenarios.json"
        ;;
        
    *)
        echo -e "${YELLOW}Unknown experiment: $EXPERIMENT${NC}"
        show_help
        exit 1
        ;;
esac

echo -e "\n${GREEN}All experiments completed!${NC}"
echo "Results are available in: $OUTPUT_DIR"