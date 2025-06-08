#!/usr/bin/env python3
"""
PoSyg Agent-Based Simulation Framework
Author: Daniil Krizhanovskyi
Date: October 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import json
import hashlib
import random
from collections import defaultdict
import networkx as nx
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentStrategy(Enum):
    """Validator behavior strategies"""
    HONEST = "honest"
    OPPORTUNISTIC = "opportunistic"
    LAZY = "lazy"
    BYZANTINE = "byzantine"
    CARTEL_MEMBER = "cartel_member"
    ADAPTIVE = "adaptive"

class BlockStatus(Enum):
    """Block finalization status"""
    PROPOSED = "proposed"
    ATTESTED = "attested"
    FINALIZED = "finalized"
    ORPHANED = "orphaned"

@dataclass
class ValidatorState:
    """Complete state of a validator"""
    id: str
    stake: float
    synergy_score: float = 0.0
    blocks_proposed: int = 0
    blocks_attested: int = 0
    governance_votes: int = 0
    slashing_count: int = 0
    is_active: bool = True
    reputation_history: List[float] = field(default_factory=list)
    last_active_epoch: int = 0
    strategy: AgentStrategy = AgentStrategy.HONEST
    cartel_id: Optional[str] = None
    
    def update_reputation(self, score: float):
        """Track reputation history with sliding window"""
        self.reputation_history.append(score)
        if len(self.reputation_history) > 100:  # Keep last 100 epochs
            self.reputation_history.pop(0)

@dataclass
class Block:
    """Block structure for consensus"""
    height: int
    epoch: int
    proposer: str
    parent_hash: str
    transactions: List[str]
    timestamp: float
    status: BlockStatus = BlockStatus.PROPOSED
    attestations: Set[str] = field(default_factory=set)
    
    @property
    def hash(self) -> str:
        """Compute block hash"""
        data = f"{self.height}{self.proposer}{self.parent_hash}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

@dataclass
class SimulationConfig:
    """Simulation parameters"""
    # Network parameters
    num_validators: int = 100
    initial_stake_distribution: str = "pareto"  # uniform, pareto, normal
    
    # Consensus parameters
    block_time: float = 6.0  # seconds
    epoch_length: int = 32  # blocks per epoch
    finality_threshold: float = 0.67  # 2/3 + 1
    
    # Synergy score weights
    stake_weight: float = 0.4
    activity_weight: float = 0.4
    governance_weight: float = 0.2
    
    # Economic parameters
    base_reward: float = 10.0
    slashing_rate: float = 0.01
    max_inflation: float = 0.05
    
    # Attack parameters
    byzantine_ratio: float = 0.05
    cartel_size: float = 0.0
    sybil_nodes: int = 0
    
    # Simulation settings
    simulation_epochs: int = 1000
    random_seed: int = 42

class PoSygSimulation:
    """Main simulation engine for PoSyg consensus"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.validators: Dict[str, ValidatorState] = {}
        self.blocks: List[Block] = []
        self.current_epoch = 0
        self.current_height = 0
        self.metrics_history = []
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Initialize components
        self._initialize_validators()
        self._initialize_genesis_block()
        
    def _initialize_validators(self):
        """Create initial validator set with stake distribution"""
        logger.info(f"Initializing {self.config.num_validators} validators")
        
        # Generate stake distribution
        if self.config.initial_stake_distribution == "pareto":
            stakes = np.random.pareto(1.5, self.config.num_validators) * 1000
        elif self.config.initial_stake_distribution == "normal":
            stakes = np.abs(np.random.normal(1000, 300, self.config.num_validators))
        else:  # uniform
            stakes = np.random.uniform(100, 2000, self.config.num_validators)
        
        # Normalize stakes
        stakes = stakes / stakes.sum() * 1000000  # Total stake = 1M
        
        # Assign strategies
        strategies = self._assign_strategies()
        
        # Create validators
        for i in range(self.config.num_validators):
            validator_id = f"val_{i:04d}"
            self.validators[validator_id] = ValidatorState(
                id=validator_id,
                stake=stakes[i],
                strategy=strategies[i],
                cartel_id=self._assign_cartel(i, strategies[i])
            )
    
    def _assign_strategies(self) -> List[AgentStrategy]:
        """Assign strategies based on configuration"""
        n = self.config.num_validators
        strategies = []
        
        # Calculate counts for each strategy
        n_byzantine = int(n * self.config.byzantine_ratio)
        n_cartel = int(n * self.config.cartel_size)
        n_lazy = int(n * 0.1)  # 10% lazy validators
        n_opportunistic = int(n * 0.1)
        n_adaptive = int(n * 0.05)
        n_honest = n - n_byzantine - n_cartel - n_lazy - n_opportunistic - n_adaptive
        
        # Build strategy list
        strategies.extend([AgentStrategy.HONEST] * n_honest)
        strategies.extend([AgentStrategy.BYZANTINE] * n_byzantine)
        strategies.extend([AgentStrategy.CARTEL_MEMBER] * n_cartel)
        strategies.extend([AgentStrategy.LAZY] * n_lazy)
        strategies.extend([AgentStrategy.OPPORTUNISTIC] * n_opportunistic)
        strategies.extend([AgentStrategy.ADAPTIVE] * n_adaptive)
        
        # Shuffle to randomize positions
        random.shuffle(strategies)
        return strategies
    
    def _assign_cartel(self, validator_idx: int, strategy: AgentStrategy) -> Optional[str]:
        """Assign cartel membership"""
        if strategy == AgentStrategy.CARTEL_MEMBER:
            # Simple cartel assignment - could be more sophisticated
            cartel_count = max(1, int(self.config.num_validators * self.config.cartel_size / 10))
            return f"cartel_{validator_idx % cartel_count}"
        return None
    
    def _initialize_genesis_block(self):
        """Create genesis block"""
        genesis = Block(
            height=0,
            epoch=0,
            proposer="genesis",
            parent_hash="0" * 16,
            transactions=[],
            timestamp=0.0,
            status=BlockStatus.FINALIZED
        )
        self.blocks.append(genesis)
    
    def compute_synergy_score(self, validator_id: str) -> float:
        """Calculate multi-dimensional synergy score"""
        validator = self.validators[validator_id]
        
        # Normalize components
        total_stake = sum(v.stake for v in self.validators.values())
        stake_score = validator.stake / total_stake
        
        # Activity score (blocks proposed + attested)
        total_blocks = max(1, self.current_height)
        activity_score = (validator.blocks_proposed + validator.blocks_attested * 0.1) / total_blocks
        
        # Governance participation
        total_votes = max(1, self.current_epoch)  # Assume one vote opportunity per epoch
        governance_score = validator.governance_votes / total_votes
        
        # Apply slashing penalties
        slashing_penalty = (1 - self.config.slashing_rate) ** validator.slashing_count
        
        # Weighted combination
        synergy = (
            self.config.stake_weight * stake_score +
            self.config.activity_weight * activity_score +
            self.config.governance_weight * governance_score
        ) * slashing_penalty
        
        # Update validator's synergy score
        validator.synergy_score = synergy * 1000  # Scale to 0-1000
        validator.update_reputation(validator.synergy_score)
        
        return validator.synergy_score
    
    def select_block_proposer(self) -> str:
        """Select proposer based on synergy scores"""
        # Update all synergy scores
        for validator_id in self.validators:
            if self.validators[validator_id].is_active:
                self.compute_synergy_score(validator_id)
        
        # Filter active validators
        active_validators = [
            (v_id, v.synergy_score) 
            for v_id, v in self.validators.items() 
            if v.is_active
        ]
        
        if not active_validators:
            raise RuntimeError("No active validators")
        
        # Weighted random selection based on synergy scores
        validator_ids, scores = zip(*active_validators)
        scores = np.array(scores)
        
        # Add small epsilon to avoid zero probabilities
        scores = scores + 0.01
        probabilities = scores / scores.sum()
        
        return np.random.choice(validator_ids, p=probabilities)
    
    def simulate_block_production(self) -> Block:
        """Simulate production of a new block"""
        proposer = self.select_block_proposer()
        self.current_height += 1
        
        # Create new block
        parent_hash = self.blocks[-1].hash if self.blocks else "0" * 16
        block = Block(
            height=self.current_height,
            epoch=self.current_epoch,
            proposer=proposer,
            parent_hash=parent_hash,
            transactions=self._generate_transactions(),
            timestamp=self.current_height * self.config.block_time
        )
        
        # Update proposer stats
        self.validators[proposer].blocks_proposed += 1
        self.validators[proposer].last_active_epoch = self.current_epoch
        
        return block
    
    def simulate_attestations(self, block: Block) -> bool:
        """Simulate validator attestations for a block"""
        for validator_id, validator in self.validators.items():
            if not validator.is_active or validator_id == block.proposer:
                continue
            
            # Decision logic based on strategy
            should_attest = self._make_attestation_decision(validator, block)
            
            if should_attest:
                block.attestations.add(validator_id)
                validator.blocks_attested += 1
                validator.last_active_epoch = self.current_epoch
        
        # Check if block reaches finality
        attestation_weight = sum(
            self.validators[v_id].synergy_score 
            for v_id in block.attestations
        )
        total_weight = sum(
            v.synergy_score 
            for v in self.validators.values() 
            if v.is_active
        )
        
        finality_achieved = attestation_weight >= (total_weight * self.config.finality_threshold)
        
        if finality_achieved:
            block.status = BlockStatus.FINALIZED
        else:
            block.status = BlockStatus.ORPHANED
            # Slash proposer for failed block
            self._apply_slashing(block.proposer, "failed_block")
        
        return finality_achieved
    
    def _make_attestation_decision(self, validator: ValidatorState, block: Block) -> bool:
        """Determine if validator should attest based on strategy"""
        if validator.strategy == AgentStrategy.HONEST:
            return True
        
        elif validator.strategy == AgentStrategy.BYZANTINE:
            # Byzantine validators randomly attest or try to cause forks
            return random.random() < 0.3
        
        elif validator.strategy == AgentStrategy.LAZY:
            # Lazy validators only attest occasionally
            return random.random() < 0.5
        
        elif validator.strategy == AgentStrategy.OPPORTUNISTIC:
            # Opportunistic validators check if attestation is profitable
            # (simplified logic - could be more sophisticated)
            return random.random() < 0.8
        
        elif validator.strategy == AgentStrategy.CARTEL_MEMBER:
            # Cartel members coordinate
            cartel_proposer = any(
                self.validators[v_id].cartel_id == validator.cartel_id
                for v_id in [block.proposer]
            )
            return cartel_proposer or random.random() < 0.7
        
        elif validator.strategy == AgentStrategy.ADAPTIVE:
            # Adaptive validators learn from history
            # Simplified: attest if recent blocks were successful
            recent_success_rate = self._calculate_recent_finality_rate()
            return random.random() < recent_success_rate
        
        return True
    
    def _calculate_recent_finality_rate(self, window: int = 10) -> float:
        """Calculate recent block finality rate"""
        if len(self.blocks) < window:
            return 0.8  # Default rate
        
        recent_blocks = self.blocks[-window:]
        finalized = sum(1 for b in recent_blocks if b.status == BlockStatus.FINALIZED)
        return finalized / window
    
    def _generate_transactions(self) -> List[str]:
        """Generate mock transactions for a block"""
        num_txs = np.random.poisson(100)  # Average 100 txs per block
        return [f"tx_{i:06d}" for i in range(num_txs)]
    
    def _apply_slashing(self, validator_id: str, reason: str):
        """Apply slashing penalty to validator"""
        validator = self.validators[validator_id]
        validator.slashing_count += 1
        
        # Progressive slashing
        penalty_multiplier = min(validator.slashing_count, 10)
        stake_penalty = validator.stake * self.config.slashing_rate * penalty_multiplier
        
        validator.stake = max(0, validator.stake - stake_penalty)
        
        # Deactivate if stake too low
        if validator.stake < 100:  # Minimum stake threshold
            validator.is_active = False
        
        logger.debug(f"Slashed {validator_id} for {reason}: -{stake_penalty:.2f} stake")
    
    def simulate_governance_round(self):
        """Simulate governance participation"""
        # Generate a governance proposal
        proposal_id = f"prop_{self.current_epoch}"
        
        for validator_id, validator in self.validators.items():
            if not validator.is_active:
                continue
            
            # Decision to participate in governance
            participate = self._make_governance_decision(validator)
            
            if participate:
                validator.governance_votes += 1
    
    def _make_governance_decision(self, validator: ValidatorState) -> bool:
        """Determine if validator participates in governance"""
        if validator.strategy == AgentStrategy.LAZY:
            return random.random() < 0.1
        elif validator.strategy == AgentStrategy.BYZANTINE:
            return random.random() < 0.3
        elif validator.strategy == AgentStrategy.CARTEL_MEMBER:
            # Cartels coordinate governance
            return True
        else:
            # Most validators participate
            return random.random() < 0.8
    
    def collect_metrics(self) -> Dict:
        """Collect current simulation metrics"""
        active_validators = [v for v in self.validators.values() if v.is_active]
        
        # Calculate Gini coefficient for stake distribution
        stakes = sorted([v.stake for v in active_validators])
        n = len(stakes)
        if n == 0:
            gini = 0
        else:
            cumsum = np.cumsum(stakes)
            gini = (2 * np.sum((np.arange(1, n+1) * stakes))) / (n * cumsum[-1]) - (n + 1) / n
        
        # Calculate other metrics
        metrics = {
            "epoch": self.current_epoch,
            "height": self.current_height,
            "active_validators": len(active_validators),
            "total_stake": sum(v.stake for v in active_validators),
            "gini_coefficient": gini,
            "avg_synergy_score": np.mean([v.synergy_score for v in active_validators]) if active_validators else 0,
            "finality_rate": self._calculate_recent_finality_rate(32),
            "slashing_events": sum(v.slashing_count for v in self.validators.values()),
            "governance_participation": sum(v.governance_votes for v in active_validators) / max(1, len(active_validators) * max(1, self.current_epoch)),
            "cartel_control": sum(v.stake for v in active_validators if v.strategy == AgentStrategy.CARTEL_MEMBER) / max(1, sum(v.stake for v in active_validators)),
            "byzantine_stake_ratio": sum(v.stake for v in active_validators if v.strategy == AgentStrategy.BYZANTINE) / max(1, sum(v.stake for v in active_validators))
        }
        
        return metrics
    
    def run_simulation(self) -> pd.DataFrame:
        """Run complete simulation"""
        logger.info(f"Starting PoSyg simulation for {self.config.simulation_epochs} epochs")
        
        for epoch in range(self.config.simulation_epochs):
            self.current_epoch = epoch
            
            # Simulate blocks in this epoch
            for block_in_epoch in range(self.config.epoch_length):
                # Produce block
                block = self.simulate_block_production()
                
                # Simulate attestations
                finalized = self.simulate_attestations(block)
                
                # Add block to chain
                self.blocks.append(block)
                
                # Log progress
                if self.current_height % 100 == 0:
                    logger.info(f"Height: {self.current_height}, Epoch: {epoch}, Finalized: {finalized}")
            
            # Governance round at epoch boundary
            self.simulate_governance_round()
            
            # Collect metrics
            metrics = self.collect_metrics()
            self.metrics_history.append(metrics)
            
            # Check for simulation termination conditions
            if metrics["active_validators"] < 10:
                logger.warning("Too few active validators, ending simulation")
                break
        
        logger.info("Simulation complete")
        return pd.DataFrame(self.metrics_history)
    
    def export_results(self, output_dir: str = "simulation_results"):
        """Export simulation results to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export metrics
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(f"{output_dir}/metrics_history.csv", index=False)
        
        # Export validator states
        validator_data = []
        for v_id, validator in self.validators.items():
            validator_data.append({
                "id": v_id,
                "final_stake": validator.stake,
                "final_synergy_score": validator.synergy_score,
                "blocks_proposed": validator.blocks_proposed,
                "blocks_attested": validator.blocks_attested,
                "governance_votes": validator.governance_votes,
                "slashing_count": validator.slashing_count,
                "strategy": validator.strategy.value,
                "is_active": validator.is_active,
                "reputation_mean": np.mean(validator.reputation_history) if validator.reputation_history else 0,
                "reputation_std": np.std(validator.reputation_history) if validator.reputation_history else 0
            })
        
        validator_df = pd.DataFrame(validator_data)
        validator_df.to_csv(f"{output_dir}/final_validator_states.csv", index=False)
        
        # Export configuration
        config_dict = {
            "num_validators": self.config.num_validators,
            "simulation_epochs": self.config.simulation_epochs,
            "stake_weight": self.config.stake_weight,
            "activity_weight": self.config.activity_weight,
            "governance_weight": self.config.governance_weight,
            "byzantine_ratio": self.config.byzantine_ratio,
            "cartel_size": self.config.cartel_size,
            "finality_threshold": self.config.finality_threshold
        }
        
        with open(f"{output_dir}/simulation_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Results exported to {output_dir}")

# Simulation execution
if __name__ == "__main__":
    # Define simulation scenarios
    scenarios = [
        {
            "name": "baseline",
            "config": SimulationConfig(
                num_validators=100,
                byzantine_ratio=0.05,
                cartel_size=0.0,
                simulation_epochs=100
            )
        },
        {
            "name": "cartel_attack",
            "config": SimulationConfig(
                num_validators=100,
                byzantine_ratio=0.05,
                cartel_size=0.3,
                simulation_epochs=100
            )
        },
        {
            "name": "high_byzantine",
            "config": SimulationConfig(
                num_validators=100,
                byzantine_ratio=0.33,
                cartel_size=0.0,
                simulation_epochs=100
            )
        }
    ]
    
    # Run scenarios
    for scenario in scenarios:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running scenario: {scenario['name']}")
        logger.info(f"{'='*50}")
        
        sim = PoSygSimulation(scenario["config"])
        metrics_df = sim.run_simulation()
        sim.export_results(f"results_{scenario['name']}")
        
        # Print summary statistics
        print(f"\nScenario: {scenario['name']}")
        print(f"Final active validators: {metrics_df['active_validators'].iloc[-1]}")
        print(f"Average finality rate: {metrics_df['finality_rate'].mean():.2%}")
        print(f"Final Gini coefficient: {metrics_df['gini_coefficient'].iloc[-1]:.3f}")
        print(f"Total slashing events: {metrics_df['slashing_events'].iloc[-1]}")
