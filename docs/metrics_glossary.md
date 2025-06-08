# Proof of Synergy (PoSyg) Metrics Glossary

This document provides a comprehensive reference for all metrics used in the PoSyg consensus protocol simulations, analysis, and formal verification.

## Core Protocol Metrics

### Synergy Score
- **Definition**: A composite score determining validator selection probability and rewards
- **Formula**: `(StakeWeight * stakeScore + ActivityWeight * activityScore + GovernanceWeight * governanceScore) * slashingPenalty * 1000`
- **Components**:
  - **Stake Score**: Normalized validator stake relative to total network stake
  - **Activity Score**: Participation rate in block proposal and attestation
  - **Governance Score**: Participation in protocol governance votes
  - **Slashing Penalty**: Exponential penalty based on validator's protocol violations
- **Units**: Dimensionless (0-1000)
- **Interpretation**: Higher scores indicate validators with better protocol alignment

### Finality Rate
- **Definition**: Percentage of proposed blocks that reach finalized status
- **Formula**: `finalizedBlocks / totalProposedBlocks`
- **Units**: Percentage (0-100%)
- **Interpretation**: Higher rates indicate more efficient consensus

### Block Time
- **Definition**: Average time between consecutive blocks
- **Units**: Seconds
- **Interpretation**: Lower times indicate faster transaction processing

### Epoch Length
- **Definition**: Number of blocks in a single epoch
- **Units**: Blocks
- **Interpretation**: Determines frequency of validator set updates and synergy score recalculations

## Network Health Metrics

### Active Validators
- **Definition**: Count of validators participating in consensus
- **Units**: Count
- **Interpretation**: Higher counts indicate better network decentralization

### Gini Coefficient
- **Definition**: Measure of stake distribution inequality
- **Formula**: Standard Gini coefficient calculation based on validator stakes
- **Units**: Dimensionless (0-1)
- **Interpretation**: Lower values indicate more equal stake distribution (better decentralization)

### Slashing Events
- **Definition**: Cumulative count of protocol violations resulting in stake penalties
- **Units**: Count
- **Interpretation**: Lower counts indicate better protocol compliance

### Byzantine Fault Tolerance (BFT)
- **Definition**: Maximum percentage of malicious validators the network can tolerate
- **Formula**: `(1 - FinalizationThreshold) * 2`
- **Units**: Percentage
- **Interpretation**: Higher percentages indicate more robust consensus

## Economic Metrics

### Reward Distribution
- **Definition**: Distribution of protocol rewards among validators
- **Units**: Token amount
- **Interpretation**: Indicates economic incentive alignment

### Inflation Rate
- **Definition**: Rate of new token issuance for validator rewards
- **Units**: Percentage per epoch
- **Interpretation**: Affects long-term token economics

### Slashing Rate
- **Definition**: Percentage of stake penalized for protocol violations
- **Units**: Percentage
- **Interpretation**: Higher rates increase violation deterrence but may increase validator risk

## Attack Resistance Metrics

### Cartel Resistance
- **Definition**: Network's ability to maintain consensus despite coordinated validator cartels
- **Measurement**: Finality rate and reward distribution under cartel attack simulations
- **Interpretation**: Smaller changes from baseline indicate better resistance

### Sybil Resistance
- **Definition**: Network's ability to resist attacks from multiple validators controlled by a single entity
- **Measurement**: Changes in stake distribution and synergy scores under Sybil attack simulations
- **Interpretation**: Smaller changes from baseline indicate better resistance

### Byzantine Resistance
- **Definition**: Network's ability to maintain consensus despite arbitrary malicious behavior
- **Measurement**: Finality rate under high Byzantine ratio simulations
- **Interpretation**: Smaller reductions in finality rate indicate better resistance

## Validator Strategy Metrics

### Strategy Performance
- **Definition**: Comparative performance of different validator strategies
- **Measurements**:
  - Final stake
  - Blocks proposed
  - Rewards earned
  - Synergy score
- **Interpretation**: Indicates which strategies are most effective under different conditions

### Opportunistic Behavior
- **Definition**: Strategic participation to maximize rewards with minimal effort
- **Measurement**: Reward-to-effort ratio compared to honest validators
- **Interpretation**: Lower effectiveness discourages free-riding

## Formal Verification Metrics

### Safety Violations
- **Definition**: Instances where safety properties are violated in formal verification
- **Units**: Count
- **Interpretation**: Any violations indicate potential consensus failures

### Liveness Violations
- **Definition**: Instances where liveness properties are violated in formal verification
- **Units**: Count
- **Interpretation**: Any violations indicate potential consensus stalling

### Decentralization Properties
- **Definition**: Formal verification of stake distribution and validator selection fairness
- **Measurement**: Satisfaction of TLA+ specified properties
- **Interpretation**: Property satisfaction indicates protocol meets decentralization goals

## Statistical Analysis Metrics

### Correlation Coefficients
- **Definition**: Pearson or Spearman correlations between protocol parameters
- **Units**: Dimensionless (-1 to 1)
- **Interpretation**: Indicates strength and direction of relationships between parameters

### Hypothesis Test Results
- **Definition**: Statistical significance of differences between scenarios
- **Measurement**: p-values from appropriate statistical tests
- **Interpretation**: p < 0.05 typically indicates significant differences

### Parameter Sensitivity
- **Definition**: Effect of parameter changes on key protocol outcomes
- **Measurement**: Elasticity or partial derivatives
- **Interpretation**: Higher values indicate parameters with stronger influence on outcomes