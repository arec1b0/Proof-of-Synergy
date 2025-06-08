# TLA+ Guide for Proof of Synergy

This guide provides an overview of the TLA+ formal specifications for the Proof of Synergy (PoSyg) consensus protocol, explaining how to understand, modify, and verify the protocol properties.

## Introduction to TLA+ in PoSyg

TLA+ (Temporal Logic of Actions) is a formal specification language used to design, model, and verify concurrent and distributed systems. In the PoSyg project, TLA+ specifications serve to:

1. Formally define the protocol behavior
2. Verify safety and liveness properties
3. Model check attack scenarios
4. Ensure protocol correctness under various conditions

## Specification Structure

The PoSyg TLA+ specifications are organized into modular files:

- **posyg.tla**: Main entry point referencing all modules
- **PosygCore.tla**: Core protocol definitions and state transitions
- **PosygProperties.tla**: Safety, liveness, and decentralization properties
- **PosygMC.tla**: Model checking configuration for TLC
- **PosygAttacks.tla**: Formalized attack scenarios and security properties

## Key Components

### Constants and Variables

The protocol is parameterized with constants defined in `PosygCore.tla`:

```tla
CONSTANTS 
    Validators,              \* Set of validator identities
    MaxStake,                \* Maximum stake per validator
    StakeWeight,             \* Weight for stake in synergy score
    ActivityWeight,          \* Weight for activity in synergy score  
    GovernanceWeight,        \* Weight for governance in synergy score
    SlashingRate,            \* Base slashing penalty rate
    FinalizationThreshold,   \* Threshold for block finalization (e.g., 0.67)
    MaxEpochs,               \* Maximum epochs to simulate
    BlocksPerEpoch           \* Number of blocks per epoch
```

State is tracked through variables:

```tla
VARIABLES
    validatorStates,    \* Function mapping validators to their states
    currentEpoch,       \* Current epoch number
    currentHeight,      \* Current block height
    blockchain,         \* Sequence of blocks
    pendingBlock,       \* Currently proposed block (if any)
    messages,           \* Set of messages in transit
    synergyScores       \* Current synergy scores
```

### State Transitions

The protocol's behavior is defined through actions (state transitions):

1. **ProposeBlock**: A validator proposes a new block
2. **AttestBlock**: Validators attest to proposed blocks
3. **FinalizeBlock**: Blocks reaching the attestation threshold become finalized
4. **UpdateSynergyScores**: Recalculate validator synergy scores
5. **SlashValidator**: Penalize protocol violations
6. **AdvanceEpoch**: Move to the next epoch

### Properties

Key properties verified in `PosygProperties.tla`:

1. **Safety Properties**:
   - **Consistency**: No conflicting blocks at the same height
   - **Agreement**: All honest validators agree on finalized blocks

2. **Liveness Properties**:
   - **Progress**: The blockchain continues to grow
   - **Finality**: Valid blocks eventually become finalized

3. **Decentralization Properties**:
   - **StakeDistribution**: Stake remains sufficiently distributed
   - **ValidatorParticipation**: Sufficient validator participation

4. **Economic Properties**:
   - **IncentiveAlignment**: Honest behavior is economically optimal
   - **SlashingEffectiveness**: Protocol violations are properly penalized

## Running Model Checking

To verify the protocol properties using the TLC model checker:

1. Install TLA+ Toolbox from https://lamport.azurewebsites.net/tla/toolbox.html
2. Open the Toolbox and create a new specification pointing to `posyg.tla`
3. Create a new model with the following settings:
   - Set model values for constants as defined in `PosygMC.tla`
   - Select properties to check from `PosygProperties.tla`
   - Configure advanced options (state constraints, symmetry, etc.)
4. Run the model checker

Alternatively, use the command-line TLC:

```bash
java -cp tla2tools.jar tlc2.TLC -config PosygMC.cfg posyg.tla
```

## Attack Modeling

`PosygAttacks.tla` defines various attack scenarios:

1. **Cartel Attack**: Coordinated behavior among a group of validators
2. **Byzantine Attack**: Arbitrary malicious behavior
3. **Sybil Attack**: Multiple validators controlled by a single entity
4. **Stake Centralization**: Excessive stake concentration

Each attack is modeled as a specific behavior pattern and verified against security properties.

## Extending the Specifications

To extend or modify the TLA+ specifications:

1. **Adding New Parameters**: Add to the `CONSTANTS` section in `PosygCore.tla`
2. **Modifying State Transitions**: Update relevant actions in `PosygCore.tla`
3. **Adding New Properties**: Define in `PosygProperties.tla`
4. **Creating New Attack Scenarios**: Add to `PosygAttacks.tla`
5. **Updating Model Checking Configuration**: Modify `PosygMC.tla`

## Best Practices

1. **Keep Modules Focused**: Each module should have a single responsibility
2. **Document Assumptions**: Use `ASSUME` statements to document constraints
3. **Use Invariants**: Define invariants to catch errors early
4. **Limit State Space**: Use reasonable bounds for model checking
5. **Incremental Verification**: Start with simple models and gradually increase complexity

## Common Issues and Solutions

1. **State Space Explosion**: 
   - Reduce the number of validators
   - Limit the number of epochs
   - Use symmetry sets for validators

2. **Property Violations**:
   - Check counterexamples carefully
   - Refine property definitions
   - Consider if the violation represents a real protocol issue

3. **Performance Optimization**:
   - Use view symmetry when possible
   - Apply state constraints
   - Consider hash compaction for large models

## Resources

- [TLA+ Home Page](https://lamport.azurewebsites.net/tla/tla.html)
- [Learn TLA+](https://learntla.com/)
- [Practical TLA+](https://www.apress.com/gp/book/9781484238288)
- [TLA+ Examples](https://github.com/tlaplus/Examples)