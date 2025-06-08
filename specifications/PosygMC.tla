---- MODULE PosygMC ----
(* Model checking configuration for PoSyg *)

EXTENDS Integers, Sequences, FiniteSets, TLC, Reals, PosygCore, PosygProperties

\* Small model for testing
MCValidators == {"v1", "v2", "v3", "v4"}
MCMaxStake == 1000
MCMaxEpochs == 10
MCBlocksPerEpoch == 4

\* Override constants for model checking
MC == INSTANCE PosygCore WITH
    Validators <- MCValidators,
    MaxStake <- MCMaxStake,
    StakeWeight <- 0.4,
    ActivityWeight <- 0.4,
    GovernanceWeight <- 0.2,
    SlashingRate <- 0.1,
    FinalizationThreshold <- 0.67,
    MaxEpochs <- MCMaxEpochs,
    BlocksPerEpoch <- MCBlocksPerEpoch

\* State constraints for bounded model checking
StateConstraint ==
    /\ currentEpoch <= MCMaxEpochs
    /\ Len(blockchain) <= MCMaxEpochs * MCBlocksPerEpoch

\* Properties to check in model checking
MCSpec == Spec /\ [][StateConstraint]_<<validatorStates, currentEpoch, currentHeight, 
                                 blockchain, pendingBlock, messages, synergyScores>>

MCTypeInvariant == TypeInvariant
MCSafetyInvariant == SafetyInvariant
MCByzantineFaultTolerance == ByzantineFaultTolerance
MCDecentralizationInvariant == DecentralizationInvariant
MCCartelResistance == CartelResistance

\* Model checking properties
MCProperties ==
    /\ MCTypeInvariant
    /\ MCSafetyInvariant
    /\ MCByzantineFaultTolerance
    /\ MCDecentralizationInvariant
    /\ MCCartelResistance

====================================================================
