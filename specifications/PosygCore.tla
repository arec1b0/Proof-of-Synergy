---- MODULE PosygCore ----
(* Core protocol definitions for Proof of Synergy *)

EXTENDS Integers, Sequences, FiniteSets, TLC, Reals

CONSTANTS 
    Validators,              \* Set of validator identities
    MaxStake,               \* Maximum stake per validator
    StakeWeight,            \* Weight for stake in synergy score
    ActivityWeight,         \* Weight for activity in synergy score  
    GovernanceWeight,       \* Weight for governance in synergy score
    SlashingRate,           \* Base slashing penalty rate
    FinalizationThreshold,  \* Threshold for block finalization (e.g., 0.67)
    MaxEpochs,              \* Maximum epochs to simulate
    BlocksPerEpoch          \* Number of blocks per epoch

ASSUME
    /\ StakeWeight + ActivityWeight + GovernanceWeight = 1
    /\ StakeWeight >= 0 /\ ActivityWeight >= 0 /\ GovernanceWeight >= 0
    /\ FinalizationThreshold > 0.5 /\ FinalizationThreshold <= 1
    /\ SlashingRate >= 0 /\ SlashingRate < 1

VARIABLES
    validatorStates,    \* Function mapping validators to their states
    currentEpoch,       \* Current epoch number
    currentHeight,      \* Current block height
    blockchain,         \* Sequence of blocks
    pendingBlock,       \* Currently proposed block (if any)
    messages,           \* Set of messages in transit
    synergyScores       \* Current synergy scores

\* Type definitions
ValidatorState == [
    stake: 0..MaxStake,
    isActive: BOOLEAN,
    blocksProposed: Nat,
    blocksAttested: Nat,
    governanceVotes: Nat,
    slashingCount: Nat,
    lastActiveEpoch: Nat
]

Block == [
    height: Nat,
    epoch: Nat,
    proposer: Validators,
    parentHash: STRING,
    status: {"proposed", "finalized", "orphaned"},
    attestations: SUBSET Validators
]

Message == [
    type: {"propose", "attest"},
    sender: Validators,
    block: Block,
    signature: STRING
]

\* Helper functions
TotalStake == 
    LET activeVals == {v \in Validators : validatorStates[v].isActive}
    IN ReduceSet(LAMBDA v, acc: acc + validatorStates[v].stake, activeVals, 0)

ComputeSynergyScore(v) ==
    LET
        state == validatorStates[v]
        stakeScore == IF TotalStake > 0 
                     THEN state.stake / TotalStake
                     ELSE 0
        activityScore == IF currentHeight > 0
                        THEN (state.blocksProposed + state.blocksAttested) / currentHeight
                        ELSE 0
        governanceScore == IF currentEpoch > 0
                          THEN state.governanceVotes / currentEpoch  
                          ELSE 0
        slashingPenalty == (1 - SlashingRate)^state.slashingCount
    IN
        (StakeWeight * stakeScore + 
         ActivityWeight * activityScore + 
         GovernanceWeight * governanceScore) * slashingPenalty * 1000

\* Select block proposer based on synergy scores
SelectProposer ==
    LET 
        activeVals == {v \in Validators : validatorStates[v].isActive}
        scores == [v \in activeVals |-> synergyScores[v]]
        maxScore == ReduceSet(LAMBDA v, acc: 
                             IF scores[v] > acc THEN scores[v] ELSE acc, 
                             activeVals, 0)
    IN
        CHOOSE v \in activeVals : scores[v] = maxScore

\* Initial state
Init ==
    /\ validatorStates = [v \in Validators |->
        [stake |-> MaxStake \div Cardinality(Validators),  \* Equal initial stakes
         isActive |-> TRUE,
         blocksProposed |-> 0,
         blocksAttested |-> 0,
         governanceVotes |-> 0,
         slashingCount |-> 0,
         lastActiveEpoch |-> 0]]
    /\ currentEpoch = 0
    /\ currentHeight = 0
    /\ blockchain = << >>  \* Empty blockchain
    /\ pendingBlock = NULL
    /\ messages = {}
    /\ synergyScores = [v \in Validators |-> ComputeSynergyScore(v)]

\* Actions
ProposeBlock(proposer) ==
    /\ pendingBlock = NULL  \* No pending block
    /\ validatorStates[proposer].isActive
    /\ LET newBlock == [
           height |-> currentHeight + 1,
           epoch |-> currentEpoch,
           proposer |-> proposer,
           parentHash |-> IF Len(blockchain) > 0 
                         THEN blockchain[Len(blockchain)].height  \* Simplified hash
                         ELSE "genesis",
           status |-> "proposed",
           attestations |-> {proposer}]  \* Self-attestation
       IN
       /\ pendingBlock' = newBlock
       /\ messages' = messages \cup {[
           type |-> "propose",
           sender |-> proposer,
           block |-> newBlock,
           signature |-> "sig"]}  \* Simplified signature
       /\ validatorStates' = [validatorStates EXCEPT 
           ![proposer].blocksProposed = @ + 1,
           ![proposer].lastActiveEpoch = currentEpoch]
       /\ UNCHANGED <<currentEpoch, currentHeight, blockchain, synergyScores>>

AttestBlock(validator) ==
    /\ pendingBlock # NULL
    /\ validator \notin pendingBlock.attestations
    /\ validatorStates[validator].isActive
    /\ validator # pendingBlock.proposer  \* Can't double-attest
    /\ pendingBlock' = [pendingBlock EXCEPT 
        !.attestations = @ \cup {validator}]
    /\ messages' = messages \cup {[
        type |-> "attest",
        sender |-> validator,
        block |-> pendingBlock,
        signature |-> "sig"]}
    /\ validatorStates' = [validatorStates EXCEPT
        ![validator].blocksAttested = @ + 1,
        ![validator].lastActiveEpoch = currentEpoch]
    /\ UNCHANGED <<currentEpoch, currentHeight, blockchain, synergyScores>>

FinalizeBlock ==
    /\ pendingBlock # NULL
    /\ LET 
        attestWeight == ReduceSet(
            LAMBDA v, acc: acc + synergyScores[v],
            pendingBlock.attestations, 0)
        totalWeight == ReduceSet(
            LAMBDA v, acc: IF validatorStates[v].isActive 
                          THEN acc + synergyScores[v] 
                          ELSE acc,
            Validators, 0)
       IN
       attestWeight >= totalWeight * FinalizationThreshold
    /\ blockchain' = Append(blockchain, [pendingBlock EXCEPT !.status = "finalized"])
    /\ currentHeight' = currentHeight + 1
    /\ pendingBlock' = NULL
    /\ messages' = {}  \* Clear messages after finalization
    /\ IF (currentHeight + 1) % BlocksPerEpoch = 0
       THEN currentEpoch' = currentEpoch + 1
       ELSE UNCHANGED currentEpoch
    /\ synergyScores' = [v \in Validators |-> ComputeSynergyScore(v)]
    /\ UNCHANGED validatorStates

OrphanBlock ==
    /\ pendingBlock # NULL
    /\ LET 
        attestWeight == ReduceSet(
            LAMBDA v, acc: acc + synergyScores[v],
            pendingBlock.attestations, 0)
        totalWeight == ReduceSet(
            LAMBDA v, acc: IF validatorStates[v].isActive 
                          THEN acc + synergyScores[v] 
                          ELSE acc,
            Validators, 0)
       IN
       attestWeight < totalWeight * FinalizationThreshold
    /\ \* Timeout or insufficient attestations
       TRUE  \* Simplified timeout condition
    /\ blockchain' = Append(blockchain, [pendingBlock EXCEPT !.status = "orphaned"])
    /\ pendingBlock' = NULL
    /\ messages' = {}
    /\ \* Slash the proposer for failed block
       validatorStates' = [validatorStates EXCEPT
           ![pendingBlock.proposer].slashingCount = @ + 1]
    /\ UNCHANGED <<currentEpoch, currentHeight, synergyScores>>

GovernanceVote(validator) ==
    /\ validatorStates[validator].isActive
    /\ validatorStates' = [validatorStates EXCEPT
        ![validator].governanceVotes = @ + 1]
    /\ UNCHANGED <<currentEpoch, currentHeight, blockchain, pendingBlock, messages, synergyScores>>

\* Next state relation
Next ==
    \/ \E v \in Validators : ProposeBlock(v)
    \/ \E v \in Validators : AttestBlock(v)
    \/ FinalizeBlock
    \/ OrphanBlock
    \/ \E v \in Validators : GovernanceVote(v)


\* Specification
Spec == Init /\ [][Next]_<<validatorStates, currentEpoch, currentHeight, 
                           blockchain, pendingBlock, messages, synergyScores>>

====================================================================
