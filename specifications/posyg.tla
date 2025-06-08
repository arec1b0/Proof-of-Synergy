---- MODULE PoSyg ----
(* 
   Formal specification of Proof of Synergy consensus protocol
   Author: Daniil Krizhanovskyi
   Date: October 2024
*)

EXTENDS Integers, Sequences, FiniteSets, TLC, Reals

\* Include all the specification modules
\* Note: TLA+ doesn't actually support includes, this is just for documentation
\* In practice, you would concatenate these files when running TLC
\* or use the Toolbox to manage the modules

(* 
   This is the main specification file that includes all components.
   To use with TLC, you'll need to concatenate all the .tla files:
   cat specifications/*.tla > posyg_full.tla
*)

\* Include core protocol definitions
\* INCLUDE Specifications/PosygCore.tla

\* Include model checking configuration
\* INCLUDE Specifications/PosygMC.tla

\* Include properties and invariants
\* INCLUDE Specifications/PosygProperties.tla

\* Include attack scenarios
\* INCLUDE Specifications/PosygAttacks.tla

====================================================================
\* End of file posyg.tla
