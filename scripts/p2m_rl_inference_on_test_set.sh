#!/usr/bin/bash

python -u pocket2mol_rl/sample/sample_actor_for_pdb.py \
        -c checkpoints/Pocket2Mol_RL.pt \
        -f 'test_outputs/*/receptor.pdb' \
        -s p2mrl_reproduce_SDF -n 100 -det