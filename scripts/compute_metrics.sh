#!/usr/bin/bash


models=("ligan" "ar" "graphbp" "flag" "targetdiff" "pocket2mol" "p2mrl")
for model in ${models[@]}; do
    python -u pocket2mol_rl/evaluation/quick_evaluate.py \
            -od test_outputs -pf receptor.pdb -sd ${model}_SDF \
            -p test_cache/test_${model}_outputs -nw 8
done
