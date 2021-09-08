#!/bin/bash

data=$1
output=$2
start=$3
end=$4
dataset=$5
gpu=$6

if [[ -n "$data" ]] && [[ -n "$start" ]] && [[ -n "$end" ]] && [[ -n "$gpu" ]]; then
    for i in $(eval echo {$start..$end})
    do
        CUDA_VISIBLE_DEVICES=$gpu python train.py --data_dir $data --output_dir $output --fold_idx $i --dataset $dataset
    done
else
    echo "argument error"
fi

