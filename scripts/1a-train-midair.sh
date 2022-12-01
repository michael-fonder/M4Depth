#!/bin/bash

savepath=$1;

python main.py --mode=train --dataset="midair" --seq_len=4 --db_seq_len=8 --arch_depth=6 --ckpt_dir="$savepath" --log_dir="$savepath/summaries" --records=data/midair/train_data/ --enable_validation $2
