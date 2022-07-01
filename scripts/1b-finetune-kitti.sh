#!/bin/bash

savepath=$1;

if [ ! -d "$savepath/train-midair" ]
then
    mv "$savepath/train" "$savepath/train-midair";
    mv "$savepath/best" "$savepath/train"; 
fi

python finetune-kitti.py --arch_depth=6 --ckpt_dir="$savepath" --log_dir="$savepath/summaries" --records=data --enable_validation
