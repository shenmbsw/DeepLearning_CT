#!/bin/bash -l

module load cuda/8.0
module load cudnn/5.1
module load python/3.6.0
module load tensorflow/r1.0_python-3.6.0

python small.py
