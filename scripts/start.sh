#!/usr/bin/env bash

python -u train.py --dataset /Your Path \
                   --arch_cfg configs/xxxx.yaml \
                   --data_cfg configs/xxxx.yaml \
                   --gpus '0, 1' \
                   --debug 0