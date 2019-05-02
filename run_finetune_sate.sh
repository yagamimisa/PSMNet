#!/bin/bash


python finetune_sate.py --maxdisp 192 \
                        --model stackhourglass \
                        --datapath /data/projects/dataFusion/Track2/Train-Track2-RGB/\
                        --loadmodel models/pretrained_model_KITTI2015.tar  \
                        --savemodel trained/finetune_15_new/\
			--epochs 50\
                        --logfile logs/finetune_15_new_log.txt
