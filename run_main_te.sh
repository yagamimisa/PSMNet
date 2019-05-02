#!/bin/bash


python main_te.py --maxdisp 192 \
                        --model stackhourglass \
                        --datapath /data/projects/dataFusion/Track2/Train-Track2-RGB/\
                        --loadmodel trained/finetune_15_new/finetune_50.tar \
                        --savemodel trained/finetune_15_new/\
			--epochs 50\
