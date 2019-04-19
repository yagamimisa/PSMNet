#!/bin/bash


python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datapath dataset/data_scene_flow/training/\
		   --datatype 2015\
                   --epochs 2\
                   --loadmodel models/pretrained_sceneflow.tar \
                   --savemodel trained/

