#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="4"

python sample_dispersive.py ODE \
    --image-size 256 \
    --seed 0 \
    --ckpt ./checkpoint/SiT-XL-2-256.pt \
    --loss InfoNCE_entire \
    --using_steps 27 \
    --update_scale 1.0 \
    --batch_size 32 \
    --layer 24 \
    --same_noise \
    --save_features \
    --exp_name InfoNCE_NormScale_entire \
    --save_path ./outputs_feature_jiwon
    # --baseline \