#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="3"

layers_list=({21..27}) 
using_steps=($(seq 1 4 25) 27)  # 1, 5, 9, 13, 17, 21, 25, 27
batch_size=32
update_scales=(0.1 0.5 1.0 5.0)
seed=0

for update_scale in "${update_scales[@]}"; do
    for layer in "${layers_list[@]}"; do
        for step in "${using_steps[@]}"; do
            echo "Running with parameters:"
            echo "  Layer Index: $layer"
            echo "  Loss Style: $loss_style"
            echo "  Using Steps: $step"
            echo "  Update Scale: $update_scale"
            echo "  Batch Size: $batch_size"
            echo "  Seed: $seed"
            echo "  Output Folder: $folder_name"
            echo "----------------------------------------"
            python sample_dispersive.py ODE \
                --image-size 256 \
                --seed "$seed" \
                --ckpt ./checkpoint/SiT-XL-2-256.pt \
                --using_steps "$step" \
                --update_scale "$update_scale" \
                --batch_size "$batch_size" \
                --layer "$layer"
        done
    done
done