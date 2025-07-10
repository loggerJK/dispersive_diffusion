#!/usr/bin/env bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="5"
# Create main results directory


# Define parameter ranges
layer_indices=(1)
loss_styles=("l2")
using_steps=(9)
update_methods=("direct")
loss=("CustomL2")
exp_name=("CustomL2_NormScale_woNormalization")
seeds=( 6 7 8 9)
update_scales=(1)
batch_sizes=(7)

# Loop through all parameter combinations
for layer_index in "${layer_indices[@]}"; do
    for loss_style in "${loss_styles[@]}"; do
        for using_step in "${using_steps[@]}"; do
            for update_method in "${update_methods[@]}"; do
                for update_scale in "${update_scales[@]}"; do
                    for seed in "${seeds[@]}"; do
                        for batch_size in "${batch_sizes[@]}"; do
                            # Create folder name based on parameters
                            # folder_name="/media/dataset1/representation_EntireBatch/dog_check/layer${layer_index}_${loss_style}_step${using_step}_${update_method}_scale${update_scale}_batch${batch_size}/seed${seed}"
                            # mkdir -p "$folder_name"
                            # file_name="${folder_name}"
                            echo "Running with parameters:"
                            echo "  Layer Index: $layer_index"
                            echo "  Loss Style: $loss_style"
                            echo "  Using Steps: $using_step"
                            echo "  Update Method: $update_method"
                            echo "  Update Scale: $update_scale"
                            echo "  Batch Size: $batch_size"
                            echo "  Seed: $seed"
                            echo "  Output Folder: $folder_name"
                            echo "----------------------------------------"
                            
                            # Run same_prompt.py with current parameters
                            python entire_batch_loss_cfg_jiwon.py \
                                --layer_index "$layer_index" \
                                --loss_style "$loss_style" \
                                --using_steps "$using_step" \
                                --update_method "$update_method" \
                                --update_scale "$update_scale" \
                                --batch_size "$batch_size" \
                                --exp_name "$exp_name" \
                                --seed "$seed" \
                                --loss "$loss"
                            
                            echo "Completed: $file_name"
                            echo "========================================"
                        done
                    done
                done
            done
        done
    done
done

echo "All parameter combinations completed!"