export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
python sample_guidance_embedding.py ODE \
    --image-size 256 \
    --ckpt /media/dataset2/jiwon/sit_train_cfg_distill/results/000-SiT-XL-2-Linear-velocity-velocity/SiT-XL-2-CFG-Distill-10000.pt \
    --guidance-embedding \
    --cfg-scale 8.0