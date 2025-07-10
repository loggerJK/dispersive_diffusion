export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

python sample_repa_dispersive.py ODE \
    --image-size 256 \
    --seed 0 \
    --baseline \
    --cfg-scale 0.0

python sample_repa_dispersive.py ODE \
    --image-size 256 \
    --seed 0 \
    --baseline \
    --cfg-scale 4.0