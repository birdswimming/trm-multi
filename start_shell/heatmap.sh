torchrun --nproc_per_node=1 heatmap.py \
  --checkpoint checkpoints/Multi-20x20-masked-less-data-ACT-torch/8GPUs-4cycles2025-11-19_11_29/step_62832 \
  --dataset data/none \
  --outdir checkpoints/heatmap-multi-20×20-masked