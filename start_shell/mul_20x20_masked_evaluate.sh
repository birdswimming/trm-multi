torchrun --nproc_per_node=1 run_eval_only.py \
  --checkpoint checkpoints/Multi-20x20-masked-less-data-ACT-torch/8GPUs-4cycles2025-11-19_11_29/step_72828 \
  --dataset data/multi-20x20-masked-less-data \
  --outdir checkpoints/multi-20×20-masked \
  --global-batch-size 10 \
  --apply-ema