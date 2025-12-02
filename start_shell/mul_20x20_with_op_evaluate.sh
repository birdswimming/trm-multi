torchrun --nproc_per_node=1 run_eval_only.py \
  --checkpoint checkpoints/Multi-20x20_with_op-ACT-torch/pretrain_multi_20x20_with_op/step_21000_best \
  --dataset data/multi-20x20_with_op \
  --outdir checkpoints/multi-20×20-with_op \
  --global-batch-size 10 \
  --apply-ema