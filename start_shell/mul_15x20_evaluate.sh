torchrun --nproc_per_node=1 run_eval_only.py \
  --checkpoint checkpoints/Multi-15x20-ACT-torch/pretrain_multi_15x20/step_1940 \
  --dataset data/multi-15x20 \
  --outdir checkpoints/multi-15×20_eval_run \
  --global-batch-size 10 \
  --apply-ema