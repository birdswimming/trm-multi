run_name="pretrain_multi_60x60"
torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/multi-60x60]" \
  evaluators="[]" \
  epochs=200 eval_interval=100 \
  lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  global_batch_size=64 lr_warmup_steps=2000 \
  checkpoint_every_eval=True \
  +run_name=${run_name} ema=True