run_name="pretrain_multi_20x20_with_op_6l"
torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/multi-20x20_with_op]" \
  evaluators="[]" \
  epochs=1000 eval_interval=10 \
  lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  global_batch_size=512 lr_warmup_steps=100 \
  checkpoint_every_eval=True \
  +run_name=${run_name} ema=True