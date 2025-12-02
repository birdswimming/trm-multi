import os
import yaml
import torch
import torch.distributed as dist
from typing import Any, Dict, Mapping, cast
from contextlib import nullcontext
import torch.backends.cudnn as cudnn
from hydra import initialize, compose
from omegaconf import OmegaConf
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
# Reuse functions and classes from pretrain.py
from pretrain import (
    create_model,
    load_checkpoint,
    PretrainConfig,
)
from utils.functions import load_model_class
from torch import nn
# Prefer new TF32 API controls to avoid deprecation warnings and ensure predictable math.
try:
    # Use strict IEEE FP32 by default for matmul to prioritize correctness.
    # Change to 'tf32' if you prefer TF32 acceleration.
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
except Exception:
    # Backend may not be available on CPU-only or some environments.
    pass
from heatmap_dataset import HeatmapDataset, get_batch, decode, get_batch_multiviews, HeatmapDatasetMultiViews
from torch.utils.data import DataLoader
import numpy as np

def ceil_to_multiple(x, m):
    return ((x + m - 1) // m) * m
height = 20
width = 20
vocab_size = 14
datasets = []
max_digits_num = 10
random_seed = 36
batch_size = 256
batch_size = ceil_to_multiple(batch_size, max_digits_num)

def create_model(config: PretrainConfig, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=batch_size,
        vocab_size=vocab_size,
        seq_len=height * width,
        num_puzzle_identifiers=1,
        causal=False  # Non-autoregressive
    )
    print(model_cfg)

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        # Default to NOT compiling unless explicitly opted-in. This avoids
        # torch.compile wrapping that introduces the "_orig_mod." prefix in
        # state_dict keys and causes checkpoint mismatches at load time.
        if os.environ.get("ENABLE_COMPILE", "0") == "1":
            try:
                model = torch.compile(model)  # type: ignore
            except Exception as _e:
                print("torch.compile failed; proceeding without compilation:", _e)

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)
        return model

def correct_ratio(model: nn.Module, config: PretrainConfig, rank: int, world_size: int):
    correct_nums = np.zeros([max_digits_num, max_digits_num], dtype=np.float32)
    total_nums = np.zeros([max_digits_num, max_digits_num], dtype=np.float32)

    for i in range(1, max_digits_num + 1):
        dataset = HeatmapDataset(
            length=1000000,
            seed=random_seed,
            width=width,
            height=height,
            min_num_len_1 = 1,
            max_num_len_1 = max_digits_num,
            num_len_2 = i,
        )
        datasets.append(dataset)

    dataloader_iters = []
    for i, dataset in enumerate(datasets):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
        )
        dataloader_iters.append(iter(dataloader))

    return_keys = set(["preds"])
    carry = None
    
    for i, dataloader_iter in enumerate(dataloader_iters):
        line_correct_nums = np.zeros(max_digits_num, dtype=np.float32)
        line_total_nums = np.full(max_digits_num, batch_size // max_digits_num, dtype=np.float32)
        current_correct_status = torch.ones(batch_size, dtype=torch.bool, device="cuda")
        batch_datas = get_batch(dataloader_iter)
        for batch in tqdm(batch_datas):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = model.initial_carry(batch)  # type: ignore
            
            # for k, v in batch.items():
            #     print(f"{k}:{v.shape}")
            
            # Forward
            inference_steps = 0

            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                if all_finish:
                    break
            
            infer_answer = preds["preds"]
            label = batch["labels"]
            batch_correct = (infer_answer == label).view(batch_size, -1).all(dim=1)
            # print(f"batch_correct with {i+1} digits: {batch_correct}")
            current_correct_status &= batch_correct
            
        indices = torch.arange(batch_size) % max_digits_num    # [batch_size]
        correct = current_correct_status.cpu().numpy()         # bool array
        line_correct_nums += np.bincount(
            indices.numpy(),
            weights=correct.astype(np.float32),
            minlength=max_digits_num,
        )
        print(f"number2 with {i+1} digits: {line_correct_nums}")
        
        
        correct_nums[i] += line_correct_nums
        total_nums[i] += line_total_nums
    return correct_nums / total_nums

def correct_ratio_multiviews(model: nn.Module, config: PretrainConfig, rank: int, world_size: int, num_views:int = 16):
    correct_nums = np.zeros([max_digits_num, max_digits_num], dtype=np.float32)
    total_nums = np.zeros([max_digits_num, max_digits_num], dtype=np.float32)

    for i in range(1, max_digits_num + 1):
        dataset = HeatmapDatasetMultiViews(
            length=1000000,
            seed=random_seed,
            width=width,
            height=height,
            min_num_len_1 = 1,
            max_num_len_1 = max_digits_num,
            num_len_2 = i,
            num_views = num_views
        )
        datasets.append(dataset)

    dataloader_iters = []
    for i, dataset in enumerate(datasets):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
        )
        dataloader_iters.append(iter(dataloader))

    return_keys = set(["preds", "q_halt_logits"])
    carry = None
    puzzle_identifiers = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i, dataloader_iter in enumerate(dataloader_iters):
        line_correct_nums = np.zeros(max_digits_num, dtype=np.float32)
        line_total_nums = np.full(max_digits_num, batch_size // max_digits_num, dtype=np.float32)
        current_correct_status = torch.ones(batch_size, dtype=torch.bool, device="cuda")
        batch_datas = get_batch_multiviews(dataloader_iter)
        for batch_views in tqdm(batch_datas):
            batch_views = {k: v.cuda() for k, v in batch_views.items()}
            best_label = None
            best_answer = None
            best_qhalt = None
            
            
            for j in range(num_views):
                batch = {k: v[j] for k, v in batch_views.items()}
                batch["puzzle_identifiers"] = puzzle_identifiers
                with torch.device("cuda"):
                    carry = model.initial_carry(batch)  # type: ignore
                
                # for k, v in batch.items():
                #     print(f"{k}:{v.shape}")
                
                # Forward
                inference_steps = 0

                while True:
                    carry, loss, metrics, preds, all_finish = model(
                        carry=carry, batch=batch, return_keys=return_keys
                    )
                    inference_steps += 1
                    if all_finish:
                        break
                
                infer_answer = preds["preds"]
                infer_q_halt = preds["q_halt_logits"]
                label = batch["labels"]
                if best_qhalt == None:
                    best_qhalt = infer_q_halt.clone()
                    best_answer = infer_answer.clone()
                    best_label = label.clone()
                else:
                    update_mask = infer_q_halt > best_qhalt
                    best_qhalt = torch.where(update_mask, infer_q_halt, best_qhalt)

                    # mask shape: [B, 1, 1, ...] 与 answer 的维度对齐
                    expand_shape = [len(update_mask)] + [1] * (infer_answer.ndim - 1)
                    update_mask_expanded = update_mask.view(*expand_shape)
                    
                    # print("expanded mask shape: ", update_mask_expanded.shape)
                    # print("answer shape: ", infer_answer.shape)
                    
                    
                    best_answer = torch.where(update_mask_expanded, infer_answer, best_answer)
                    best_label  = torch.where(update_mask_expanded, label, best_label)

            batch_correct = (best_answer == best_label).view(batch_size, -1).all(dim=1)
            current_correct_status &= batch_correct
            
        indices = torch.arange(batch_size) % max_digits_num    # [batch_size]
        correct = current_correct_status.cpu().numpy()         # bool array
        line_correct_nums += np.bincount(
            indices.numpy(),
            weights=correct.astype(np.float32),
            minlength=max_digits_num,
        )
        print(f"number2 with {i+1} digits: {line_correct_nums}")
        
        
        correct_nums[i] += line_correct_nums
        total_nums[i] += line_total_nums
    return correct_nums / total_nums




def draw_heat_map(correct_ratios, name="correct_ratio_heatmap_masked.png"):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        correct_ratios,
        cmap='seismic_r',   # 蓝(高,正确) → 白 → 红(低,错误)
        vmin=0,
        vmax=1,
        origin='upper'      # 左上角为原点
    )

    plt.colorbar(im, label='Correct Ratio')

    max_num_length = correct_ratios.shape[0]
    plt.xticks(np.arange(max_num_length), np.arange(1, max_num_length + 1))
    plt.yticks(np.arange(max_num_length), np.arange(1, max_num_length + 1))

    plt.xlabel('First number length')
    plt.ylabel('Second number length')
    plt.title('Correct Ratio Heatmap')

    plt.tight_layout()
    plt.savefig(name, dpi=300)
    plt.close()


def parse_args():
    """Parse CLI arguments for the evaluation runner.

    Returns:
        argparse.Namespace with config path, checkpoint, dataset path, output
        directory, eval outputs to save, batch size override, EMA options,
        eval-only toggle, bf16 toggle, and one-batch mode.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/cfg_pretrain.yaml', help='YAML config file (pydantic fields)')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint file to load')
    p.add_argument('--dataset', required=False, help='Path to dataset directory to evaluate (overrides data_paths_test)')
    p.add_argument('--outdir', default=None, help='Directory to save evaluation preds (overrides checkpoint_path in config)')
    p.add_argument('--eval-save-outputs', nargs='+', default=['inputs','labels','puzzle_identifiers','preds'], help='List of keys to save during evaluation')
    p.add_argument('--global-batch-size', type=int, default=None, help='Global batch size override for evaluation')
    # Defaults: eval-only, bf16, and apply-ema are enabled unless explicitly disabled
    p.add_argument('--apply-ema', action='store_true', default=True, help='Apply EMA weights for evaluation (default: on). Use --no-apply-ema to disable')
    p.add_argument('--ema-shadow', default=None, help='Path to EMA shadow state dict (optional). If provided, it will be loaded into EMAHelper before applying EMA.')
        # repeats/seed-start removed: we evaluate exactly once per invocation
    p.add_argument('--eval-only', action='store_true', default=True, help='Run in eval-only mode (skip optimizer creation). Default: on. Use --no-eval-only to disable')
    p.add_argument('--bf16', action='store_true', default=True, help='Use CUDA autocast with bfloat16 during evaluation (default: on). Use --no-bf16 to disable')
    # Negative toggles for convenience
    p.add_argument('--no-apply-ema', dest='apply_ema', action='store_false', help='Disable EMA application during evaluation')
    p.add_argument('--no-eval-only', dest='eval_only', action='store_false', help='Disable eval-only (will construct optimizer); not recommended')
    p.add_argument('--no-bf16', dest='bf16', action='store_false', help='Disable bfloat16 autocast during evaluation')
    p.add_argument('--one-batch', action='store_true', help='Evaluate only a single random batch of size global_batch_size from the test split (faster smoke test).')
    return p.parse_args()

def main():
    args = parse_args()
    os.environ.setdefault('DISABLE_COMPILE', '1')
    RANK = 0
    WORLD_SIZE = 1

    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    # Compose config via Hydra on rank 0 and broadcast

    config_obj = None
    objects = [None]
    if RANK == 0:
    # Derive config directory and base name from args.config
        config_path = os.path.dirname(args.config) or 'config'
        config_name = os.path.splitext(os.path.basename(args.config))[0]

    # Compose Hydra config; CLI overrides applied programmatically below
        with initialize(config_path=config_path, job_name="run_eval_only"):
            hydra_cfg = compose(config_name=config_name)

    # Convert to plain dict (resolve interpolations)
        cfg_any = OmegaConf.to_container(hydra_cfg, resolve=True)
        if not isinstance(cfg_any, dict):
            raise RuntimeError('Composed config is not a mapping after OmegaConf.to_container')
        cfg: Dict[str, Any] = dict(cast(Mapping[str, Any], cfg_any))

    # Apply programmatic overrides
        cfg['data_paths_test'] = [args.dataset]
        cfg['load_checkpoint'] = args.checkpoint
        if args.outdir is not None:
            cfg['checkpoint_path'] = args.outdir
        if args.global_batch_size is not None:
            cfg['global_batch_size'] = args.global_batch_size
        cfg['eval_save_outputs'] = args.eval_save_outputs

    # Print composed config on rank 0
        try:
            print('\nComposed config (after Hydra compose + CLI overrides):')
            print(yaml.safe_dump(cfg, sort_keys=False))
        except Exception:
            print('Warning: failed to pretty-print composed config')

    # Build pydantic PretrainConfig
        config_obj = PretrainConfig(**cfg)
        objects = [config_obj]

    if WORLD_SIZE > 1:
        dist.broadcast_object_list(objects, src=0)

    config = objects[0]

    # Ensure config present
    if config is None:
        raise RuntimeError('Failed to load config via broadcast; config is None on this rank')

    # Seed RNGs
    torch.random.manual_seed(config.seed + RANK)
    # Let cuDNN pick fastest algorithms
    try:
        cudnn.benchmark = True
    except Exception:
        pass
    
    model = create_model(config, RANK, WORLD_SIZE)
    model.eval()
    
    # Set checkpoint output directory and ensure it exists
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join('checkpoints', 'eval_run')
    if RANK == 0:
        os.makedirs(config.checkpoint_path, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    if args.bf16 and use_cuda:
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    with torch.inference_mode(), amp_ctx:
        correct_ratios = correct_ratio_multiviews(model, config, RANK, WORLD_SIZE)
    
    draw_heat_map(correct_ratios, "correct_ratio_heatmap_masked_multiviews.png")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
