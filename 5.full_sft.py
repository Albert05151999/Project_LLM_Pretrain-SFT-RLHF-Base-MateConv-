# ========= 顶部 import 保持不变 =========
import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from model.model import Transformer
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

# ===== 新增：安全地获取是否主进程 =====
def is_main_process():
    return (not ddp) or (ddp and ddp_local_rank == 0)

def get_lr(it, all):
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch, wandb, start_step=0):
    start_time = time.time()
    # ===== DDP：让 sampler 每个 epoch 重置种子，避免抽样偏 =====
    if ddp and train_loader.sampler is not None and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    for step, (X, Y, loss_mask) in enumerate(train_loader, start=start_step):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        global_step = epoch * iter_per_epoch + step
        lr = get_lr(global_step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        t0 = time.time()
        with ctx:
            logits = model(X, Y)["logits"]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                Y.view(-1),
                ignore_index=0,
                reduction='none'
            )
            # 只对非 mask 的 token 计入 loss
            loss_mask_flat = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-9)

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # === 统计 ===
        step_time = time.time() - t0
        tokens_this_step = int(X.numel())  # 近似：batch * seq_len
        tokens_per_sec = tokens_this_step / max(step_time, 1e-6)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.4f} lr:{:.7f} t/s:{:.1f} tok/s:{:.0f}'.format(
                    epoch, args.epochs, step, iter_per_epoch,
                    float(loss.item()),
                    optimizer.param_groups[-1]['lr'],
                    step_time, tokens_per_sec
                )
            )

            if (wandb is not None) and is_main_process():
                gpu_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                log_dict = {
                    "loss": float(loss.item()),
                    "lr": float(optimizer.param_groups[-1]['lr']),
                    "global_step": int(global_step),
                    "epoch": float(epoch + step / max(iter_per_epoch, 1)),
                    "step_time_s": float(step_time),
                    "tokens_per_sec": float(tokens_per_sec),
                    "gpu_mem_gb": float(gpu_mem),
                }
                wandb.log(log_dict)

        if (step + 1) % args.save_interval == 0 and is_main_process():
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'

            checkpoint = {
                'model_state': model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'epoch': epoch,
                'step': step,
                'args': vars(args)
            }
            torch.save(checkpoint, ckp)
            Logger(f"Checkpoint saved at {ckp}")

            # ===== 可选：把 checkpoint 上传为 W&B Artifact =====
            if wandb is not None:
                try:
                    art = wandb.Artifact(name=f"sft_ckpt_dim{lm_config.dim}{moe_path}", type="model")
                    art.add_file(ckp)
                    wandb.log_artifact(art)
                except Exception:
                    pass

            model.train()

def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    if os.path.exists(checkpoint_path):
        Logger(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        if 'model_state' in checkpoint:
            state_dict = checkpoint["model_state"]
            model.load_state_dict(state_dict, strict=False)
        else:
            Logger("No model_state found, skipping model loading.")
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scaler_state' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        start_step = checkpoint.get('step', 0) + 1
        return start_epoch, start_step
    else:
        Logger(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
        return 0, 0

def init_model():
    tokenizer = AutoTokenizer.from_pretrained('./model/mateconv_tokenizer')
    model_from = 1
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    if model_from == 1:
        model = Transformer(lm_config)
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=args.device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModel.from_pretrained('./MateConv/out', trust_remote_code=True)
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MateConv Full SFT")
    # ===== 你的原有参数不变 =====
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=19, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="MateConv-Full-SFT", help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name (fallback to $WANDB_NAME)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')

    args = parser.parse_args()

    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # ====== W&B 运行名：优先用 --wandb_name 或 $WANDB_NAME，兜底自动生成 ======
    default_run = f"sft-{time.strftime('%Y%m%d-%H%M%S')}-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}"
    args.wandb_run_name = args.wandb_name or os.getenv("WANDB_NAME") or default_run
    # 项目名也允许环境变量覆盖
    args.wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT", "MateConv-Full-SFT")
    # W&B 目录兜底
    os.environ.setdefault("WANDB_DIR", os.path.join(os.getcwd(), "wandb"))

    # ===== autocast ctx 保持你的写法 =====
    if device_type == "cuda":
        if args.dtype == "bfloat16":
            ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        elif args.dtype == "float16":
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            ctx = nullcontext()
    else:
        ctx = nullcontext()

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # ===== 仅主进程初始化 W&B =====
    if args.use_wandb and is_main_process():
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    else:
        wandb = None  # 保持下方统一判断

    model, tokenizer = init_model()
    if (wandb is not None) and is_main_process():
        try:
            wandb.watch(model, log="all", log_freq=max(1, args.log_interval))
        except Exception:
            pass

    # ====== 数据集 ======
    df = pd.read_csv('./dataset/sft_data_mixed_single.csv')
    df = df.sample(frac=1.0)
    train_ds = SFTDataset(df, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # ====== 优化器与 scaler ======
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    moe_path = '_moe' if lm_config.use_moe else ''
    checkpoint_path = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'

    start_epoch, start_step = load_checkpoint(checkpoint_path, model, optimizer, scaler)

    if False and not lm_config.use_moe and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank], find_unused_parameters=lm_config.use_moe)

    iter_per_epoch = len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        train_epoch(epoch, wandb, start_step)
        start_step = 0

    # ===== 训练结束优雅收尾 =====
    if (wandb is not None) and is_main_process():
        import wandb as _wandb
        _wandb.finish()