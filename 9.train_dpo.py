import os, math, time, argparse, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
from contextlib import nullcontext

# 你工程内的模块
from model.model import Transformer
from model.LMConfig import LMConfig
from dataset_dpo import DpoDataset

# ====== 全局开关（供 is_main_process 使用）======
ddp = False
ddp_local_rank = 0

def is_main_process():
    return (not ddp) or (ddp and ddp_local_rank == 0)

def Logger(msg):
    if is_main_process():
        print(msg)

# ====== 学习率调度（cosine + warmup_iters/ratio 二选一）======
def get_warmup_iters(warmup_iters, warmup_ratio, total_steps):
    if warmup_ratio is not None and warmup_ratio > 0:
        return max(1, int(total_steps * float(warmup_ratio)))
    return int(warmup_iters or 0)

def get_lr(it, total_steps, base_lr, warmup_iters=0):
    if it < warmup_iters:
        return base_lr * it / max(1, warmup_iters)
    if it >= total_steps:
        return base_lr / 10
    decay_ratio = (it - warmup_iters) / max(1, (total_steps - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (base_lr / 10) + coeff * (base_lr - base_lr / 10)

# ====== 计算 logprob（仅 Mask==1 的 token）======
def seq_logprob(logits, tgt, loss_mask):
    # logits: [B, T, V]; tgt: [B, T]; mask: [B, T] (0/1)
    logp = F.log_softmax(logits, dim=-1)
    tgt_logp = torch.gather(logp, -1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T]
    tgt_logp = tgt_logp * loss_mask
    denom = loss_mask.sum(dim=1).clamp_min(1)
    return (tgt_logp.sum(dim=1) / denom)  # [B]

# ====== token 级 KL(policy ‖ ref)，仅在回答段 ======
def masked_token_kl(policy_logits, ref_logits, loss_mask):
    # 两边都 softmax，再做逐 token KL，然后按回答长度归一化
    p = F.log_softmax(policy_logits, dim=-1)
    q = F.softmax(ref_logits, dim=-1)
    kl_t = F.kl_div(p, q, log_target=False, reduction='none').sum(-1)  # [B, T]
    kl_t = kl_t * loss_mask
    denom = loss_mask.sum(dim=1).clamp_min(1)
    return (kl_t.sum(dim=1) / denom)  # [B]

# ====== DPO pairwise 损失 ======
def dpo_pair_loss(pos_logp, neg_logp, beta=0.2):
    # L = -log sigmoid( beta * (pos - neg) )
    diff = (pos_logp - neg_logp) * beta
    return F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))

# ====== （新增）带 mask 的 CE ======
def masked_ce_loss(logits, tgt, loss_mask):
    # logits:[B,T,V], tgt:[B,T], loss_mask:[B,T](0/1)
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        tgt.view(-1),
        reduction='none'
    ).view_as(loss_mask)
    ce = ce * loss_mask
    denom = loss_mask.sum().clamp_min(1)
    return ce.sum() / denom

# ====== DDP 初始化 ======
def init_distributed():
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MateConv DPO Distillation")
    parser.add_argument("--pairs_path", type=str, default="./dataset/dpo_pairs.jsonl")
    parser.add_argument("--out_dir",    type=str, default="out_dpo")
    parser.add_argument("--epochs",     type=int, default=2)
    parser.add_argument("--max_steps",  type=int, default=-1, help=">0 时优先生效，按总步数停止")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--device",     type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype",      type=str, default="bfloat16", choices=["bfloat16","float16","float32"])
    parser.add_argument("--use_wandb",  action="store_true")
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "MateConv-DPO"))
    parser.add_argument("--wandb_name", type=str, default=os.environ.get("WANDB_NAME", None))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip",  type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=None, help="与 warmup_iters 互斥，优先使用 ratio")
    parser.add_argument("--beta", type=float, default=0.2, help="DPO 温度")
    parser.add_argument("--kl_coef", type=float, default=0.05, help="参考 KL 正则强度（0 关闭）")
    parser.add_argument("--loss", type=str, default="dpo", choices=["dpo"], help="占位，当前仅 dpo")

    # ===（新增）损失权重：默认 CE 比 DPO 更重===
    parser.add_argument("--ce_coef",  type=float, default=0.7, help="监督 CE 权重（建议 0.3~0.8）")
    parser.add_argument("--dpo_coef", type=float, default=0.3, help="DPO 权重（建议 0.2~0.7）")

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    # === 模型与 tokenizer 初始化（对齐 full_sft.py） ===
    lm_config = LMConfig()
    os.makedirs(args.out_dir, exist_ok=True)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    if device_type == "cuda":
        if args.dtype == "bfloat16":
            ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        elif args.dtype == "float16":
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            ctx = nullcontext()
    else:
        ctx = nullcontext()

    # DDP
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, args.device
    if ddp:
        init_distributed()
        args.device = torch.device(DEVICE)

    # W&B
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        run_name = args.wandb_name or f"dpo-{time.strftime('%Y%m%d-%H%M%S')}"
        wandb.init(project=args.wandb_project or "MateConv-DPO", name=run_name, config=vars(args))

    # tokenizer & model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model/mateconv_tokenizer')
    model = Transformer(lm_config).to(args.device)

    # 参考模型（用于 KL）：结构相同、冻结参数
    ref_model = None

    # Warm start（与 full_sft 对齐）
    moe_flag = '_moe' if lm_config.use_moe else ''
    warm_ckpt = f'./out/pretrain_{lm_config.dim}{moe_flag}.pth'
    if os.path.exists(warm_ckpt):
        state = torch.load(warm_ckpt, map_location=args.device)
        unwanted_prefix = '_orig_mod.'
        if isinstance(state, dict) and 'model_state' in state:
            state = state['model_state']
        for k in list(state.keys()):
            if k.startswith(unwanted_prefix):
                state[k[len(unwanted_prefix):]] = state.pop(k)
        model.load_state_dict(state, strict=False)
        Logger(f"Loaded warm start weights from {warm_ckpt}")

        if args.kl_coef > 0.0:
            # 构建同构参考模型
            ref_model = Transformer(lm_config).to(args.device)
            ref_model.load_state_dict(state, strict=True)
            for p in ref_model.parameters():
                p.requires_grad_(False)
            ref_model.eval()
            Logger("Built frozen reference model for KL regularization.")

    # DDP 包装（只包 policy model）
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp_local_rank],
            find_unused_parameters=lm_config.use_moe
        )

    # 数据集
    ds = DpoDataset(args.pairs_path, tokenizer, max_length=lm_config.max_seq_len)
    sampler = DistributedSampler(ds) if ddp else None
    loader = DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # 优化器/AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 计算总步数（用于 warmup_ratio 与 cosine 调度）
    steps_per_epoch = len(loader)
    planned_steps = args.max_steps if args.max_steps and args.max_steps > 0 else args.epochs * steps_per_epoch
    warmup_iters_eff = get_warmup_iters(args.warmup_iters, args.warmup_ratio, planned_steps)

    global_it = 0
    stop_flag = False

    for epoch in range(args.epochs if (args.max_steps <= 0) else 10**9):
        if stop_flag:
            break
        if ddp and sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if is_main_process():
            Logger(f"Epoch {epoch+1} start... (planned total steps={planned_steps})")

        model.train()
        t0 = time.time()

        for step, batch in enumerate(loader):
            if args.max_steps > 0 and global_it >= args.max_steps:
                stop_flag = True
                break

            (X_pos, Y_pos, LM_pos, X_neg, Y_neg, LM_neg) = [b.to(args.device) for b in batch]

            # 学习率调度
            lr = get_lr(global_it, planned_steps, args.learning_rate, warmup_iters_eff)
            for g in optimizer.param_groups:
                g['lr'] = lr

            with ctx:
                # 前向：正样本
                out_pos = model(X_pos, Y_pos)
                logp_pos = seq_logprob(out_pos["logits"], Y_pos, LM_pos)

                # 前向：负样本
                out_neg = model(X_neg, Y_neg)
                logp_neg = seq_logprob(out_neg["logits"], Y_neg, LM_neg)

                # DPO 损失
                if args.loss != "dpo":
                    raise ValueError(f"--loss={args.loss} 暂不支持，仅支持 dpo")
                loss_dpo = dpo_pair_loss(logp_pos, logp_neg, beta=args.beta)

                # （新增）监督 CE：对回答段
                ce_pos = masked_ce_loss(out_pos["logits"], Y_pos, LM_pos)
                ce_neg = masked_ce_loss(out_neg["logits"], Y_neg, LM_neg)
                loss_ce = 0.5 * (ce_pos + ce_neg)

                # MoE 辅助损失
                aux_loss = 0.0
                if out_pos.get("aux_loss") is not None:
                    aux_loss = aux_loss + out_pos["aux_loss"]
                if out_neg.get("aux_loss") is not None:
                    aux_loss = aux_loss + out_neg["aux_loss"]

                # KL 正则（可选，对回答段）
                kl_loss = 0.0
                if args.kl_coef > 0.0 and ref_model is not None:
                    with torch.no_grad():
                        ref_pos = ref_model(X_pos, Y_pos)
                        ref_neg = ref_model(X_neg, Y_neg)
                    kl_pos = masked_token_kl(out_pos["logits"], ref_pos["logits"], LM_pos)  # [B]
                    kl_neg = masked_token_kl(out_neg["logits"], ref_neg["logits"], LM_neg)  # [B]
                    kl_loss = (kl_pos.mean() + kl_neg.mean()) * float(args.kl_coef)

                # 总损失：CE 权重大于 DPO；再加 KL 与 aux
                loss = float(args.dpo_coef) * loss_dpo \
                     + float(args.ce_coef)  * loss_ce \
                     + kl_loss \
                     + (aux_loss if isinstance(aux_loss, torch.Tensor) else 0.0)

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (global_it % args.log_interval == 0) and is_main_process():
                took = time.time() - t0
                t0 = time.time()
                Logger(
                    f"it={global_it}  "
                    f"dpo={float(loss_dpo):.4f}  ce={float(loss_ce):.4f}  "
                    f"aux={float(aux_loss) if isinstance(aux_loss, torch.Tensor) else 0.0:.4f}  "
                    f"kl={float(kl_loss):.4f}  total={float(loss):.4f}  lr={lr:.6f}  {took:.2f}s/it"
                )
                if args.use_wandb:
                    import wandb
                    wandb.log({
                        "loss_dpo": float(loss_dpo),
                        "loss_ce": float(loss_ce),
                        "aux_loss": float(aux_loss) if isinstance(aux_loss, torch.Tensor) else 0.0,
                        "kl_loss": float(kl_loss),
                        "loss_total": float(loss),
                        "lr": lr,
                        "it": global_it,
                        "epoch": epoch + step / max(1, steps_per_epoch),
                    })

            global_it += 1

        # 保存 checkpoint（按 epoch 存一次）
        if is_main_process():
            moe_tag = '_moe' if lm_config.use_moe else ''
            save_path = os.path.join(args.out_dir, f"dpo_{lm_config.dim}{moe_tag}_epoch{epoch+1}.pth")
            state = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
            torch.save({"model_state": state, "epoch": epoch}, save_path)
            Logger(f"Saved: {save_path}")

        if args.max_steps > 0 and stop_flag:
            break

    if args.use_wandb and is_main_process():
        import wandb as _w
        _w.finish()