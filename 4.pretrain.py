import os
import platform
import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from model.model import Transformer
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


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

'''
在预热之后、保持一段学习率、没有直接退火的方案
def get_lr_hold(it, all):
    warmup_iters = args.warmup_iters  # 预热迭代数
    hold_iters = int(0.1 * all)  # 设定一个保持阶段，比如总步数的10%
    lr_decay_iters = all - hold_iters  # 余弦衰减从这里开始
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters  # 线性预热

    if it < warmup_iters + hold_iters:
        return args.learning_rate  # 保持高学习率一段时间

    decay_ratio = (it - warmup_iters - hold_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_lr + coeff * (args.learning_rate - min_lr)  # 余弦退火
'''

def train_epoch(epoch, wandb):
    start_time = time.time()
    for step, (X, Y) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            out = model(X, Y)  # 可能是 dict（你现在就是），也可能是 HF 的 ModelOutput（兜底）

            # ---- 统一解包：优先使用 model.forward 已给出的 loss_total ----
            if isinstance(out, dict):
                logits     = out["logits"]
                loss_total = out.get("loss_total", None)
                loss_ce    = out.get("loss_ce", None)
                aux        = out.get("aux_loss", 0.0)
                # 回退：如果没给 loss_total，就用 loss_ce (+ aux) 组装一次
                if loss_total is None:
                    if loss_ce is None:
                        raise RuntimeError("Model output dict missing both loss_total and loss_ce.")
                    loss_ce_t = torch.as_tensor(loss_ce, device=X.device)
                    aux_t     = torch.as_tensor(aux or 0.0, device=X.device, dtype=loss_ce_t.dtype)
                    loss_total = loss_ce_t + aux_t
            else:
                # 兜底：如果有人把 forward 改回 HF 的 ModelOutput
                logits  = out.logits
                loss_ce = getattr(out, "loss", None) or getattr(out, "last_loss", None)
                aux     = getattr(out, "aux_loss", 0.0)
                loss_ce_t = torch.as_tensor(loss_ce, device=X.device) if loss_ce is not None else torch.tensor(0.0, device=X.device)
                aux_t     = torch.as_tensor(aux or 0.0, device=X.device, dtype=loss_ce_t.dtype)
                loss_total = loss_ce_t + aux_t

            loss = loss_total / args.accumulation_steps

        #with ctx:
        #    out = model(X, Y)
        #    # 主损失 + MoE辅助损失（若存在）。注意：MoE 的 aux_loss 在门控中已乘以 alpha；
        #    # 这里对逐层累加的 aux 再做层数归一化，避免层数越多惩罚越大。
        #    aux = out.aux_loss if hasattr(out, 'aux_loss') and (out.aux_loss is not None) else 0.0
        #    if isinstance(aux, float):
        #        aux = torch.tensor(aux, device=X.device, dtype=out.last_loss.dtype)
        #    if lm_config.use_moe and moe_layer_count > 0:
        #        aux = aux / moe_layer_count
        #    loss = (out.last_loss + aux) / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss_total:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            # 额外打印 CE/AUX 便于无 wandb 场景排查
            ce_scalar = float(loss_ce) if (isinstance(loss_ce, torch.Tensor)) else (float(loss_ce) if loss_ce is not None else 0.0)
            aux_scalar = float(aux)     if (isinstance(aux, torch.Tensor))     else (float(aux)     if aux     is not None else 0.0)
            Logger('  CE:{:.3f} AUX:{:.3f}'.format(ce_scalar, aux_scalar))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                ce_scalar  = float(loss_ce) if (isinstance(loss_ce, torch.Tensor)) else (float(loss_ce) if loss_ce is not None else 0.0)
                aux_scalar = float(aux)     if (isinstance(aux, torch.Tensor))     else (float(aux)     if aux     is not None else 0.0)
                log_dict = {
                    "loss_total": float(loss_total),
                    "loss_ce": ce_scalar,
                    "loss_aux": aux_scalar,
                    "lr": optimizer.param_groups[-1]['lr'],
                }
                route_load = out["routing_load_total"] if isinstance(out, dict) else getattr(out, "routing_load_total", None)
                if route_load is not None:
                    load = route_load.detach().float().cpu()
                    total = load.sum()
                    if total > 0:
                        prob = load / total
                        entropy = float((-prob * (prob + 1e-8).log()).sum().item())
                        log_dict.update({
                            "route_entropy": entropy,
                            "route_min_p": float(prob.min().item()),
                            "route_max_p": float(prob.max().item()),
                            "route_load_hist": wandb.Histogram(load.numpy())
                        })
                wandb.log(log_dict)

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.model_name}{moe_path}.pth'  # 根据是否启用MoE决定保存文件名

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            Logger(f"保存模型到 {ckp}")
            optimizer_state_path = f'{args.save_dir}/{args.model_name}{moe_path}_optimizer.pth'
            torch.save(optimizer.state_dict(), optimizer_state_path)
            Logger(f"保存优化器状态到 {optimizer_state_path}")
            
            model.train()


def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = Transformer(lm_config).to(args.device)
    moe_path = '_moe' if lm_config.use_moe else ''

    # 加入恢复训练的逻辑（MoE 运行使用带后缀的文件名）
    checkpoint_path = f'{args.save_dir}/{args.model_name}{moe_path}.pth'
    if os.path.exists(checkpoint_path):
        Logger(f"加载模型检查点 {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    else:
        Logger(f"没有找到模型检查点，开始从头训练")
    
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model



def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MateConv Pretraining")
    parser.add_argument("--deepspeed", type=str, default="ds_config.json", help="DeepSpeed config file")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="MateConv-Pretrain", help="Weights & Biases project name")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--data_path", type=str, default="./dataset/Data/seq-monkey/pretrain_data.bin", help="Path to training data")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    parser.add_argument("--model_name", type=str, default="pretrain_512", help="模型名称，用于保存和加载检查点")

    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    checkpoint_path = f'{args.save_dir}/{args.model_name}{moe_path}.pth'
    # MoE 层数，用于对逐层累加的 aux_loss 做归一化，防止层数越多惩罚越大
    moe_layer_count = lm_config.n_layers if lm_config.use_moe else 0
    
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MateConv-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    #ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    if device_type == "cuda":
        if args.dtype == "bfloat16":
            ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        elif args.dtype == "float16":
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            ctx = nullcontext()
    else:
        ctx = nullcontext()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        wandb.config.update(vars(args), allow_val_change=True)
    else:
        wandb = None

    data_path_list = [args.data_path]
    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len, memmap=True)
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

    model = init_model()
    if wandb is not None and (not ddp or ddp_local_rank == 0):
        wandb.watch(model, log="all", log_freq=max(1, args.log_interval))

    #scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 恢复优化器状态
    optimizer_state_path = f'{args.save_dir}/{args.model_name}{moe_path}_optimizer.pth'
    if os.path.exists(checkpoint_path):
        if os.path.exists(optimizer_state_path):
            Logger(f"加载优化器状态 {optimizer_state_path}")
            optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=args.device))
        else:
            Logger(f"没有找到优化器状态，使用新的优化器")

    if False and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank], find_unused_parameters = lm_config.use_moe)
    
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
