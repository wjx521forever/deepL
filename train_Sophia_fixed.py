
"""
train_Sophia_fixed.py
- 以 nanoGPT 训练脚本为基线，复现 ICLR 2024 Sophia-G 的训练方法（对角Hessian EMA + elementwise clipping）
- 重点修复你当前实现中的两类问题：
  1) 优化器 step() 内部重复 forward/backward（会破坏梯度累积并极易 OOM）
  2) Hessian 估计按“每个参数单独 forward/backward”计算（复杂度/显存不可接受）
- 本版本遵循官方 SophiaG 的实践：训练梯度正常反传；每 k 步做一次 sampled-loss 反传更新 Hessian EMA（可用更小 batch）
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_Sophia_fixed import GPTConfig, GPTWithSophia

# -----------------------------------------------------------------------------
# 配置（你可以按作业要求/显存情况再调）
out_dir = 'gpt-124M-Sophia'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb
wandb_log = True
wandb_project = 'Nanogpt-124M'
wandb_run_name = 'Sophia-G-fixed'

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 20
batch_size = 24
block_size = 1024

# model (GPT-2 small like, ~124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Sophia-G hyperparams (对齐官方思路：rho clipping + Hessian EMA)
learning_rate = 6e-4
weight_decay = 0.2
beta1 = 0.965
beta2 = 0.99
eps = 1e-12
rho = 0.05               # 逐元素裁剪阈值（官方 GPT2-125M 常用量级）
update_freq = 50         # Hessian 更新频率 k（显存紧张时调大）
hess_batch_frac = 0.5    # Hessian 更新使用更小 batch（例如 0.5=半个batch）

# training target
target_tokens = 5e7

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
backend = 'nccl'
grad_clip = 1.0

# -----------------------------------------------------------------------------
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
max_iters = int(target_tokens / tokens_per_iter) + 100
lr_decay_iters = max_iters
min_lr = learning_rate / 10
warmup_iters = 2000
decay_lr = True

print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"max_iters will be: {max_iters:,}")

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it <= lr_decay_iters:
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    return min_lr

# allow overriding from command line config
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# data loader
data_dir = os.path.join('data', dataset)

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# init
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None)
    if meta_vocab_size is not None:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPTWithSophia(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPTWithSophia(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"init_from={init_from} not supported in this fixed script.")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number of parameters: {num_params/1e6:.2f}M")

# optimizer: Sophia-G (ours)
optimizer = model.configure_optimizers_sophia(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    eps=eps,
    rho=rho
)

# resume optimizer if needed
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# grad scaler only for float16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# compile
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# ddp
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.detach()
        out[split] = losses.mean().item()
    model.train()
    return out

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.define_metric("tokens")
    wandb.define_metric("train/loss", step_metric="tokens")
    wandb.define_metric("val/loss", step_metric="tokens")
    wandb.define_metric("iter")
    wandb.define_metric("train/step_loss", step_metric="iter")
    wandb.define_metric("lr_step", step_metric="iter")
    wandb.define_metric("sophia/hessian_updates", step_metric="iter")

# -----------------------------------------------------------------------------
# training
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
hessian_updates = 0

def do_hessian_update(X_full):
    """Sophia-G Hessian EMA 更新：采样标签 + sampled loss backward -> update_hessian_from_grads."""
    nonlocal hessian_updates
    # smaller batch for hessian
    bsz = X_full.size(0)
    hb = max(1, int(bsz * hess_batch_frac))
    X_h = X_full[:hb]

    # 1) sample y from logits (no graph)
    with torch.no_grad():
        with ctx:
            logits, _ = model(X_h, None)  # forward only
    # sample without softmax to save memory
    y_sample = torch.distributions.Categorical(logits=logits).sample()

    # 2) compute sampled loss (needs graph), backward, update hessian EMA
    optimizer.zero_grad(set_to_none=True)
    with ctx:
        logits2, _ = model(X_h, None)
        loss_sampled = F.cross_entropy(
            logits2.view(-1, logits2.size(-1)),
            y_sample.view(-1),
            ignore_index=-1
        )
    # fp16 needs scaler; bf16 does not
    if scaler.is_enabled():
        scaler.scale(loss_sampled).backward()
        scaler.unscale_(optimizer)
        optimizer.update_hessian_from_grads(batch_size=hb)
        optimizer.zero_grad(set_to_none=True)
    else:
        loss_sampled.backward()
        optimizer.update_hessian_from_grads(batch_size=hb)
        optimizer.zero_grad(set_to_none=True)

    hessian_updates += 1

# memory stats
if master_process and device_type == 'cuda':
    try:
        print(f"Memory allocated before training: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Memory reserved before training: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    except Exception:
        pass

while True:
    current_tokens = iter_num * tokens_per_iter
    if current_tokens >= target_tokens:
        print(f"Reached target token count ({target_tokens/1e9:.3f}B). Stopping training.")
        break
    if iter_num > max_iters:
        break

    # lr
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # eval & ckpt
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        current_tokens = iter_num * tokens_per_iter
        progress = current_tokens / target_tokens * 100.0
        print(f"Progress: {current_tokens/1e9:.3f}B / {target_tokens/1e9:.3f}B tokens ({progress:.2f}%)")
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "tokens": current_tokens,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr_step": lr,
                "sophia/hessian_updates": hessian_updates,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # ---------------------------------------------------------
    # train one iteration (grad accumulation)
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        X, Y = get_batch('train')

    # grad clip
    if grad_clip != 0.0:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # optimizer step
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    # Sophia Hessian update every k steps (do it AFTER params updated; matches common practice)
    if update_freq > 0 and (iter_num + 1) % update_freq == 0:
        # reuse last X (already fetched), or you can fetch new batch; we reuse X to reduce IO
        do_hessian_update(X)

    # timing/log
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.detach().item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, hess_updates {hessian_updates}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/step_loss": lossf,
                "lr_step": lr,
                "sophia/hessian_updates": hessian_updates,
            })

    iter_num += 1
    local_iter_num += 1

if ddp:
    destroy_process_group()

if master_process and device_type == 'cuda':
    try:
        print(f"Final memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        print(f"Total Hessian updates: {hessian_updates}")
    except Exception:
        pass
