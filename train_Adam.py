import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_Adam import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 配置 - 124M参数GPT模型（约为GPT-2 small的一半大小）
out_dir = 'gpt-124M-AdamW'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
# wandb logging
wandb_log = True
wandb_project = 'Nanogpt-124M'
wandb_run_name = 'Adam-v0-lr6-15B-BS480'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 20  # 适合64M模型的梯度累积步骤
batch_size = 24  # 针对24GB显存优化的批量大小
block_size = 1024
# 模型参数 - 124M GPT规格
n_layer = 12    # 减少到8层
n_head = 12     # 减少到8个注意力头
n_embd = 768   # 减少到512维嵌入
dropout = 0.0
bias = False
# 显存优化技术
gradient_checkpointing = True  # 启用梯度检查点以节省显存
# muon优化器配置
learning_rate = 6e-4  # 为较小的模型调整学习率
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# 学习率调度
decay_lr = True
warmup_iters = 2000  # 对于较小的模型，预热可以更短
# 训练目标
target_tokens = 49.2e9  # 针对较小模型设置适当的目标
# 系统设置
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # 启用编译提高训练速度
optimizer_in_cpu = False  # 将优化器状态放在GPU上以加快训练速度
backend = 'nccl'  # ddp backend
# -----------------------------------------------------------------------------

# 计算训练迭代次数和学习率调度参数
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size  # 假设非DDP环境
max_iters = int(target_tokens / tokens_per_iter) + 100
lr_decay_iters = max_iters
min_lr = learning_rate / 10  # 最小学习率为初始学习率的1/10
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"max_iters will be: {max_iters:,}")


# 学习率调度器
def get_lr(it):
    # 1) 线性预热阶段
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) 余弦衰减到lr_decay_iters
    if it <= lr_decay_iters:
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    # 3) lr_decay_iters后保持最小学习率
    return min_lr

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# DDP 设置
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
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32提高性能
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 数据加载器
data_dir =  os.path.join('data', dataset)
# data_dir =  'autodl-tmp'
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

# 初始化变量
iter_num = 0
best_val_loss = 1e9

# 尝试获取词汇表大小
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# 模型初始化
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # 打印模型参数数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {num_params/1e6:.2f}M")
    
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# 启用梯度检查点
if gradient_checkpointing and hasattr(model, 'transformer'):
    model.transformer.gradient_checkpointing = True
    print("Gradient checkpointing enabled")

# 创建优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
print("Optimizer states placed on GPU")

# 初始化GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# 编译模型
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# 包装为DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 估计损失函数
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 在wandb初始化部分添加iter作为step指标
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.define_metric("tokens")
    wandb.define_metric("val/loss", step_metric="tokens")
    wandb.define_metric("train/loss", step_metric="tokens")
    # 添加以iter为步骤的指标
    wandb.define_metric("iter")
    wandb.define_metric("val/loss_by_step", step_metric="iter")
    wandb.define_metric("train/loss_by_step", step_metric="iter")
    wandb.define_metric("train/step_loss", step_metric="iter")
    wandb.define_metric("lr_step", step_metric="iter")

# 训练循环
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
total_tokens = 0  # 在循环外定义总token计数

# 尝试测量一下显存使用情况
if master_process:
    try:
        print(f"Memory allocated before training: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Memory reserved before training: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    except:
        print("Unable to query CUDA memory stats")

while True:
    # 终止条件
    current_tokens = iter_num * tokens_per_iter
    if current_tokens >= target_tokens:
        print(f"Reached target token count ({target_tokens/1e9:.1f}B). Stopping training.")
        break
    if iter_num > max_iters:
        break
    
    # 设置学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 评估和检查点
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        current_tokens = iter_num * tokens_per_iter
        progress = current_tokens / target_tokens * 100.0
        print(f"Progress: {current_tokens/1e9:.3f}B / {target_tokens/1e9:.1f}B tokens ({progress:.2f}%)")
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            # 更新日志以包含以step为基础的指标
            wandb.log({
                "iter": iter_num,
                "tokens": current_tokens,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/loss_by_step": losses['train'],  # 添加以iter为基础的指标
                "val/loss_by_step": losses['val'],      # 添加以iter为基础的指标
                "lr": lr,
                "lr_step": lr,                          # 添加以iter为基础的学习率
                "mfu": running_mfu*100,
                "token_progress": progress,
                "tokens_processed_billions": current_tokens/1e9,
                "memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3)
            })
            
            # 添加tokens为x轴的折线图
            wandb.log({"loss_vs_tokens": wandb.plot.line(
                table=wandb.Table(columns=["tokens", "val_loss", "train_loss"],
                                data=[[current_tokens, losses['val'], losses['train']]]),
                x="tokens",
                y=["val_loss", "train_loss"],
                title="Loss vs. Tokens"
            )})
            
            # 添加iter为x轴的折线图
            wandb.log({"loss_vs_step": wandb.plot.line(
                table=wandb.Table(columns=["iter", "val_loss", "train_loss"],
                                data=[[iter_num, losses['val'], losses['train']]]),
                x="iter",
                y=["val_loss", "train_loss"],
                title="Loss vs. Step"
            )})
            
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

    # 前向传播、反向传播和优化
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    
    # 梯度裁剪
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # 优化器步骤
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # 计时和日志
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.detach().item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        # 添加每个log_interval的简单损失记录到wandb
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/step_loss": lossf,  # 更频繁记录训练损失
                "step_time_ms": dt*1000,
                "lr_step": lr,  # 记录每步的学习率
                "memory_snapshot_gb": torch.cuda.memory_allocated() / (1024**3)
            })
    
    iter_num += 1
    local_iter_num += 1
    
    # 定期检查是否接近显存限制
    if iter_num % 100 == 0 and master_process:
        mem_allocated = torch.cuda.memory_allocated() / (1024**3)
        if mem_allocated > 22.0:  # 如果接近24GB限制，打印警告
            print(f"WARNING: High memory usage: {mem_allocated:.2f} GB")

if ddp:
    destroy_process_group()

# 打印最终显存统计
if master_process:
    print(f"Final memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")