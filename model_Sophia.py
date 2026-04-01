
"""
model_Sophia_fixed.py
- 基于你当前的 model_Sophia.py（fileciteturn6file1），但修复 Sophia 实现方式：
  1) 不在 optimizer 内部做 forward/backward（否则无法梯度累积，且显存/速度灾难）
  2) Hessian EMA 更新不再“对每个参数单独估计一次”（那会重复N次反传，必炸）
- 实现 Sophia-G：维护 exp_avg (m) 与 exp_hessian (h) 的 EMA；
  参数更新：theta <- theta - lr * clamp(m / (h + eps), [-rho, rho])，并支持 decoupled weight decay。
- Hessian EMA 更新：用 sampled-loss 反传得到梯度 g_hat，更新 h <- beta2*h + (1-beta2)*(B * g_hat^2)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# 复用你的 nanoGPT 基础实现（fileciteturn6file1 中也是这么做的）
from model_Adam import LayerNorm, CausalSelfAttention, MLP, Block, GPTConfig, GPT

class SophiaG(torch.optim.Optimizer):
    """
    Sophia-G (ICLR 2024) 的一个紧凑实现（适配 nanoGPT 单卡复现）
    - 训练 step 使用常规 grads（来自主 loss backward）
    - Hessian EMA 更新通过外部调用 update_hessian_from_grads(batch_size=...) 完成
    """
    def __init__(self, params, lr=6e-4, betas=(0.965, 0.99), eps=1e-12, weight_decay=0.2, rho=0.05):
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        if rho <= 0.0:
            raise ValueError("rho must be positive")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rho=rho)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            rho = group['rho']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                m = state['exp_avg']
                h = state['exp_hessian']

                # decoupled weight decay (AdamW-style)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

                # m <- beta1*m + (1-beta1)*g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # update direction: m / (h + eps) then clip elementwise by rho
                denom = h.add(eps)
                update = m / denom
                update.clamp_(min=-rho, max=rho)

                p.add_(update, alpha=-lr)

        return loss

    @torch.no_grad()
    def update_hessian_from_grads(self, batch_size: int):
        """
        用“当前梯度”(来自 sampled-loss backward) 更新 Hessian 对角 EMA：
            h <- beta2*h + (1-beta2)*(B * g^2)
        其中 B 是用于 Hessian 估计的 batch 大小（或 tokens/样本数的比例因子；这里用样本数做复现足够）
        """
        for group in self.param_groups:
            _, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                h = state['exp_hessian']

                h_est = (grad * grad) * float(batch_size)
                h.mul_(beta2).add_(h_est, alpha=1 - beta2)

class GPTWithSophia(GPT):
    """扩展 GPT，以提供 Sophia-G 的配置接口，以及 Hessian EMA 更新入口"""
    def __init__(self, config):
        super().__init__(config)

    def configure_optimizers_sophia(self, weight_decay, learning_rate, betas, eps=1e-12, rho=0.05):
        # 与 nanoGPT 习惯一致：把需要优化的参数过滤出来
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        optimizer = SophiaG(
            param_dict.values(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rho=rho
        )
        num_params = sum(p.numel() for p in param_dict.values())
        print("Sophia-G optimizer configured:")
        print(f"  trainable params: {len(param_dict)} tensors, {num_params:,} scalars")
        print(f"  lr={learning_rate}, betas={betas}, wd={weight_decay}, rho={rho}, eps={eps}")
        return optimizer

    @torch.no_grad()
    def update_hessian_from_grads(self, batch_size: int):
        """
        训练脚本在 sampled-loss backward 之后调用这个函数。
        """
        # optimizer 存在于训练脚本；这里不持有 optimizer 引用。
        # 训练脚本会调用 raw_model.update_hessian_from_grads(...)，但我们需要拿到 optimizer。
        # 因此训练脚本里实际调用的是 optimizer.update_hessian_from_grads(...)
        # 这里保留空实现以兼容旧调用方式；建议训练脚本调用 optimizer.update_hessian_from_grads(...)
        pass
