"""
model_Sophia.py - 包含Sophia优化器的GPT模型实现
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# 导入LayerNorm和其他基础组件
from model_Adam import LayerNorm, CausalSelfAttention, MLP, Block, GPTConfig, GPT

class SophiaOptimizer:
    """Sophia优化器实现 (Second-order Clipped Stochastic Optimization)"""
    
    def __init__(self, params, lr=1e-3, betas=(0.96, 0.99), eps=1e-12, 
                 weight_decay=0.1, rho=1.0, update_freq=10, 
                 estimator_type='gnb', subset_size=240):
        """
        初始化Sophia优化器
        
        参数:
            params: 需要优化的参数
            lr: 学习率
            betas: 用于计算一阶矩和二阶矩的beta系数
            eps: 防止除零的小常数
            weight_decay: 权重衰减系数
            rho: 裁剪阈值
            update_freq: Hessian估计更新频率 (论文中k=10)
            estimator_type: Hessian估计器类型 ('gnb' 或 'hutchinson')
            subset_size: 用于Hessian估计的子批次大小
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       rho=rho, update_freq=update_freq, 
                       estimator_type=estimator_type, subset_size=subset_size)
        super().__init__()
        self.params = list(params)
        self.defaults = defaults
        self.state = dict()
        
        # 初始化状态
        for param in self.params:
            if param.requires_grad:
                self.state[param] = dict(
                    step=0,
                    exp_avg=torch.zeros_like(param.data),
                    exp_hessian=torch.zeros_like(param.data),
                    hessian_update_counter=0
                )
    
    def step(self, model, loss_fn, inputs, targets, subset_indices=None):
        """
        执行一步优化
        
        参数:
            model: GPT模型
            loss_fn: 损失函数
            inputs: 输入数据
            targets: 目标数据
            subset_indices: 用于Hessian估计的子集索引
        """
        # 获取配置参数
        lr = self.defaults['lr']
        beta1, beta2 = self.defaults['betas']
        eps = self.defaults['eps']
        weight_decay = self.defaults['weight_decay']
        rho = self.defaults['rho']
        update_freq = self.defaults['update_freq']
        estimator_type = self.defaults['estimator_type']
        subset_size = self.defaults['subset_size']
        
        # 计算梯度
        model.zero_grad()
        logits, loss = model(inputs, targets)
        loss.backward()
        
        # 应用权重衰减
        for param in self.params:
            if param.grad is not None and weight_decay != 0:
                param.grad.data.add_(param.data, alpha=weight_decay)
        
        # 更新参数
        for param in self.params:
            if param.grad is None:
                continue
                
            grad = param.grad.data
            state = self.state[param]
            state['step'] += 1
            
            # 更新一阶矩估计
            state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
            
            # 每update_freq步更新一次Hessian估计
            if state['step'] % update_freq == 1:
                if estimator_type == 'gnb':
                    # 使用GNB估计器估计对角Hessian
                    hessian_estimate = self._estimate_hessian_gnb(
                        model, param, inputs, targets, subset_indices, subset_size
                    )
                else:  # hutchinson
                    # 使用Hutchinson估计器估计对角Hessian
                    hessian_estimate = self._estimate_hessian_hutchinson(
                        model, param, inputs, targets
                    )
                
                # 更新Hessian的二阶矩估计
                state['exp_hessian'].mul_(beta2).add_(
                    hessian_estimate, alpha=1 - beta2
                )
                state['hessian_update_counter'] += 1
            
            # 计算预条件梯度
            denom = state['exp_hessian'].add(eps)
            
            # 应用逐元素裁剪
            update = state['exp_avg'] / denom
            update = torch.clamp(update, min=-rho, max=rho)
            
            # 更新参数
            param.data.add_(update, alpha=-lr)
    
    def _estimate_hessian_gnb(self, model, param, inputs, targets, subset_indices=None, subset_size=240):
        """
        使用GNB(Gauss-Newton-Bartlett)估计器估计对角Hessian
        
        根据论文Algorithm 2实现
        """
        model.eval()
        
        # 如果提供了子集索引，使用它们；否则随机选择子集
        if subset_indices is None:
            batch_size = inputs.size(0)
            if batch_size <= subset_size:
                subset_indices = torch.arange(batch_size)
            else:
                subset_indices = torch.randperm(batch_size)[:subset_size]
        
        # 获取子批次
        inputs_subset = inputs[subset_indices]
        targets_subset = targets[subset_indices]
        
        # 前向传播获取logits
        with torch.no_grad():
            logits, _ = model(inputs_subset)
        
        # 从模型输出中采样标签 (Gumbel softmax采样)
        probs = F.softmax(logits, dim=-1)
        sampled_labels = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
        sampled_labels = sampled_labels.view(*logits.shape[:-1])
        
        # 计算采样标签的损失和梯度
        model.zero_grad()
        logits_sampled, loss_sampled = model(inputs_subset, sampled_labels)
        loss_sampled.backward()
        
        # 获取采样损失的梯度
        grad_sampled = param.grad.data.clone() if param.grad is not None else torch.zeros_like(param.data)
        
        # 计算GNB估计: B * (∇L̂(θ) ⊙ ∇L̂(θ))
        batch_size_subset = inputs_subset.size(0)
        hessian_estimate = batch_size_subset * (grad_sampled * grad_sampled)
        
        model.train()
        return hessian_estimate
    
    def _estimate_hessian_hutchinson(self, model, param, inputs, targets):
        """
        使用Hutchinson估计器估计对角Hessian
        
        根据论文Algorithm 1实现
        """
        model.eval()
        
        # 创建随机向量u
        u = torch.randn_like(param.data)
        
        # 第一次前向传播计算梯度
        model.zero_grad()
        logits, loss = model(inputs, targets)
        loss.backward(retain_graph=True)
        grad = param.grad.data.clone() if param.grad is not None else torch.zeros_like(param.data)
        
        # 计算Hessian-vector乘积: Hv
        model.zero_grad()
        grad_dot_u = torch.sum(grad * u)
        grad_dot_u.backward(retain_graph=True)
        
        # 获取Hv
        hv = param.grad.data.clone() if param.grad is not None else torch.zeros_like(param.data)
        
        # Hutchinson估计: u ⊙ (Hv)
        hessian_estimate = u * hv
        
        model.train()
        return hessian_estimate

    def zero_grad(self):
        """清空梯度"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class GPTWithSophia(GPT):
    """扩展GPT模型以支持Sophia优化器"""
    
    def __init__(self, config):
        super().__init__(config)
    
    def configure_optimizers_sophia(self, weight_decay, learning_rate, betas, 
                                  device_type, rho=1.0, update_freq=10,
                                  estimator_type='gnb', subset_size=240):
        """
        配置Sophia优化器
        
        参数:
            weight_decay: 权重衰减
            learning_rate: 学习率
            betas: (beta1, beta2) 用于一阶矩和二阶矩
            device_type: 设备类型
            rho: 裁剪阈值
            update_freq: Hessian更新频率
            estimator_type: Hessian估计器类型 ('gnb' 或 'hutchinson')
            subset_size: Hessian估计的子批次大小
        """
        # 获取需要优化的参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # 创建优化器
        optimizer = SophiaOptimizer(
            param_dict.values(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            rho=rho,
            update_freq=update_freq,
            estimator_type=estimator_type,
            subset_size=subset_size
        )
        
        # 打印参数信息
        num_params = sum(p.numel() for p in param_dict.values())
        print(f"Sophia优化器配置:")
        print(f"  参数数量: {len(param_dict)}, 共 {num_params:,} 个参数")
        print(f"  学习率: {learning_rate}")
        print(f"  betas: {betas}")
        print(f"  权重衰减: {weight_decay}")
        print(f"  裁剪阈值(rho): {rho}")
        print(f"  Hessian更新频率: 每 {update_freq} 步")
        print(f"  Hessian估计器: {estimator_type}")
        print(f"  子批次大小: {subset_size}")
        
        return optimizer