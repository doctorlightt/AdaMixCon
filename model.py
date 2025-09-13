import sys
import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import numpy as np
import random
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import torchvision
import os


class Inter_att(nn.Module):
    
    def __init__(self, in_channels, hidden_dim=64, dropout=0.0):
        super(Inter_att, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        self.q_proj = nn.Linear(in_channels, hidden_dim)
        self.k_proj = nn.Linear(in_channels, hidden_dim)
        self.v_proj = nn.Linear(in_channels, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, in_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)
    
    def cosine_similarity(self, Q, K):
        Q_norm = F.normalize(Q, p=2, dim=-1) 
        K_norm = F.normalize(K, p=2, dim=-1) 
        return torch.matmul(Q_norm, K_norm.t())

    def forward(self, x):
        x = x.permute(0,2,1).contiguous()
        b, c, N = x.shape
        
        x_pool = x.mean(dim=2)

        attn_scores = self.cosine_similarity(x_pool, x_pool)
        mask = torch.eye(b, dtype=torch.bool, device=x.device)
        attn_scores = attn_scores.masked_fill(mask, 0)

        max_values, max_indices = torch.max(attn_scores, dim=-1, keepdim=True)
        attn = torch.full_like(attn_scores, 0)
        attn.scatter_(dim=-1, index=max_indices, src=max_values)

        attn = self.dropout(attn)
        
        agg = torch.matmul(attn, x_pool) 

        agg_out = agg.unsqueeze(2).expand(-1, -1, N)
        out = x + agg_out
        
        return out.permute(0,2,1).contiguous()

class relation_weight(nn.Module):
    def __init__(self):
        super(relation_weight, self).__init__()
        self.w = nn.Parameter(torch.tensor([1.0,1.0]),requires_grad=True)

    def forward(self,x1,x2):
        x = (self.w[0] * x1+self.w[1] * x2)/(self.w[0]+self.w[1])
        return x 
    
class relation_weight_3(nn.Module):
    def __init__(self):
        super(relation_weight_3, self).__init__()
        self.w = nn.Parameter(torch.tensor([1.0,1.0,1.0]),requires_grad=True)

    def forward(self,x1,x2,x3):
        x = (self.w[0] * x1+self.w[1] * x2+self.w[2] * x3)/(self.w[0]+self.w[1]+self.w[2])
        return x 

class relation_weight_4(nn.Module):
    def __init__(self):
        super(relation_weight_4, self).__init__()
        self.w = nn.Parameter(torch.tensor([1.0,1.0,1.0,1.0]),requires_grad=True)

    def forward(self,x1,x2,x3,x4):
        x = (self.w[0] * x1+self.w[1] * x2+self.w[2] * x3+self.w[3] * x4)/(self.w[0]+self.w[1]+self.w[2]+self.w[3])
        return x 

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    
class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=2):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([Mlp(in_features=input_size, hidden_features=hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        x_idt = x
        if len(x_idt.shape) == 4:
            B, H, W, C = x.shape
        else:
            B, L, C = x.shape
        x = x.view(-1, C)
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        if len(x_idt.shape) == 4:
            y = dispatcher.combine(expert_outputs).view(B, H, W, C).contiguous()
        else:
            y = dispatcher.combine(expert_outputs).view(B, L, C).contiguous()
        return y, loss

class imagetoclass(nn.Module):
    def __init__(self, N_class, feature_dim):
        super(imagetoclass, self).__init__()
        self.n_class = N_class
        self.feature_dim = feature_dim

    def compute_similarity(self, Si, Q, mode="dot"):
        Si = Si.permute(0, 2, 1) 
        Q = Q 
        if mode == "dot":
            return torch.matmul(Si.permute(0, 2, 1), Q)
        
        if mode == "cosine":
            Si_norm = F.normalize(Si, p=2, dim=1) 
            Q_norm = F.normalize(Q, p=2, dim=1)  
            return torch.einsum("bdi,bdj->bij", Si_norm, Q_norm) 

    def cal_L(self, S, Q, task_index, special_l, mode, k, k2):
        special_list = special_l.copy() 
        S_add_T_norm = [] 
        count = 0
        for i in range(self.n_class): 
            Si = []
            num_samples = S.size(0)//self.n_class
            Si = S[count:count + num_samples]  
            count += num_samples
            Si = torch.split(Si, 1, dim=0)  
            Si = torch.cat(Si, dim=2) 
            Si = Si.view(Si.size(0), Si.size(1), -1) 
            Si = torch.transpose(Si, 1, 2) 
            support_set_sam = Si[0]
            support_set_sam_norm = torch.norm(support_set_sam, 2, 1, True)
            support_set_sam = support_set_sam / support_set_sam_norm
            S_add_T_norm.append(support_set_sam.unsqueeze(0))

        Q = Q.view(Q.size(0), Q.size(1), -1) 
        Q_norm = [] 
        for i in range(Q.size(0)): 
            query_set_sam = Q[i]
            query_set_sam_norm = torch.norm(query_set_sam, 2, 0, True)
            query_set_sam = query_set_sam / query_set_sam_norm
            Q_norm.append(query_set_sam.unsqueeze(0))

        b = [] 
        Q_norm = torch.cat(Q_norm, dim=0) 
        similarity_scores = []
        for i in range(len(S_add_T_norm)): 
            Si = torch.tensor(S_add_T_norm[i])
            similarity_matrix = self.compute_similarity(Si, Q_norm, "dot")
            top_k_values, indices = torch.topk(similarity_matrix, k, 1)  
            top_k_values, indices = torch.topk(top_k_values, k2, 2)
            similarity_scores.append(top_k_values.unsqueeze(0))

        similarity_scores = torch.cat(similarity_scores, dim=0)
        similarity_scores = similarity_scores.unsqueeze(2)

        return similarity_scores
    
    def forward(self, support, query, task_index,special_list,mode,k, k2):
        _, _, h2, w2 = query.size() 
        L = self.cal_L(support, query,task_index,special_list, mode, k, k2).view(-1, 1, k, k2)
        return L

class imagetoclass2(nn.Module):
    def __init__(self, feature_dim):
        super(imagetoclass2, self).__init__()
        self.feature_dim = feature_dim

    def compute_similarity(self, Si, Q, mode="dot"):

        Si = Si.permute(0, 2, 1) 
        Q = Q 
        if mode == "dot":
            return torch.matmul(Si.permute(0, 2, 1), Q)
        
        if mode == "cosine":
            Si_norm = F.normalize(Si, p=2, dim=1) 
            Q_norm = F.normalize(Q, p=2, dim=1) 
            return torch.einsum("bdi,bdj->bij", Si_norm, Q_norm)

    def cal_L(self, S, Q, task_index, special_l, mode, k, k2):
        special_list = special_l.copy() 
        S_add_T_norm = [] 
        count = 0
        for i in range(S.size(0)): 
            Si = []
            num_samples = 1
            Si = S[count:count + num_samples]  
            count += num_samples
            Si = torch.split(Si, 1, dim=0)  
            Si = torch.cat(Si, dim=2) 
            Si = Si.view(Si.size(0), Si.size(1), -1) 
            Si = torch.transpose(Si, 1, 2) 
            support_set_sam = Si[0]
            support_set_sam_norm = torch.norm(support_set_sam, 2, 1, True)
            support_set_sam = support_set_sam / support_set_sam_norm
            S_add_T_norm.append(support_set_sam.unsqueeze(0))

        Q = Q.view(Q.size(0), Q.size(1), -1) 
        Q_norm = [] 
        for i in range(Q.size(0)): 
            query_set_sam = Q[i]
            query_set_sam_norm = torch.norm(query_set_sam, 2, 0, True)
            query_set_sam = query_set_sam / query_set_sam_norm
            Q_norm.append(query_set_sam.unsqueeze(0))

        b = [] 
        Q_norm = torch.cat(Q_norm, dim=0) 
        similarity_scores = []
        for i in range(len(S_add_T_norm)): 
            Si = torch.tensor(S_add_T_norm[i])
            similarity_matrix = self.compute_similarity(Si, Q_norm, "dot") 
            top_k_values, indices = torch.topk(similarity_matrix, k, 1) 
            top_k_values, indices = torch.topk(top_k_values, k2, 2)
            similarity_scores.append(top_k_values.unsqueeze(0))

        similarity_scores = torch.cat(similarity_scores, dim=0)
        similarity_scores = similarity_scores.unsqueeze(2)

        return similarity_scores
    
    def forward(self, support, query, task_index,special_list,mode,k, k2):
        L = self.cal_L(support, query,task_index,special_list, mode, k, k2).view(-1, 1, k, k2)
        return L

class Initial_embedding(nn.Module):
    def __init__(self,input_channels,feature_dim=64):
        super(Initial_embedding, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_channels, feature_dim, kernel_size=1, padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU(),
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU(),
                        )
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        result1 = self.layer1(x)
        result2 = self.layer2(result1)
        return result2

class DownEncoder(nn.Module):
    def __init__(self,input_channels,feature_dim=64):
        super(DownEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_channels, feature_dim, kernel_size=1, padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU(),
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU(),
                        )
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        result1 = self.layer1(x)
        if result1.size(2) >= 6:
            result1 = self.pool(result1)
        result2 = self.layer2(result1)
        if result2.size(2) >= 6:
            result2 = self.pool(result2)
        return result2

class AdaptiveAttSpec(nn.Module):
    def __init__(self, in_channels, reduction=8, online_lr=1e-5):
        super(AdaptiveAttSpec, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.online_lr = online_lr

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.adaptive_matrix = nn.Parameter(torch.randn(1, in_channels)) 

    def forward(self, x):
        B, HW, C = x.shape
        x_perm = x.permute(0, 2, 1)

        avg_out = self.avg_pool(x_perm).view(B, C)
        avg_out = self.fc2(self.relu(self.fc1(avg_out))) 

        max_out = self.max_pool(x_perm).view(B, C)
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        att = self.sigmoid(avg_out + max_out) 

        att_expanded = att.view(B, C)
        adaptive_adjust = att_expanded * self.adaptive_matrix 
        new_att = self.sigmoid(att + adaptive_adjust) 

        if not self.training:
            error = (new_att - att_expanded) 
            update = self.online_lr * torch.mean(error * att_expanded, dim=0, keepdim=True) 
            self.adaptive_matrix.data = (1 - self.online_lr) * self.adaptive_matrix.data + update

        new_att = new_att.view(B, C, 1)
        out = new_att * x_perm 
        return out.permute(0, 2, 1)

class AdaptiveAttSpa(nn.Module):
    def __init__(self, kernel_size=3, spatial_size=[15,15], online_lr=1e-5):

        super(AdaptiveAttSpa, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.online_lr = online_lr
        self.spatial_size = spatial_size

        if spatial_size is not None:
            H, W = spatial_size
            self.num_spatial = H * W
            self.adaptive_weight = nn.Parameter(torch.randn(1, self.num_spatial))
        else:
            self.adaptive_weight = None

    def forward(self, x):
        B, C, H, W = x.shape
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        cat = torch.cat([avg_out, max_out], dim=1)       
        att = self.sigmoid(self.conv(cat))             

        if self.adaptive_weight is not None:
            att_flat = att.view(B, -1)
            adaptive_adjust = att_flat * self.adaptive_weight 
            new_att_flat = torch.sigmoid(att_flat + adaptive_adjust)
            new_att = new_att_flat.view(B, 1, H, W)
            
            if not self.training:
                error = (new_att_flat - att_flat).unsqueeze(2) 
                update = self.online_lr * torch.mean(error.squeeze(2) * att_flat, dim=0, keepdim=True) 
                self.adaptive_weight.data = (1 - self.online_lr) * self.adaptive_weight.data + update

            out = new_att * x
        else:
            out = att * x

        return out

class AdaptiveAttSpa_1d(nn.Module):
    def __init__(self, kernel_size=3, length=9, online_lr=1e-5):

        super(AdaptiveAttSpa_1d, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.online_lr = online_lr
        self.length = length

        if length is not None:
            self.adaptive_weight = nn.Parameter(torch.randn(1, length))
        else:
            self.adaptive_weight = None

    def forward(self, x):
        B, C, L = x.shape
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)   
        cat = torch.cat([avg_out, max_out], dim=1)        
        att = self.sigmoid(self.conv(cat))          

        if self.adaptive_weight is not None:
            att_flat = att.view(B, L)
            adaptive_adjust = att_flat * self.adaptive_weight 
            new_att_flat = torch.sigmoid(att_flat + adaptive_adjust)   
            new_att = new_att_flat.view(B, 1, L)

            if not self.training:
                error = (new_att_flat - att_flat).unsqueeze(2)
                update = self.online_lr * torch.mean(error.squeeze(2) * att_flat, dim=0, keepdim=True) 
                self.adaptive_weight.data = (1 - self.online_lr) * self.adaptive_weight.data + update

            out = new_att * x
        else:
            out = att * x

        return out
    
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size=[64, 21], hidden_size=8, loss_type='softmax'):
        super(RelationNetwork, self).__init__()

        self.loss_type = loss_type
        # when using Resnet,
        # conv map without avgpooling is 7x7, need padding in block to do pooling
        padding = 1 if (input_size[1] < 10) else 0
        shrink_s = lambda s: int((int((s - 2 + 2 * padding) / 2) - 2 + 2 * padding) / 2)

        self.layer1 = nn.Sequential(
                        nn.Conv1d(input_size[0], input_size[0], kernel_size=3, padding=padding),
                        nn.BatchNorm1d(input_size[0], affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(input_size[0],input_size[0], kernel_size=3, padding=padding),
                        nn.BatchNorm1d(input_size[0], affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.fc1 = nn.Linear(input_size[0]*shrink_s(input_size[1]), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.loss_type == 'mse':
            out = F.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)
        return out

class RelationNetwork2(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size=[64, 21, 21], hidden_size=8, loss_type='softmax'):
        super(RelationNetwork2, self).__init__()

        self.loss_type = loss_type
        # when using Resnet,
        # conv map without avgpooling is 7x7, need padding in block to do pooling
        padding = 1 if (input_size[1] < 10) and (input_size[2] < 10) else 0
        shrink_s = lambda s: int((int((s - 2 + 2 * padding) / 2) - 2 + 2 * padding) / 2)

        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2, input_size[0], kernel_size=3, padding=padding),
                        nn.BatchNorm2d(input_size[0], affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(input_size[0],input_size[0], kernel_size=3, padding=padding),
                        nn.BatchNorm2d(input_size[0], affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size[0]*shrink_s(input_size[1])*shrink_s(input_size[1]), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.loss_type == 'mse':
            out = F.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)
        return out


class Mlp(nn.Module):
    """ Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        patch_size,
        len=225,
        d_state=16,
        ratio=4,
        layer_idx=0,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.len = len
        self.d_model = d_model
        self.d_state = d_state
        self.ratio = ratio  # downsample/window size
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.sum = relation_weight()

        self.ada_att_spec = AdaptiveAttSpec(self.d_inner, reduction=self.d_inner)
        self.ada_att_spa = AdaptiveAttSpa(kernel_size=3, spatial_size=patch_size)            
        
        self.in_proj_q = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_s = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.conv2d_q = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d_s = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        
        self.act = nn.SiLU()
        
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
  
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True) # (K, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True) # (K, D, N)

        self.x_proj_spectral = (
            nn.Linear(self.len, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.len, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight_spectral = nn.Parameter(torch.stack([t.weight for t in self.x_proj_spectral], dim=0)) # (K, N, inner)
        del self.x_proj_spectral

        self.dt_projs_spectral = (
            self.dt_init(self.dt_rank, self.len, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.len, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight_spectral = nn.Parameter(torch.stack([t.weight for t in self.dt_projs_spectral], dim=0)) # (K, inner, rank)
        self.dt_projs_bias_spectral = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_spectral], dim=0)) # (K, inner)
        del self.dt_projs_spectral
        
        self.A_logs_spectral = self.A_log_init(self.d_state, self.len, copies=2, merge=True) # (K, D, N)
        self.Ds_spectral = self.D_init(self.len, copies=2, merge=True)
        
        self.selective_scan = selective_scan_fn

        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.BatchNorm2d(self.d_inner)
        
        self.out_proj_q = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_s = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        self.inter_att_spatial = Inter_att(self.len)
        self.inter_att_spectral = Inter_att(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def self_mamba_spatial(self, x):
        B, C, H, W = x.shape 
        L = H * W
        K = 1
        
        x = self.inter_att_spatial(x.view(B, C, -1)).view(B, C, H, W)
    
        xs = x.view(B, 1, -1, L)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        
        out_y = self.ada_att_spa(out_y.view(B, -1, H, W))
        out_y = out_y.view(B, -1, L)
        
        out_y = out_y.view(B, K, -1, L)  # (b, 4, c, l)
        assert out_y.dtype == torch.float

        return out_y[:, 0]
    
    
    def self_mamba_spectral(self, x):
        B, C, H, W = x.shape 
        
        x = self.inter_att_spectral(x.view(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        
        L = C
        K = 2
        x = x.permute(0,2,3,1).contiguous()

        xs = torch.stack([x.view(B, -1, L), torch.flip(x.view(B, -1, L), dims=[-1])], dim=1).contiguous()
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight_spectral.contiguous())
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight_spectral.contiguous())

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds_spectral.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs_spectral.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias_spectral.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        
        out_y = self.ada_att_spec(out_y)
        
        out_y = out_y.view(B, K, -1, L) 
        assert out_y.dtype == torch.float
        
        return (out_y[:, 0].view(B, -1, L).permute(0,2,1).contiguous(), out_y[:, 1].view(B, -1, L).permute(0,2,1).contiguous())
    
    def channel_wise_cross_attention(self, x1, x2):
        B, C, L = x1.shape
        x1_norm = F.normalize(x1, dim=2)
        x2_norm = F.normalize(x2, dim=2) 
        sim_matrix = torch.matmul(x1_norm, x2_norm.transpose(1, 2)) 
        attn_weights = F.softmax(sim_matrix, dim=-1)
        output = torch.matmul(attn_weights, x2)  
        return output

    def forward_core(self, x_q, x_s):

        B, C, H, W = x_q.size()
        B_s, _, _, _ = x_s.size()
        
        q1 = self.self_mamba_spatial(x_q)
        s1 = self.self_mamba_spatial(x_s)
        (q5, q6) = self.self_mamba_spectral(x_q)
        (s5, s6) = self.self_mamba_spectral(x_s)
        y_q_spatial = q1
        y_q_spectral = q5 + q6
        y_s_spatial = s1
        y_s_spectral = s5 + s6
        y_q_spectral = self.ada_att_spec(y_q_spectral.transpose(-1, -2)).transpose(-1, -2)
        y_s_spectral = self.ada_att_spec(y_s_spectral.transpose(-1, -2)).transpose(-1, -2)
        y_q_spatial_new = self.channel_wise_cross_attention(y_q_spectral.transpose(-1, -2), y_q_spatial.transpose(-1, -2)).transpose(-1, -2)
        y_q_spectral_new = self.channel_wise_cross_attention(y_q_spatial, y_q_spectral)
        y_s_spatial_new = self.channel_wise_cross_attention(y_s_spectral.transpose(-1, -2), y_s_spatial.transpose(-1, -2)).transpose(-1, -2)
        y_s_spectral_new = self.channel_wise_cross_attention(y_s_spatial, y_s_spectral)
        y_q = self.sum(y_q_spatial_new, y_q_spectral_new)
        y_s = self.sum(y_s_spatial_new, y_s_spectral_new)
        
        return y_q, y_s

    def forward(self, q_feat, s_feat):

        B, H, W, C = q_feat.shape
        B_s, H, W, C = s_feat.shape

        # input projection
        xz_q = self.in_proj_q(q_feat)
        xz_s = self.in_proj_s(s_feat)
        
        x_q, z_q = xz_q.chunk(2, dim=-1)
        x_s, z_s = xz_s.chunk(2, dim=-1)

        # depth-wise convolution
        x_q = x_q.permute(0, 3, 1, 2).contiguous()
        x_s = x_s.permute(0, 3, 1, 2).contiguous()
        
        x_q = self.act(self.conv2d_q(x_q)) 
        x_s = self.act(self.conv2d_s(x_s)) 
        
        # mamba core
        y_q, y_s = self.forward_core(x_q, x_s)

        y_q = torch.transpose(y_q, dim0=1, dim1=2).contiguous().view(B, H, W, -1) 
        y_s = torch.transpose(y_s, dim0=1, dim1=2).contiguous().view(B_s, H, W, -1) 
        
        y_q = self.out_norm_1(y_q)
        y_s = self.out_norm_1(y_s)
        
        # silu activation
        y_q = y_q * F.silu(z_q)
        y_s = y_s * F.silu(z_s)
        
        # output projection
        out_q = self.out_proj_q(y_q) 
        out_s = self.out_proj_s(y_s) 
        if self.dropout is not None:
            out_q = self.dropout(out_q)
            out_s = self.dropout(out_s)
        return out_q, out_s

class SS2D_q(nn.Module):
    def __init__(
        self,
        d_model,
        patch_size,
        len=225,
        d_state=16,
        ratio=4,
        layer_idx=0,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.len = len
        self.d_model = d_model
        self.d_state = d_state
        self.ratio = ratio  # downsample/window size
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.sum = relation_weight()

        self.ada_att_spec = AdaptiveAttSpec(self.d_inner, reduction=self.d_inner)
        self.ada_att_spa = AdaptiveAttSpa_1d(kernel_size=3)       
        
        self.in_proj_q = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.conv1d_q = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        
        self.act = nn.SiLU()
        
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True) # (K, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True) # (K, D, N)

        self.x_proj_spectral = (
            nn.Linear(self.len, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.len, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight_spectral = nn.Parameter(torch.stack([t.weight for t in self.x_proj_spectral], dim=0)) # (K, N, inner)
        del self.x_proj_spectral

        self.dt_projs_spectral = (
            self.dt_init(self.dt_rank, self.len, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.len, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight_spectral = nn.Parameter(torch.stack([t.weight for t in self.dt_projs_spectral], dim=0)) # (K, inner, rank)
        self.dt_projs_bias_spectral = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_spectral], dim=0)) # (K, inner)
        del self.dt_projs_spectral
        
        self.A_logs_spectral = self.A_log_init(self.d_state, self.len, copies=2, merge=True) # (K, D, N)
        self.Ds_spectral = self.D_init(self.len, copies=2, merge=True)
        
        self.selective_scan = selective_scan_fn

        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.BatchNorm2d(self.d_inner)
        
        self.out_proj_q = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        
        self.inter_att_spatial = Inter_att(self.len)
        self.inter_att_spectral = Inter_att(self.d_inner)
        
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    
    def self_mamba_spatial(self, x):
        B, C, L = x.shape
        K = 2

        x = self.inter_att_spatial(x.view(B, C, -1))
        
        xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1).view(B, 2, -1, L)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        
        out_y = self.ada_att_spa(out_y)
        
        out_y = out_y.view(B, K, -1, L)  # (b, 4, c, l)
        assert out_y.dtype == torch.float
        
        return (out_y[:, 0], out_y[:, 1])
    
    def self_mamba_spectral(self, x):
        B, C, L = x.shape #1,512,64,64
        
        x = self.inter_att_spectral(x.view(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, L)
        
        L = C
        K = 2
        x = x.permute(0,2,1).contiguous()

        xs = torch.stack([x.view(B, -1, L), torch.flip(x.view(B, -1, L), dims=[-1])], dim=1).contiguous()
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight_spectral.contiguous())
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight_spectral.contiguous())

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds_spectral.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs_spectral.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias_spectral.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        
        out_y = self.ada_att_spec(out_y)
        
        out_y = out_y.view(B, K, -1, L) 
        assert out_y.dtype == torch.float

        return (out_y[:, 0].view(B, -1, L).permute(0,2,1).contiguous(), out_y[:, 1].view(B, -1, L).permute(0,2,1).contiguous())
    
    def channel_wise_cross_attention(self, x1, x2):
        B, C, L = x1.shape
        x1_norm = F.normalize(x1, dim=2)
        x2_norm = F.normalize(x2, dim=2)
        sim_matrix = torch.matmul(x1_norm, x2_norm.transpose(1, 2)) 
        attn_weights = F.softmax(sim_matrix, dim=-1)
        output = torch.matmul(attn_weights, x2) 
        return output
        
    def forward_core(self, x_q):
        B, C, L = x_q.size()
        (q1, q2) = self.self_mamba_spatial(x_q)
        (q5, q6) = self.self_mamba_spectral(x_q)
        y_q_spatial = q1 + q2
        y_q_spectral = q5 + q6
        y_q_spatial = self.ada_att_spa(y_q_spatial)
        y_q_spectral = self.ada_att_spec(y_q_spectral.transpose(-1, -2)).transpose(-1, -2)
        y_q_spatial_new = self.channel_wise_cross_attention(y_q_spectral.transpose(-1, -2), y_q_spatial.transpose(-1, -2)).transpose(-1, -2)
        y_q_spectral_new = self.channel_wise_cross_attention(y_q_spatial, y_q_spectral)
        y_q = self.sum(y_q_spatial_new, y_q_spectral_new)

        return y_q

    def forward(self, q_feat):
        B, L, C = q_feat.shape

        # input projection
        xz_q = self.in_proj_q(q_feat)
        
        x_q, z_q = xz_q.chunk(2, dim=-1)

        # depth-wise convolution
        x_q = x_q.permute(0, 2, 1).contiguous()
        
        x_q = self.act(self.conv1d_q(x_q))
        
        # mamba core
        y_q = self.forward_core(x_q)

        y_q = torch.transpose(y_q, dim0=1, dim1=2).contiguous().view(B, L, -1)  
        
        y_q = self.out_norm_1(y_q)
        
        # silu activation
        y_q = y_q * F.silu(z_q)
        
        # output projection
        out_q = self.out_proj_q(y_q) 

        if self.dropout is not None:
            out_q = self.dropout(out_q)

        return out_q

class VSSBlock(nn.Module):
    def __init__(
        self,
        spatial_len: int = 225,
        patch_size: list = [25,25],
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        ratio: int = 4,
        layer_idx: int = 0,
        mlp_ratio: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(patch_size=patch_size, d_model=hidden_dim, len=spatial_len, dropout=attn_drop_rate, d_state=d_state, ratio=ratio, layer_idx=layer_idx, **kwargs)
        
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        
        self.mlp_q = MoE(input_size=hidden_dim, output_size=None, num_experts=32, k=16, hidden_size=hidden_dim * mlp_ratio)
        self.mlp_s = MoE(input_size=hidden_dim, output_size=None, num_experts=32, k=16, hidden_size=hidden_dim * mlp_ratio)

    def forward(self, q_feat, s_feat):
        q_skip = q_feat
        s_skip = s_feat 
       
        q_feat, s_feat = self.self_attention(self.ln_1(q_feat), self.ln_1(s_feat))
        
        q_feat = q_skip + self.drop_path(q_feat) 
        s_feat = s_skip + self.drop_path(s_feat) 
        
        q_mlp, loss_q = self.mlp_q(self.ln_2(q_feat))
        s_mlp, loss_s = self.mlp_s(self.ln_2(s_feat))
        
        q_feat = q_feat + self.drop_path(q_mlp)
        s_feat = s_feat + self.drop_path(s_mlp)
        
        return q_feat, s_feat, loss_q + loss_s

class VSSBlock_q(nn.Module):
    def __init__(
        self,
        spatial_len: int = 225,
        patch_size: list = [25,25],
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        ratio: int = 4,
        layer_idx: int = 0,
        mlp_ratio: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_q(patch_size = patch_size, d_model=hidden_dim, len=spatial_len, dropout=attn_drop_rate, d_state=d_state, ratio=ratio, layer_idx=layer_idx, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp_q = MoE(input_size=hidden_dim, output_size=None, num_experts=16, k=2, hidden_size=hidden_dim * mlp_ratio)

    def forward(self, q_feat):
        q_skip = q_feat 
       
        q_feat= self.self_attention(self.ln_1(q_feat))
        
        q_feat = q_skip + self.drop_path(q_feat) 
        
        q_mlp, loss_q = self.mlp_q(self.ln_2(q_feat))

        q_feat = q_feat + self.drop_path(q_mlp)
        
        return q_feat, loss_q


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self, 
        patch_size,
        k2,
        i_layer, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        d_state=16,
        mlp_ratio=1,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.i_layer = i_layer
        self.ratio = patch_size[0]
        self.k2 = k2
        
        self.blocks = nn.ModuleList([
            VSSBlock(
                spatial_len = patch_size[0] * patch_size[1],
                patch_size = patch_size,
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                ratio=self.ratio,
                layer_idx=self.i_layer,
                mlp_ratio=mlp_ratio
            ) if self.i_layer == 0 else VSSBlock_q(
                spatial_len = self.k2,
                patch_size = patch_size,
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                ratio=self.ratio,
                layer_idx=self.i_layer,
                mlp_ratio=mlp_ratio
            )    
            for i in range(depth)])
        
        if True: 
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

    def forward(self, q_feat, s_feat):

        for blk in self.blocks:
            if s_feat is not None:
                q_feat, s_feat, loss = blk(q_feat, s_feat)
                return q_feat, s_feat, loss
            else:
                q_feat, loss = blk(q_feat)
                return q_feat, loss
        

class DASR(nn.Module):
    def __init__(self, dims = [64,64], patch_size = [15,15], num_features = 64, thres = 0.4):
        super().__init__()
        self.thres = thres
        self.dims = dims
        self.patch_size = [patch_size, patch_size]
        self.relation_net1 = RelationNetwork2(input_size=[self.dims[0], self.patch_size[0][0], self.patch_size[0][1]])
        self.relation_net2 = RelationNetwork2(input_size=[self.dims[0], self.patch_size[0][0], self.patch_size[0][1]])
        
        self.norm = nn.BatchNorm2d(num_features)
        self.norm_2 = nn.BatchNorm2d(num_features)
        
    def forward(self, q_local, s_local, q_global, s_global, support_num):
        B, _, _, _ = q_local.size()
        B_s, _, _, _ = s_local.size()
        
        #### global metric ####
        q_feat = q_global
        s_feat = s_global
        
        q_list = []
        s_feat = s_feat.repeat(B,1,1,1)
        for i in range(B):
            q_list.append(q_feat[i:i+1].repeat(B_s, 1, 1, 1))
        q_feat = torch.cat(q_list, dim=0)
        
        feat = self.norm(torch.cat([q_feat, s_feat], dim=0))
        q_feat = feat[:q_feat.shape[0]]
        s_feat = feat[-s_feat.shape[0]:]
        
        feat_pairs = torch.cat((q_feat, s_feat), dim=1).contiguous()
        output_global = self.relation_net1(feat_pairs)
        
        output_global_idt = output_global.contiguous().view(B, output_global.size(0)//B//support_num, support_num)
        
        output_global_reshaped = output_global.view(output_global.size(0)//support_num, support_num)
        output_global = output_global_reshaped.mean(dim=1)
        output_global = output_global.contiguous().view(B,output_global.size(0)//B)
        
        #### local metric ####
        q_list = []
        s_feat = s_local.repeat(B,1,1,1)
        for i in range(B):
            q_list.append(q_local[i:i+1].repeat(B_s, 1, 1, 1))
        q_feat = torch.cat(q_list, dim=0)
        
        feat = self.norm_2(torch.cat([q_feat, s_feat], dim=0))
        q_feat = feat[:q_feat.shape[0]]
        s_feat = feat[-s_feat.shape[0]:]
        
        feat_pairs = torch.cat((q_feat, s_feat), dim=1).contiguous()
        output_local = self.relation_net2(feat_pairs)
        
        output_local_idt = output_local.contiguous().view(B, output_local.size(0)//B//support_num, support_num)
        
        output_local_reshaped = output_local.view(output_local.size(0)//support_num, support_num)
        output_local = output_local_reshaped.mean(dim=1)
        output_local = output_local.contiguous().view(B,output_local.size(0)//B)
        
        
        indices_1 = torch.argmax(output_global, dim=1)
        indices_2 = torch.argmax(output_local, dim=1)
        top_probs_1, _ = torch.topk(F.softmax(output_global, dim=1), 2, dim=1)
        top_probs_2, _ = torch.topk(F.softmax(output_local, dim=1), 2, dim=1)
        
        if self.training:
            mask = (indices_1 == indices_2) & (top_probs_1[:, 0] - top_probs_1[:, 1] > 0.98) & (top_probs_2[:, 0] - top_probs_2[:, 1] > 0.98)
        else:
            mask = (indices_1 == indices_2) & (top_probs_1[:, 0] - top_probs_1[:, 1] > self.thres) & (top_probs_2[:, 0] - top_probs_2[:, 1] > self.thres)
    
        output_all = torch.zeros_like(output_global)
        output_all[mask] = output_global[mask] + output_local[mask]
        
        return (output_local, output_local_idt, output_global, output_global_idt, output_all, mask)
        
class VSSM(nn.Module):
    def __init__(self,
                 num_classes = 9,
                 patch_size = [15,15],
                 thres = 0.5,
                 PCA_num = 10,
                 depths=[1,1], 
                 dims=[64,64],
                 mlp_ratio=1,
                 d_state=8,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = [patch_size, patch_size]
        self.thres = thres
        self.PCA_num = PCA_num
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.k2 = 9

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.HSIMixCon = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                patch_size = self.patch_size[i_layer],
                k2 = self.k2,
                i_layer = i_layer,
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                mlp_ratio=mlp_ratio,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer
            )
            self.HSIMixCon.append(layer)
        
        self.apply(self._init_weights)
        
        self.initial_embedding = Initial_embedding(self.PCA_num, self.dims[0])
        self.imagetoclass = imagetoclass(self.num_classes, self.dims[0])
        self.imagetoclass2 = imagetoclass2(self.dims[0])
        self.k = 1
        self.expand = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=self.dims[0], kernel_size=1), nn.BatchNorm1d(self.dims[0]), nn.ReLU(inplace=True))
        self.relation_net3 = RelationNetwork(input_size=[self.dims[0], self.k2])
        self.DASR = DASR(dims = self.dims, patch_size = self.patch_size[0], num_features = self.num_features, thres = self.thres)
        
        self.sum_3 = relation_weight_3()
        self.sum_4 = relation_weight_4()
        self.down = DownEncoder(self.dims[0], self.dims[0])

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def Initial_prediction(self, q_feat, s_feat):
        q_local = self.initial_embedding(q_feat)
        s_local = self.initial_embedding(s_feat)
        
        q_feat = rearrange(q_local, 'b c h w -> b h w c')
        s_feat = rearrange(s_local, 'b c h w -> b h w c')
        q_feat = self.pos_drop(q_feat)
        s_feat = self.pos_drop(s_feat)

        q_global, s_global, loss_MoE1 = self.HSIMixCon[0](q_feat, s_feat)
        
        q_global = rearrange(q_global, 'b h w c -> b c h w')
        s_global = rearrange(s_global, 'b h w c -> b c h w')
        
        return (q_local, s_local, q_global, s_global, loss_MoE1)
    
    def Refine_embedding(self, q_local, s_local, q_global, s_global, mask):
        B, _, _, _ = q_local.size()
        if self.training:
            q_global_down = self.down(q_global)
            s_global_down = self.down(s_global)
            q_local_down = self.down(q_local)
            L1 = self.imagetoclass(s_global_down, q_global_down, [], [], 'test', k=self.k, k2=self.k2)
            L1 = L1.view(-1, self.k2).unsqueeze(1)

            L3 = self.imagetoclass2(q_global_down, s_global_down, [], [], 'test', k=self.k, k2=self.k2)
            L3 = L3.view(-1, self.k2).unsqueeze(1)
            
            L2 = self.imagetoclass(s_local, q_local_down, [], [], 'test', k=self.k, k2=self.k2)
            L2 = L2.view(-1, self.k2).unsqueeze(1)

            L4 = self.imagetoclass2(q_local_down, s_local, [], [], 'test', k=self.k, k2=self.k2)
            L4 = L4.view(-1, self.k2).unsqueeze(1)
            
            L = torch.cat([L1, L2, L3, L4], dim=0)
            L = self.expand(L)
            
            L = rearrange(L, 'b c h -> b h c')
            L = self.pos_drop(L)
        else:
            q_global_down = self.down(q_global[~mask])
            s_global_down = self.down(s_global)
            q_local = self.down(q_local)
            s_local = self.down(s_local)
            B_new, _, _, _ = q_global_down.shape
            L1 = self.imagetoclass(s_global_down, q_global_down, [], [], 'test', k=self.k, k2=self.k2)
            L1 = L1.view(-1, self.k2).unsqueeze(1)

            L3 = self.imagetoclass2(q_global_down, s_global_down, [], [], 'test', k=self.k, k2=self.k2)
            L3 = L3.view(-1, self.k2).unsqueeze(1)
            
            L2 = self.imagetoclass(s_local, q_local[~mask], [], [], 'test', k=self.k, k2=self.k2)
            L2 = L2.view(-1, self.k2).unsqueeze(1)

            L4 = self.imagetoclass2(q_local[~mask], s_local, [], [], 'test', k=self.k, k2=self.k2)
            L4 = L4.view(-1, self.k2).unsqueeze(1)
    
            L = torch.cat([L1, L2, L3, L4], dim=0)
            L = self.expand(L)
            
            L = rearrange(L, 'b c h -> b h c')
            L = self.pos_drop(L)
            
            B = B_new
            
        return (L, L1, L2, L3, L4), B
    
    def Adaptive_refinement(self, q_local, s_local, q_global, s_global, output_local, output_local_idt, output_global, output_global_idt, output_all, mask, support_num):
        (L, L1, L2, L3, L4), B = self.Refine_embedding(q_local, s_local, q_global, s_global, mask)
        if self.training:
            q_feat, loss_MoE2 = self.HSIMixCon[1](L, None)

            q_feat = rearrange(q_feat, 'b h c -> b c h')
            output = self.relation_net3(q_feat.contiguous())
            
            output_3 = output[:L1.size(0),:].view(self.num_classes, B, self.k).mean(dim=2).permute(1,0).contiguous()
            output_3_idt = output[:L1.size(0),:].view(self.num_classes, B, self.k).permute(1,0,2).contiguous()
            output_3_2 = output[L1.size(0):L1.size(0)+L2.size(0),:].view(self.num_classes, B, self.k).mean(dim=2).permute(1,0).contiguous()
            output_3_2_idt = output[L1.size(0):L1.size(0)+L2.size(0),:].view(self.num_classes, B, self.k).permute(1,0,2).contiguous()
            output_4 = output[-L4.size(0)-L3.size(0):-L4.size(0),:].view(B, self.num_classes, support_num, self.k).mean(dim=3).mean(dim=2)
            output_4_idt = output[-L4.size(0)-L3.size(0):-L4.size(0),:].view(B, self.num_classes, support_num * self.k)
            output_4_2 = output[-L4.size(0):,:].view(B, self.num_classes, support_num, self.k).mean(dim=3).mean(dim=2)
            output_4_2_idt = output[-L4.size(0):,:].view(B, self.num_classes, support_num * self.k)
            
            output_3 = self.sum_4(output_3, output_3_2, output_4, output_4_2)
            output_3_idt = torch.cat([output_3_idt, output_3_2_idt, output_4_idt, output_4_2_idt], dim=2)
            
            output_all[~mask, :] = self.sum_3(output_3[~mask, :], output_global[~mask, :], output_local[~mask, :],)
        else:
            B_new = B
            q_feat, loss_MoE2 = self.HSIMixCon[1](L, None)

            q_feat = rearrange(q_feat, 'b h c -> b c h')
            output = self.relation_net3(q_feat.contiguous())
            
            output_3 = output[:L1.size(0),:].view(self.num_classes, B_new, self.k).mean(dim=2).permute(1,0).contiguous()
            output_3_idt = output[:L1.size(0),:].view(self.num_classes, B_new, self.k).permute(1,0,2).contiguous()
            output_3_2 = output[L1.size(0):L1.size(0)+L2.size(0),:].view(self.num_classes, B_new, self.k).mean(dim=2).permute(1,0).contiguous()
            output_3_2_idt = output[L1.size(0):L1.size(0)+L2.size(0),:].view(self.num_classes, B_new, self.k).permute(1,0,2).contiguous()
            output_4 = output[-L4.size(0)-L3.size(0):-L4.size(0),:].view(B_new, self.num_classes, support_num, self.k).mean(dim=3).mean(dim=2)
            output_4_idt = output[-L4.size(0)-L3.size(0):-L4.size(0),:].view(B_new, self.num_classes, support_num * self.k)
            output_4_2 = output[-L4.size(0):,:].view(B_new, self.num_classes, support_num, self.k).mean(dim=3).mean(dim=2)
            output_4_2_idt = output[-L4.size(0):,:].view(B_new, self.num_classes, support_num * self.k)
            
            output_3 = self.sum_4(output_3, output_3_2, output_4, output_4_2)
            
            output_all[~mask, :] = self.sum_3(output_3, output_global[~mask, :], output_local[~mask, :])
    
        return (output_all, output_global_idt, output_local_idt, output_3_idt), loss_MoE2
    
    def forward(self, q_feat, s_feat, support_num):
        
        (q_local, s_local, q_global, s_global, loss_MoE1) = self.Initial_prediction(q_feat, s_feat)
        
        (output_local, output_local_idt, output_global, output_global_idt, output_all, mask) = self.DASR(q_local, s_local, q_global, s_global, support_num)
        
        if not self.training and (~mask).sum() == 0:
            return (output_all, output_global_idt, output_local_idt, None), None
        
        output, loss_MoE2 = self.Adaptive_refinement(q_local, s_local, q_global, s_global, output_local, output_local_idt, output_global, output_global_idt, output_all, mask, support_num)
        
        loss_MoE = loss_MoE1 + loss_MoE2

        return output, loss_MoE