# Adapted from HazyResearch/hyena-dna, commit d553021b483b82980aa4b868b37ec2d4332e198a.
# Apache-2.0; see LICENSE.hyena-dna and THIRD_PARTY.md.
# Tokenizer dependencies removed; stochastic depth implemented with plain PyTorch.
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import Optional
from functools import partial
from torch import Tensor
from collections import namedtuple
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

class StochasticDepth(nn.Module):
    def __init__(self, p, mode="row"):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)

def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    # FP32 FFT also works for non-power-of-two lengths under mixed precision.
    k_f = torch.fft.rfft(k.float(), n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.float(), n=fft_size)
    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)

@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)

class OptimModule(nn.Module):

    def register(self, name, tensor, lr=None, wd=0.0):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {}
            if lr is not None:
                optim['lr'] = lr
            if wd is not None:
                optim['weight_decay'] = wd
            setattr(getattr(self, name), '_optim', optim)

class Sin(nn.Module):

    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)

class PositionalEmbedding(OptimModule):

    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float=1e-05, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        f = torch.linspace(0.0001, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register('z', z, lr=lr_pos_emb)
        self.register('t', t, lr=0.0)

    def forward(self, L):
        return (self.z[:, :L], self.t[:, :L])

class ExponentialModulation(OptimModule):

    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5, target=0.01, modulation_lr=0.0, modulate: bool=True, shift: float=0.05, **kwargs):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register('deltas', deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x

class HyenaFilter(OptimModule):

    def __init__(self, d_model, emb_dim=3, order=16, fused_fft_conv=False, seq_len=1024, lr=0.001, lr_pos_emb=1e-05, dropout=0.0, w=1, wd=0, bias=True, num_inner_mlps=2, normalized=False, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, 'emb_dim must be odd and greater or equal to 3 (time, sine and cosine)'
        self.seq_len = seq_len
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)
        self.implicit_filter = nn.Sequential(nn.Linear(emb_dim, order), act)
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)
        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
        self.modulation = ExponentialModulation(d_model, **kwargs)
        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {'weight_decay': wd, 'lr': lr}
                setattr(getattr(c, name), '_optim', optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None:
            k = self.filter(L)
        k = k[0] if type(k) is tuple else k
        y = fftconv(x, k, bias)
        return y

class HyenaOperator(nn.Module):

    def __init__(self, d_model, l_max, order=2, filter_order=64, dropout=0.0, filter_dropout=0.0, **filter_args):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)
        self.short_filter = nn.Conv1d(inner_width, inner_width, 3, padding=2, groups=inner_width)
        self.filter_fn = HyenaFilter(d_model * (order - 1), order=filter_order, seq_len=l_max, channels=1, dropout=filter_dropout, **filter_args)

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')
        uc = self.short_filter(u)[..., :l_filter]
        *x, v = uc.split(self.d_model, dim=1)
        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])
        y = rearrange(v * x[0], 'b d l -> b l d')
        y = self.out_proj(y)
        return y

class SelfAttention(nn.Module):

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, causal=None, key_padding_mask=None):
        batch_size, seqlen = (qkv.shape[0], qkv.shape[1])
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + rearrange(padding_mask, 'b s -> b 1 1 s')
        if causal:
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
        return output

class MHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0, softmax_scale=None, causal=False, layer_idx=None, dwconv=False, return_residual=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.return_residual = return_residual
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, 'self.kdim must be divisible by num_heads'
        self.head_dim = self.embed_dim // num_heads
        linear_cls = nn.Linear
        linear_resid_cls = LinearResidual
        inner_attn_cls = SelfAttention
        if not self.return_residual:
            self.Wqkv = linear_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        else:
            self.Wqkv = linear_resid_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        if self.dwconv:
            self.dwconv_qkv = nn.Conv1d(3 * embed_dim, 3 * embed_dim, kernel_size=3, padding=2, groups=3 * embed_dim)
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.out_proj = linear_cls(embed_dim, embed_dim, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, **kwargs):
        kwargs = {'key_padding_mask': key_padding_mask, **kwargs}
        if not self.return_residual:
            qkv = self.Wqkv(x)
        else:
            qkv, x = self.Wqkv(x)
        if self.dwconv:
            qkv = rearrange(self.dwconv_qkv(rearrange(qkv, 'b s d -> b d s'))[..., :-2], 'b d s -> b s d').contiguous()
        qkv = rearrange(qkv, '... (three h d) -> ... three h d', three=3, d=self.head_dim)
        context = self.inner_attn(qkv, **kwargs)
        out = self.out_proj(rearrange(context, '... h d -> ... (h d)'))
        return out if not self.return_residual else (out, x)

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu, return_residual=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

class LinearResidual(nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (super().forward(input), input)

class Block(nn.Module):

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm, dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0.0, resid_dropout2=0.0, drop_path1=0.0, drop_path2=0.0, return_residual=False, residual_in_fp32=False):
        super().__init__()
        self.prenorm = prenorm
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls()
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual=None, mixer_subset=None, mixer_kwargs=None):
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = dropped + residual if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
                hidden_states = self.mlp(hidden_states)
            return (hidden_states, residual)
        else:
            assert residual is None
            mixer_out = self.mixer(hidden_states, **mixer_kwargs if mixer_kwargs is not None else {})
            if self.return_residual:
                mixer_out, hidden_states = mixer_out
            hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out)) + hidden_states).to(dtype=self.norm1.weight.dtype))
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:
                    mlp_out, hidden_states = mlp_out
                hidden_states = self.norm2((self.drop_path2(self.dropout2(mlp_out)) + hidden_states).to(dtype=self.norm2.weight.dtype))
            return hidden_states

def create_mixer_cls(layer=None, attn_layer_idx=None, attn_cfg=None, layer_idx=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        causal = True if attn_cfg is None else attn_cfg.pop('causal', True)
        mha_cls = MHA
        mixer_cls = partial(mha_cls, causal=causal, layer_idx=layer_idx, **attn_cfg if attn_cfg is not None else {}, **factory_kwargs)
    else:
        mixer_cls = partial(HyenaOperator, **layer)
    return mixer_cls

def create_mlp_cls(d_model, d_inner=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model
    mlp_cls = partial(Mlp, hidden_features=inner_dim, activation=partial(F.gelu, approximate='tanh'), **factory_kwargs)
    return mlp_cls

def create_block(d_model, d_inner=None, layer=None, attn_layer_idx=None, attn_cfg=None, layer_norm_epsilon=1e-05, resid_dropout1=0.0, resid_dropout2=0.0, residual_in_fp32=False, layer_idx=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    mixer_cls = create_mixer_cls(layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg, layer_idx=layer_idx, **factory_kwargs)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls, prenorm=True, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block

def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ['out_proj.weight', 'fc2.weight']:
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
            elif name in ['output_linear.0.weight']:
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    nn.init.normal_(p[:out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)

class GPT2Embeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None, word_embed_proj_dim=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim, padding_idx=padding_idx, **factory_kwargs)
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False, **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim, **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

class LMBackbone(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int, process_group=None, layer=None, attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0, resid_dropout: float=0.0, embed_dropout: float=0.1, layer_norm_epsilon: float=1e-05, initializer_cfg=None, residual_in_fp32=False, device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.process_group = process_group
        self.residual_in_fp32 = residual_in_fp32
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings, **factory_kwargs)
        self.layers = nn.ModuleList([create_block(d_model, d_inner=d_inner, layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg, layer_norm_epsilon=layer_norm_epsilon, resid_dropout1=embed_dropout if i == 0 else resid_dropout, resid_dropout2=resid_dropout, residual_in_fp32=residual_in_fp32, layer_idx=i, **factory_kwargs) for i in range(n_layer)])
        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)
        self.apply(partial(_init_weights, n_layer=n_layer, **initializer_cfg if initializer_cfg is not None else {}))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        return hidden_states

class SequenceDecoder(nn.Module):

    def __init__(self, d_model, d_output=None, l_output=None, use_lengths=False, mode='last'):
        super().__init__()
        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)
        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False
        self.use_lengths = use_lengths
        self.mode = mode
        if mode == 'ragged':
            assert not use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None):
        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)
            else:
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze
        if self.mode == 'last':
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == 'first':
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == 'pool':
            restrict = lambda x: (torch.cumsum(x, dim=-2) / torch.arange(1, 1 + x.size(-2), device=x.device, dtype=x.dtype).unsqueeze(-1))[..., -l_output:, :]

            def restrict(x):
                L = x.size(-2)
                s = x.sum(dim=-2, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x[..., -(l_output - 1):, :].flip(-2), dim=-2)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c
                    s = s.flip(-2)
                denom = torch.arange(L - l_output + 1, L + 1, dtype=x.dtype, device=x.device)
                s = s / denom
                return s
        elif self.mode == 'sum':
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
        elif self.mode == 'ragged':
            assert lengths is not None, 'lengths must be provided for ragged mode'
            restrict = lambda x: x[..., :max(lengths), :]
        else:
            raise NotImplementedError("Mode must be ['last' | 'first' | 'pool' | 'sum']")
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack([restrict(out[..., :length, :]) for out, length in zip(torch.unbind(x, dim=0), lengths)], dim=0)
        else:
            x = restrict(x)
        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)
        x = self.output_transform(x)
        return x

    def step(self, x, state=None):
        return self.output_transform(x)

class HyenaDNAModel(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int, layer=None, attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0, resid_dropout: float=0.0, embed_dropout: float=0.1, layer_norm_epsilon: float=1e-05, initializer_cfg=None, residual_in_fp32=False, pad_vocab_size_multiple: int=1, use_head=False, n_classes: int=2, device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - vocab_size % pad_vocab_size_multiple
        self.use_head = use_head
        if 'd_model' not in layer:
            layer['d_model'] = d_model
        self.backbone = LMBackbone(d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size, layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg, max_position_embeddings=max_position_embeddings, resid_dropout=resid_dropout, embed_dropout=embed_dropout, layer_norm_epsilon=layer_norm_epsilon, initializer_cfg=initializer_cfg, residual_in_fp32=residual_in_fp32, **factory_kwargs, **kwargs)
        if self.use_head:
            self.head = SequenceDecoder(d_model=d_model, d_output=n_classes, l_output=0, mode='pool')
        self.apply(partial(_init_weights, n_layer=n_layer, **initializer_cfg if initializer_cfg is not None else {}))

    def forward(self, input_ids, position_ids=None, state=None):
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        if self.use_head:
            return self.head(hidden_states)
        else:
            return hidden_states

