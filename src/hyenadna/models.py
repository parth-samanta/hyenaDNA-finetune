import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import HyenaDNAModel
from .data import reverse_complement


TINY = dict(d_model=256, n_layer=2, d_inner=1024, vocab_size=12,
    pad_vocab_size_multiple=8, residual_in_fp32=True,
    layer=dict(emb_dim=5, filter_order=64, l_max=1026, modulate=True, w=10,
               lr=1e-4, wd=0.0, lr_pos_emb=0.0))


def canonical(key):
    for prefix in ("module.", "model.", "encoder."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key.replace(".mixer.layer.", ".mixer.").replace(".mlp.layer.", ".mlp.")


def load_backbone(model, path):
    # Lightning checkpoints contain metadata in addition to tensors. Only load trusted files.
    raw = torch.load(path, map_location="cpu", weights_only=False)
    raw = raw.get("state_dict", raw)
    source = {}
    for key, value in raw.items():
        name = canonical(key)
        if name.startswith("backbone."):
            if name in source:
                raise ValueError("Duplicate normalized checkpoint key: " + name)
            source[name] = value
    target = model.state_dict()
    if set(target) != set(source):
        raise ValueError("Backbone mismatch. Missing: %s; unexpected: %s" %
                         (sorted(set(target) - set(source)), sorted(set(source) - set(target))))
    for name in target:
        if target[name].shape != source[name].shape:
            raise ValueError("Shape mismatch at %s: model %s, checkpoint %s" %
                             (name, target[name].shape, source[name].shape))
    model.load_state_dict(source, strict=True)


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


class Adapter(nn.Module):
    def __init__(self, width, bottleneck):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


class AdaptedMLP(nn.Module):
    def __init__(self, base, width, bottleneck):
        super().__init__()
        self.base, self.adapter = base, Adapter(width, bottleneck)

    def forward(self, x):
        return self.adapter(self.base(x))


class Readout(nn.Module):
    def __init__(self, width, cfg):
        super().__init__()
        self.kind, self.radius = cfg["head"], cfg["local_radius"]
        self.bins = cfg["regional_bins"]
        self.summaries = cfg["attention_summaries"]
        multiplier = {"mean_max_last": 3, "local_global": 2, "site_regions": 3,
                      "regional": self.bins, "regional_attention": self.bins,
                      "residual_regional_attention": self.bins,
                      "multi_attention": self.summaries}.get(self.kind, 1)
        self.output_size = width * multiplier
        if self.kind in {"attention", "multi_attention"}:
            self.score = nn.Linear(width, self.summaries if self.kind == "multi_attention" else 1)
        if self.kind in {"regional_attention", "residual_regional_attention"}:
            # A zero query gives regional means without consuming initialization RNG.
            self.regional_query = nn.Parameter(torch.zeros(width))
            if self.kind == "residual_regional_attention":
                self.regional_gate_logits = nn.Parameter(torch.zeros(self.bins))
        if self.kind == "cnn":
            k = cfg["cnn_kernel"]
            self.conv = nn.Conv1d(width, width, k, padding=k // 2)

    def forward(self, h, mask, site):
        if not mask.any(1).all():
            raise ValueError("Cannot aggregate an empty sequence")
        valid = mask.unsqueeze(-1)
        clean = h.masked_fill(~valid, 0.)
        if self.kind in {"regional_attention", "residual_regional_attention"}:
            return self.regional_attention(clean, mask)
        def average(region):
            return h.masked_fill(~region.unsqueeze(-1), 0.).sum(1) / region.sum(1, keepdim=True).clamp_min(1)
        mean = average(mask)
        if self.kind == "mean":
            return mean
        if self.kind in {"max", "mean_max_last"}:
            maximum = h.masked_fill(~valid, -torch.inf).max(1).values
            if self.kind == "max":
                return maximum
        if self.kind in {"last", "mean_max_last"}:
            positions = torch.arange(h.shape[1], device=h.device).expand_as(mask)
            last_index = positions.masked_fill(~mask, -1).max(1).values
            last = h[torch.arange(h.shape[0], device=h.device), last_index]
            return last if self.kind == "last" else torch.cat([mean, maximum, last], dim=-1)
        if self.kind in {"attention", "multi_attention"}:
            scores = self.score(clean).float().masked_fill(~valid, -torch.inf)
            weights = scores.softmax(1).to(h.dtype)
            return torch.einsum("blh,bld->bhd", weights, clean).flatten(1)
        if self.kind == "regional":
            # Bin real-token ranks, so adding padding never moves a nucleotide to another bin.
            rank = mask.long().cumsum(1) - 1
            bins = rank * self.bins // mask.sum(1, keepdim=True)
            return torch.cat([average(mask & (bins == i)) for i in range(self.bins)], dim=-1)
        if self.kind == "cnn":
            # Zero padding before convolution so trailing token states don't become features.
            z = F.gelu(self.conv(clean.transpose(1, 2))).transpose(1, 2)
            return z.masked_fill(~valid, -torch.inf).max(1).values
        if (site < 0).any() or (site >= mask.shape[1]).any():
            raise ValueError("Site-based pooling needs a valid nucleotide index for every sample")
        if not mask.gather(1, site[:, None]).all():
            raise ValueError("A site index points to padding")
        positions = torch.arange(h.shape[1], device=h.device)[None, :]
        local = mask & ((positions - site[:, None]).abs() <= self.radius)
        pooled = average(local)
        if self.kind == "local_global":
            return torch.cat([mean, pooled], dim=-1)
        upstream = mask & (positions < site[:, None] - self.radius)
        downstream = mask & (positions > site[:, None] + self.radius)
        return torch.cat([average(upstream), pooled, average(downstream)], dim=-1)

    def regional_attention(self, clean, mask):
        rank = mask.long().cumsum(1) - 1
        bins = rank * self.bins // mask.sum(1, keepdim=True)
        region_ids = torch.arange(self.bins, device=clean.device)[None, :, None]
        regions = mask[:, None, :] & (bins[:, None, :] == region_ids)
        count = regions.sum(-1, keepdim=True)
        scores = F.linear(clean, self.regional_query[None, :]).squeeze(-1).float()
        logits = scores[:, None, :].expand(-1, self.bins, -1).masked_fill(~regions, -torch.inf)
        # Very short sequences can leave bins empty; avoid an all-inf softmax.
        logits = torch.where(count > 0, logits, torch.zeros_like(logits))
        weights = logits.softmax(-1).masked_fill(~regions, 0.)
        if self.kind == "residual_regional_attention":
            uniform = regions.float() / count.clamp_min(1)
            gates = self.regional_gate_logits.sigmoid()[None, :, None]
            weights = uniform + gates * (weights - uniform)
        return torch.bmm(weights.to(clean.dtype), clean).flatten(1)


class Model(nn.Module):
    def __init__(self, cfg, initialize=True, resolved=None):
        super().__init__()
        self.cfg = cfg
        if resolved is not None:
            arch = copy.deepcopy(resolved)
        else:
            arch = copy.deepcopy(TINY)
            if cfg["model_config"]:
                path = Path(cfg["model_config"])
                if not path.exists() and cfg["pretrained"]:
                    raise FileNotFoundError("Download the checkpoint config first: " + str(path))
                if path.exists():
                    arch = json.loads(path.read_text())
            arch.update(copy.deepcopy(cfg["model_overrides"]))
        arch.pop("use_head", None)
        arch.pop("n_classes", None)
        arch["layer"].pop("_name_", None)
        arch["layer"]["lr"] = cfg["backbone_lr"]
        arch["embed_dropout"], arch["resid_dropout"] = cfg["embed_dropout"], cfg["resid_dropout"]
        if cfg["max_length"] > arch["layer"]["l_max"]:
            raise ValueError("max_length exceeds the checkpoint context; use a longer-context checkpoint/config")
        self.architecture = copy.deepcopy(arch)
        width = arch["d_model"]
        if cfg["model"] == "gpt":
            arch["attn_layer_idx"] = list(range(arch["n_layer"]))
            arch["attn_cfg"] = dict(embed_dim=width, num_heads=max(1, width // 64), causal=True)
            arch["max_position_embeddings"] = cfg["max_length"]
        if cfg["model"] == "cnn":
            self.encoder = nn.Sequential(nn.Embedding(12, width, padding_idx=4))
            self.cnn = nn.Sequential(nn.Conv1d(width, width, 15, padding=7), nn.GELU(),
                                     nn.Conv1d(width, width, 7, padding=3), nn.GELU())
        else:
            self.encoder = HyenaDNAModel(**arch, use_head=False)
            if cfg["pretrained"] and initialize:
                load_backbone(self.encoder, cfg["checkpoint"])
        self.readout = Readout(width, cfg)
        size = self.readout.output_size * (2 if cfg["bidirectional"] else 1)
        # Every aggregation feeds the same width into the classifier. This projection stays trainable.
        self.projection = nn.Sequential(nn.Linear(size, width), nn.GELU())
        self.heads = nn.ModuleList(nn.Sequential(nn.Dropout(cfg["head_dropout"]), nn.Linear(width, 2))
                                  for _ in cfg["tasks"])
        self.adaptation_modules = []
        self._adaptation_installed = False
        self.stage = "full"

    def install_adaptation(self):
        if self._adaptation_installed:
            return
        if self.cfg["method"] == "lora":
            targets = self.cfg["lora_targets"]
            for name, module in list(self.encoder.named_modules()):
                if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
                    parent_name, attr = name.rsplit(".", 1)
                    parent = self.encoder.get_submodule(parent_name)
                    setattr(parent, attr, LoRALinear(module, self.cfg["lora_rank"], self.cfg["lora_alpha"]))
                    self.adaptation_modules.append(name)
            if not self.adaptation_modules:
                raise ValueError("No LoRA modules matched; check lora_targets")
        elif self.cfg["method"] == "adapters":
            for i, block in enumerate(self.encoder.backbone.layers):
                block.mlp = AdaptedMLP(block.mlp, self.architecture["d_model"], self.cfg["adapter_dim"])
                self.adaptation_modules.append("backbone.layers.%d.mlp.adapter" % i)
        self._adaptation_installed = True

    def set_stage(self, epoch):
        method = self.cfg["method"]
        if method == "gradual":
            method = ("frozen" if epoch < self.cfg["head_epochs"] else
                      "partial" if epoch < self.cfg["head_epochs"] + self.cfg["partial_epochs"] else "full")
        self.encoder.requires_grad_(method == "full")
        if self.cfg["model"] == "cnn":
            self.cnn.requires_grad_(method == "full")
        elif method == "partial":
            self.encoder.backbone.layers[-1].requires_grad_(True)
            self.encoder.backbone.ln_f.requires_grad_(True)
        elif method in {"mixers", "mlps"}:
            for block in self.encoder.backbone.layers:
                getattr(block, "mixer" if method == "mixers" else "mlp").requires_grad_(True)
        elif method == "lora":
            for module in self.encoder.modules():
                if isinstance(module, LoRALinear):
                    module.a.requires_grad_(True)
                    module.b.requires_grad_(True)
        elif method == "adapters":
            for module in self.encoder.modules():
                if isinstance(module, Adapter):
                    module.requires_grad_(True)
        self.stage = method

    def train(self, mode=True):
        super().train(mode)
        if mode and self.stage != "full":
            self.encoder.eval()
            if self.cfg["model"] == "cnn":
                self.cnn.eval()
            elif self.stage == "partial":
                self.encoder.backbone.layers[-1].train()
            elif self.stage in {"mixers", "mlps"}:
                for block in self.encoder.backbone.layers:
                    getattr(block, "mixer" if self.stage == "mixers" else "mlp").train()
        return self

    def hidden(self, ids):
        if self.cfg["model"] == "cnn":
            return self.cnn(self.encoder(ids).transpose(1, 2)).transpose(1, 2)
        return self.encoder(ids)

    def features(self, batch):
        ids, mask, site = batch["ids"], batch["mask"], batch["site"]
        forward = self.readout(self.hidden(ids), mask, site)
        if not self.cfg["bidirectional"]:
            return forward
        rc = reverse_complement(ids, mask)
        rc_site = torch.where(site >= 0, mask.sum(1) - 1 - site, site)
        backward = self.readout(self.hidden(rc), mask, rc_site)
        return torch.cat([forward, backward], dim=-1)

    def forward(self, batch):
        features = batch["features"] if "features" in batch else self.features(batch)
        features = self.projection(features)
        logits = torch.stack([head(features) for head in self.heads], dim=1)
        return logits[torch.arange(len(features), device=features.device), batch["task"]]


def make_optimizer(model, cfg):
    groups = {}
    # Include currently frozen parameters so gradual unfreezing preserves optimizer state.
    for name, param in model.named_parameters():
        backbone = name.startswith(("encoder.", "cnn."))
        lr = cfg["backbone_lr"] if backbone else cfg["head_lr"]
        if cfg["method"] in {"lora", "adapters"} and (name.endswith((".a", ".b")) or ".adapter." in name):
            lr = cfg["head_lr"]
        wd = cfg["weight_decay"]
        if param.ndim < 2 or (cfg["model"] == "hyena" and ".mixer." in name):
            wd = 0.
        groups.setdefault((lr, wd), []).append(param)
    return torch.optim.AdamW([dict(params=params, lr=lr, weight_decay=wd)
                              for (lr, wd), params in groups.items()])

