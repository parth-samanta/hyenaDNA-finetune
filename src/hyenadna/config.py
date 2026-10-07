import copy
import json
from pathlib import Path

DEFAULTS = dict(
    name="baseline", data="data/prepared", output="runs/baseline", tasks=["promoter_all"],
    checkpoint="checkpoints/tiny/weights.ckpt", model_config="checkpoints/tiny/config.json",
    model="hyena", pretrained=True, method="frozen", head="mean", bidirectional=False,
    max_length=300, local_radius=20, cnn_kernel=7, lora_rank=8, lora_alpha=16,
    lora_targets=["in_proj", "out_proj"], adapter_dim=32, seed=0, subset_seed=101,
    label_fraction=1.0, epochs=100, patience=15, batch_size=256, micro_batch=8,
    backbone_lr=0.0001, head_lr=0.001, weight_decay=0.1, embed_dropout=0.1,
    resid_dropout=0.0, head_dropout=0.1, warmup_fraction=0.01, min_lr_ratio=0.1,
    head_epochs=2, partial_epochs=3, loss="ce", focal_gamma=2.0,
    label_smoothing=0.0, rc_augmentation=False, rc_labels_verified=False,
    tapt_epochs=0, tapt_lr=0.00001, device="auto", precision="fp32",
    workers=0, cache_features=False, model_overrides={}, max_steps_per_epoch=None,
    attention_summaries=4, regional_bins=4,
)


def load_config(path_or_dict):
    cfg = copy.deepcopy(DEFAULTS)
    supplied = (json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
                if not isinstance(path_or_dict, dict) else path_or_dict)
    unknown = set(supplied) - set(cfg)
    if unknown:
        raise ValueError("Unknown settings: " + str(sorted(unknown)))
    cfg.update(copy.deepcopy(supplied))
    if cfg["method"] not in {"full", "frozen", "partial", "gradual", "lora", "mixers", "mlps", "adapters"}:
        raise ValueError("Unknown training method")
    if cfg["model"] not in {"hyena", "gpt", "cnn"}:
        raise ValueError("Unknown model")
    if cfg["head"] not in {"mean", "max", "last", "mean_max_last", "attention", "multi_attention", "regional", "regional_attention", "residual_regional_attention", "cnn", "local_global", "site_regions"}:
        raise ValueError("Unknown head")
    if cfg["model"] != "hyena" and (cfg["pretrained"] or cfg["tapt_epochs"]):
        raise ValueError("GPT and CNN are scratch baselines; set pretrained=false and tapt_epochs=0")
    if cfg["model"] == "cnn" and cfg["method"] not in {"full", "frozen"}:
        raise ValueError("CNN supports full or frozen training")
    if cfg["rc_augmentation"] and not cfg["rc_labels_verified"]:
        raise ValueError("Verify label semantics before setting rc_labels_verified=true")
    if cfg["cache_features"] and (cfg["method"] != "frozen" or cfg["head"] not in {"mean", "max", "last", "mean_max_last", "regional", "local_global", "site_regions"} or cfg["rc_augmentation"]):
        raise ValueError("Caching needs a frozen backbone, a nonlearned aggregation and no RC augmentation")
    for key in ("epochs", "patience", "batch_size", "micro_batch", "max_length", "lora_rank", "adapter_dim", "attention_summaries", "regional_bins"):
        if cfg[key] < 1:
            raise ValueError(key + " must be positive")
    if cfg["batch_size"] % cfg["micro_batch"]:
        raise ValueError("micro_batch must divide batch_size")
    if not 0 < cfg["label_fraction"] <= 1:
        raise ValueError("label_fraction must be in (0,1]")
    if len(set(cfg["tasks"])) != len(cfg["tasks"]) or not cfg["tasks"]:
        raise ValueError("tasks must be nonempty and unique")
    if cfg["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16 or bf16")
    if cfg["loss"] not in {"ce", "weighted_ce", "focal"}:
        raise ValueError("Unknown loss")
    if cfg["cnn_kernel"] % 2 != 1:
        raise ValueError("cnn_kernel must be odd")
    for key in ("head_epochs", "partial_epochs", "tapt_epochs", "local_radius"):
        if cfg[key] < 0:
            raise ValueError(key + " cannot be negative")
    for key in ("embed_dropout", "resid_dropout", "head_dropout", "label_smoothing", "warmup_fraction"):
        if not 0 <= cfg[key] < 1:
            raise ValueError(key + " must be in [0,1)")
    if cfg["max_steps_per_epoch"] is not None and cfg["max_steps_per_epoch"] < 1:
        raise ValueError("max_steps_per_epoch must be positive")
    if min(cfg["backbone_lr"], cfg["head_lr"], cfg["tapt_lr"]) <= 0:
        raise ValueError("Learning rates must be positive")
    if cfg["weight_decay"] < 0 or not 0 <= cfg["min_lr_ratio"] <= 1:
        raise ValueError("Invalid weight decay or minimum LR ratio")
    return cfg

