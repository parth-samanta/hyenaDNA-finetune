import csv
import json
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import load_config
from .data import SequenceDataset, load_split, sha, subsample, write_json
from .metrics import by_task, slices
from .models import Model, make_optimizer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(cfg):
    choice = cfg["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if choice == "auto" else torch.device(choice)
    if cfg["precision"] != "fp32" and device.type != "cuda":
        raise ValueError("Use fp32 for CPU runs")
    if cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16")
    return device


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(cfg, device):
    return torch.autocast(device_type=device.type, enabled=cfg["precision"] != "fp32",
                          dtype=torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16)


def loader(rows, cfg, train=False, dataset=None):
    dataset = dataset if dataset is not None else SequenceDataset(
        rows, cfg["tasks"], cfg["max_length"], cfg["rc_augmentation"] and train)
    sampler = None
    if train and len(cfg["tasks"]) > 1:
        counts = Counter(r["task"] for r in rows)
        sampler = WeightedRandomSampler([1 / counts[r["task"]] for r in rows], len(rows), replacement=True)
    return DataLoader(dataset, batch_size=cfg["micro_batch"], shuffle=train and sampler is None,
                      sampler=sampler, num_workers=cfg["workers"], pin_memory=False)


class Cached(Dataset):
    def __init__(self, batches):
        self.data = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
    def __len__(self):
        return len(self.data["label"])
    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}


def cache(model, rows, cfg, device):
    model.eval()
    batches = []
    with torch.no_grad():
        for b in loader(rows, cfg):
            features = model.features(move(b, device)).float().cpu()
            batches.append(dict(features=features, label=b["label"], task=b["task"], index=b["index"]))
    return Cached(batches)


def evaluate(model, batches, rows, cfg, device):
    model.eval()
    probabilities = np.zeros(len(rows), dtype=np.float64)
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            with autocast(cfg, device):
                logits = model(move(batch, device))
            probabilities[batch["index"].numpy()] = logits.float().softmax(-1)[:, 1].cpu().numpy()
    synchronize(device)
    return by_task(rows, probabilities), probabilities, time.perf_counter() - start


def class_weights(rows, tasks, device):
    weights = []
    for task in tasks:
        counts = Counter(r["label"] for r in rows if r["task"] == task)
        if min(counts.get(0, 0), counts.get(1, 0)) == 0:
            raise ValueError("Training needs both classes for " + task)
        weights.append([sum(counts.values()) / (2 * counts[y]) for y in (0, 1)])
    return torch.tensor(weights, device=device)


def classification_loss(logits, b, cfg, weights):
    ce = F.cross_entropy(logits.float(), b["label"], reduction="none", label_smoothing=cfg["label_smoothing"])
    if cfg["loss"] == "weighted_ce":
        ce = ce * weights[b["task"], b["label"]]
    if cfg["loss"] == "focal":
        p = logits.float().softmax(-1).gather(1, b["label"][:, None]).squeeze(1)
        ce = ce * (1 - p).pow(cfg["focal_gamma"])
    return ce.sum()


def step_optimizer(model, opt, scaler, count):
    scaler.unscale_(opt)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(count)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not scaler.is_enabled() and not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient norm")
    old_scale = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return scaler.get_scale() >= old_scale


def tapt(model, rows, cfg, device, out):
    if not cfg["tapt_epochs"]:
        return
    model.encoder.requires_grad_(True)
    opt = torch.optim.AdamW(model.encoder.parameters(), lr=cfg["tapt_lr"], weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16")
    batches = loader(rows, dict(cfg, rc_augmentation=False), train=True)
    accumulation = cfg["batch_size"] // cfg["micro_batch"]
    for epoch in range(cfg["tapt_epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        tokens, total_loss, total_tokens = 0, 0., 0
        for i, batch in enumerate(batches):
            b = move(batch, device)
            targets = b["ids"][:, 1:].clone()
            valid = b["mask"][:, 1:] & b["mask"][:, :-1]
            targets[~valid] = -100
            n = int(valid.sum())
            if n == 0:
                raise ValueError("TAPT requires sequences with at least two valid bases")
            with autocast(cfg, device):
                hidden = model.hidden(b["ids"][:, :-1])
                logits = F.linear(hidden, model.encoder.backbone.embeddings.word_embeddings.weight)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                       ignore_index=-100, reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite TAPT loss")
            scaler.scale(loss).backward()
            tokens += n
            total_tokens += n
            total_loss += loss.item()
            if (i + 1) % accumulation == 0 or i + 1 == len(batches):
                step_optimizer(model, opt, scaler, tokens)
                tokens = 0
        record = dict(stage="tapt", epoch=epoch + 1, loss=total_loss / total_tokens)
        with open(out / "tapt_history.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(record, flush=True)
    torch.save(dict(state_dict=model.encoder.state_dict()), out / "adapted_backbone.pt")


def fit(config):
    cfg = load_config(config)
    out = Path(cfg["output"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty run directory: " + str(out))
    train_full = load_split(cfg["data"], "train", cfg["tasks"])
    train = subsample(train_full, cfg["label_fraction"], cfg["subset_seed"])
    val = load_split(cfg["data"], "val", cfg["tasks"])
    if cfg["head"] in {"local_global", "site_regions"} and any(r["site"] < 0 for r in train + val):
        raise ValueError("Missing verified site indices for the selected head")
    # Check lengths before allocating the model or writing a run directory.
    SequenceDataset(train + val, cfg["tasks"], cfg["max_length"])
    seed_all(cfg["seed"])
    device = device_for(cfg)
    model = Model(cfg).to(device)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", cfg)
    write_json(out / "architecture.json", model.architecture)
    write_json(out / "provenance.json", dict(data_manifest_sha256=sha(Path(cfg["data"]) / "manifest.json"),
        checkpoint_sha256=sha(cfg["checkpoint"]) if cfg["pretrained"] else None,
        python=platform.python_version(), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        source_hashes={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        train_ids=[(r["task"], r["id"]) for r in train],
        tapt_uses="full training partition, including unlabeled examples outside the label subset"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    tapt(model, train_full, cfg, device, out)
    model.install_adaptation()
    model.to(device)
    model.set_stage(0)
    opt = make_optimizer(model, cfg)
    train_cache = cache(model, train, cfg, device) if cfg["cache_features"] else None
    val_cache = cache(model, val, cfg, device) if cfg["cache_features"] else None
    train_loader = loader(train, cfg, train=True, dataset=train_cache)
    val_loader = loader(val, cfg, dataset=val_cache)
    accumulation = cfg["batch_size"] // cfg["micro_batch"]
    batch_limit = len(train_loader)
    if cfg["max_steps_per_epoch"] is not None:
        batch_limit = min(batch_limit, cfg["max_steps_per_epoch"] * accumulation)
    total_steps = math.ceil(batch_limit / accumulation) * cfg["epochs"]
    warmup = max(1, round(cfg["warmup_fraction"] * total_steps))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(1, total_steps - warmup))
        return cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16")
    weights = class_weights(train, cfg["tasks"], device)
    best, stale, previous_stage = -1., 0, None
    total_examples, updates = 0, 0
    for epoch in range(cfg["epochs"]):
        model.set_stage(epoch)
        if model.stage != previous_stage:
            stale = 0
            previous_stage = model.stage
        model.train()
        opt.zero_grad(set_to_none=True)
        count, seen, loss_sum = 0, 0, 0.
        for i, batch in enumerate(train_loader):
            if i >= batch_limit:
                break
            b = move(batch, device)
            with autocast(cfg, device):
                loss = classification_loss(model(b), b, cfg, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite classification loss")
            scaler.scale(loss).backward()
            n = len(b["label"])
            count += n
            seen += n
            loss_sum += loss.item()
            if (i + 1) % accumulation == 0 or i + 1 == batch_limit:
                if step_optimizer(model, opt, scaler, count):
                    scheduler.step()
                    updates += 1
                count = 0
        total_examples += seen
        result, _, inference_seconds = evaluate(model, val_loader, val, cfg, device)
        metric = float(np.mean([s["f1_macro"] for s in result.values()]))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        record = dict(epoch=epoch + 1, stage=model.stage, loss=loss_sum / seen,
                      validation=result, selection_score=metric, trainable_parameters=trainable,
                      examples_seen=total_examples, optimizer_updates=updates,
                      validation_seconds=inference_seconds)
        with open(out / "history.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if metric > best:
            best, stale = metric, 0
            torch.save(dict(model=model.state_dict(), epoch=epoch + 1), out / "best.pt")
            write_json(out / "validation.json", record)
        else:
            stale += 1
        if stale >= cfg["patience"] and not (cfg["method"] == "gradual" and model.stage != "full"):
            break
    synchronize(device)
    write_json(out / "complete.json", dict(best_validation=best,
        elapsed_seconds=time.perf_counter() - started, examples_seen=total_examples, optimizer_updates=updates,
        parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=trainable,
        adaptation_modules=model.adaptation_modules,
        aggregation_parameters=sum(p.numel() for p in model.readout.parameters()),
        projection_parameters=sum(p.numel() for p in model.projection.parameters()),
        classifier_parameters=sum(p.numel() for p in model.heads.parameters()),
        peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None))
    return out


def test(run_dir, device=None):
    out = Path(run_dir)
    if not (out / "complete.json").exists():
        raise ValueError("Run is incomplete")
    if (out / "test.json").exists():
        raise FileExistsError("Test results already exist; keep the original evaluation")
    cfg = load_config(out / "config.json")
    if device:
        cfg["device"] = device
    provenance = json.loads((out / "provenance.json").read_text())
    if sha(Path(cfg["data"]) / "manifest.json") != provenance["data_manifest_sha256"]:
        raise ValueError("Data manifest changed since fitting")
    rows = load_split(cfg["data"], "test", cfg["tasks"])
    seed_all(cfg["seed"])
    dev = device_for(cfg)
    model = Model(cfg, initialize=False, resolved=json.loads((out / "architecture.json").read_text()))
    model.install_adaptation()
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=True)["model"])
    model.to(dev)
    result, probabilities, seconds = evaluate(model, loader(rows, cfg), rows, cfg, dev)
    write_json(out / "test.json", dict(metrics=result, inference_seconds=seconds,
                                       samples_per_second=len(rows) / seconds))
    write_json(out / "test_slices.json", slices(rows, probabilities))
    with open(out / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "id", "label", "probability_positive"])
        writer.writerows((r["task"], r["id"], r["label"], float(p)) for r, p in zip(rows, probabilities))
    print(json.dumps(result, indent=2))
    return result

