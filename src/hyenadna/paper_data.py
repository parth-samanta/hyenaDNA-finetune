"""Load the original HyenaDNA acceptor FASTA data and preserve its partitions."""
from collections import Counter
from pathlib import Path
import json

from .data import VOCAB, load_split, write_json, write_rows, sha

PAPER_IMAGE = "hyenadna/hyena-dna-nt6"
PAPER_REVISION = "sha256:b8181cbe2d0dc9219d6a1204251b351e654f56593733bd58e17a68eba235a438"
PAPER_SPLIT_PROTOCOL = "paper_train_test_val_equals_test"
PAPER_HASHES = {
    "train": "a146a8186528982b5b6f717eba0512e21fea936982a2fc14b41a1399c57a6e17",
    "test": "624105498ae677b06dd3257fb66dcac246e7bc0762bf3e48ff467f49ffc1dc32",
}

def _paper_files(raw):
    """Retrieve just the pinned application layer; Docker is not required."""
    import tarfile
    import urllib.parse
    import urllib.request

    paths = {split: raw / (split + ".fasta") for split in PAPER_HASHES}
    if all(path.is_file() and sha(path) == PAPER_HASHES[split] for split, path in paths.items()):
        return paths
    raw.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"service": "registry.docker.io", "scope": f"repository:{PAPER_IMAGE}:pull"})
    with urllib.request.urlopen("https://auth.docker.io/token?" + query, timeout=60) as response:
        token = json.load(response)["token"]
    url = f"https://registry-1.docker.io/v2/{PAPER_IMAGE}/blobs/{PAPER_REVISION}"
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    archive = raw / "application_layer.tar.gz"
    if not archive.is_file() or "sha256:" + sha(archive) != PAPER_REVISION:
        print("Downloading original paper data (153 MiB application layer)...", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as stream:
            total = 0
            while block := response.read(1024 * 1024):
                stream.write(block)
                total += len(block)
                if total % (32 * 1024 * 1024) == 0:
                    print("Downloaded %d MiB..." % (total // (1024 * 1024)), flush=True)
        if "sha256:" + sha(archive) != PAPER_REVISION:
            raise ValueError("Original paper application layer checksum mismatch")
    wanted = {f"wdr/data/nucleotide_transformer/splice_sites_acceptor/splice_sites_acceptor_{split}.fasta": split
              for split in paths}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.name in wanted:
                if not member.isfile() or member.size > 100 * 1024 * 1024:
                    raise ValueError("Unexpected original dataset archive entry")
                split = wanted[member.name]
                with tar.extractfile(member) as source:
                    content = source.read()
                import hashlib
                if hashlib.sha256(content).hexdigest() != PAPER_HASHES[split]:
                    raise ValueError("Original FASTA checksum mismatch: " + split)
                paths[split].write_bytes(content)
    if not all(path.is_file() and sha(path) == PAPER_HASHES[split] for split, path in paths.items()):
        raise ValueError("Original image is missing the expected acceptor files")
    return paths


def _paper_records(path, split):
    rows, header, chunks = [], None, []
    def append():
        if header is None:
            return
        sequence = "".join(chunks).upper()
        if not header or header[-1] not in "01" or len(sequence) != 600 or set(sequence) - set(VOCAB):
            raise ValueError("Invalid original acceptor FASTA record: " + str(header))
        rows.append(dict(id=f"{split}:{header}", sequence=sequence, label=int(header[-1]),
                         task="splice_sites_acceptor", split=split, group="", site=-1,
                         subtype="", species="", chrom=""))
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith(">"):
                append()
                header, chunks = line[1:].rstrip(), []
            elif line:
                if header is None:
                    raise ValueError("FASTA sequence appears before its header")
                chunks.append(line)
    append()
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Original FASTA is empty or has duplicate headers")
    return rows


def prepare_paper(tasks, output, revision=PAPER_REVISION, raw_dir=None):
    """Match the official loader: all paper training; val aliases paper test."""
    tasks = list(tasks)
    if tasks != ["splice_sites_acceptor"] or revision != PAPER_REVISION:
        raise ValueError("Original paper data requires splice_sites_acceptor and the pinned image revision")
    output = Path(output)
    source_path = output / "source.json"
    if (output / "manifest.json").exists():
        source = json.loads(source_path.read_text())
        if (source.get("repository") != PAPER_IMAGE or source.get("revision") != revision or
                source.get("tasks") != tasks or source.get("raw_sha256") != PAPER_HASHES or
                source.get("split_protocol") != PAPER_SPLIT_PROTOCOL):
            raise ValueError("Prepared dataset does not match the original paper source")
        for split in ("train", "val", "test"):
            load_split(output, split, tasks)
        print("Original paper data already prepared and verified:", output.resolve())
        return output
    raw = Path(raw_dir) if raw_dir is not None else output.parent.parent / "paper" / "splice_sites_acceptor"
    paths = _paper_files(raw)
    parts = {split: _paper_records(path, split) for split, path in paths.items()}
    if {r["sequence"] for r in parts["train"]} & {r["sequence"] for r in parts["test"]}:
        raise ValueError("Unexpected sequence overlap in the checksum-pinned paper train/test files")
    rows = parts["train"] + parts["test"]
    counts = Counter(f"{r['task']}/{r['split']}/{r['label']}" for r in rows)
    # The official loader maps val to test. Materialize that alias for this
    # engine's existing three-file interface without repartitioning any data.
    # Preserve all original labels, including the contradictory training pair.
    parts["val"] = [dict(row, split="val", id="val:" + row["id"].partition(":")[2]) for row in parts["test"]]
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "val", "test"):
        write_rows(output / (split + ".csv"), parts[split])
        files[split] = sha(output / (split + ".csv"))
    prepared_counts = Counter(f"{r['task']}/{split}/{r['label']}" for split, subset in parts.items() for r in subset)
    write_json(output / "manifest.json", dict(files=files, split_seed=None,
        counts=dict(sorted(prepared_counts.items())), split_protocol=PAPER_SPLIT_PROTOCOL,
        grouping="original partitions unchanged; val and test deliberately reference the same examples"))
    labels = {}
    for row in rows:
        labels.setdefault(row["sequence"], set()).add(row["label"])
    conflicts = sum(len(values) > 1 for values in labels.values())
    write_json(source_path, dict(repository=PAPER_IMAGE, revision=revision, tasks=tasks,
        raw_sha256=PAPER_HASHES, original_counts=dict(sorted(counts.items())),
        split_protocol=PAPER_SPLIT_PROTOCOL,
        validation="original paper test FASTA, matching official val-to-test alias",
        test="same original paper holdout as validation; not an independent test partition",
        conflicting_sequence_label_groups=conflicts,
        conflicting_label_policy="preserve original records and labels unchanged",
        site_indices="not supplied; never inferred from the sequence midpoint"))
    print("Prepared original paper data:", output.resolve(), flush=True)
    if conflicts:
        print("Preserved %d original sequence group(s) with conflicting labels; no rows dropped." % conflicts, flush=True)
    return output
