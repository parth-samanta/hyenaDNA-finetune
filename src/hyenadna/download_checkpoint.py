"""Download weights and config; no model code is downloaded or executed."""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

from hyenadna.data import sha, write_json


def download(repo="LongSafari/hyenadna-tiny-1k-seqlen-d256", output="checkpoints/tiny", revision="main"):
    paths = {}
    for name in ("config.json", "weights.ckpt"):
        path = hf_hub_download(repo, name, revision=revision, local_dir=output)
        paths[name] = sha(path)
        print(path)
    write_json(Path(output) / "download.json", dict(repository=repo, revision=revision, sha256=paths))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="LongSafari/hyenadna-tiny-1k-seqlen-d256")
    p.add_argument("--out", default="checkpoints/tiny")
    p.add_argument("--revision", default="main")
    a = p.parse_args()
    download(a.repo, a.out, a.revision)

