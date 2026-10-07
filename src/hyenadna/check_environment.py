import sys
import torch
from hyenadna.config import load_config
from hyenadna.models import Model


def check():
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    ids = torch.randint(7, 12, (2, 32), device=device)
    batch = dict(ids=ids, mask=ids.ne(4), task=torch.zeros(2, dtype=torch.long, device=device),
                 site=torch.full((2,), -1, device=device))
    for method in ("frozen", "partial", "lora"):
        cfg = load_config(dict(pretrained=False, model_config="", max_length=32, method=method, head="multi_attention"))
        model = Model(cfg)
        model.install_adaptation()
        model.set_stage(0)
        model.to(device).train()
        model(batch).sum().backward()
        assert model.projection[0].weight.grad is not None
        assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
        print(method, "with multi-attention forward/backward passed on", device)


if __name__ == "__main__":
    check()

