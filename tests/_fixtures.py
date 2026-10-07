"""Small CPU models and synthetic data shared by the test suites."""
import random
import torch
from hyenadna.config import load_config
torch.set_num_threads(2)

def small(**updates):
    cfg = dict(pretrained=False, model_config='', method='full', max_length=24, micro_batch=2, batch_size=4, device='cpu', embed_dropout=0.0, resid_dropout=0.0, head_dropout=0.0, model_overrides=dict(d_model=16, n_layer=2, d_inner=32, layer=dict(emb_dim=5, filter_order=8, l_max=32, lr_pos_emb=0.0, modulate=True, w=10)))
    cfg.update(updates)
    return load_config(cfg)

def batch():
    ids = torch.tensor([[7, 8, 9, 10, 11, 4, 4], [10, 9, 8, 7, 4, 4, 4]])
    return dict(ids=ids, mask=ids.ne(4), task=torch.tensor([0, 0]), site=torch.tensor([2, 1]), label=torch.tensor([0, 1]))

def rows_fixture():
    rng = random.Random(81)
    rows = []
    for task in ('promoter_all', 'splice_sites_acceptor'):
        for split in ('train', 'val', 'test'):
            for i in range(8):
                rows.append(dict(id=split + str(i), sequence=''.join(rng.choices('ACGT', k=20)), task=task, label=i % 2, split=split, group='', site=10, subtype='', species='test', chrom=''))
    return rows
