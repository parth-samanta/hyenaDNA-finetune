"""Pooling formulas, masking, gradients, model loading, and fine-tuning behavior."""
import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
import torch
from hyenadna.config import load_config
from hyenadna.data import reverse_complement
from hyenadna.experiments import generate
from hyenadna.models import Model, Readout, LoRALinear, load_backbone, make_optimizer
from _fixtures import small, batch
ALL_HEADS = ['mean', 'max', 'last', 'mean_max_last', 'attention', 'multi_attention', 'regional', 'cnn', 'local_global', 'site_regions', 'regional_attention', 'residual_regional_attention']
REGIONAL_HEADS = ('regional_attention', 'residual_regional_attention')

class ModelTests(unittest.TestCase):

    def test_heads_and_bidirectional_backward(self):
        for head in ('mean', 'max', 'last', 'mean_max_last', 'attention', 'multi_attention', 'regional', 'cnn', 'local_global', 'site_regions', 'regional_attention', 'residual_regional_attention'):
            for bidirectional in (False, True):
                with self.subTest(head=head, bidirectional=bidirectional):
                    model = Model(small(head=head, bidirectional=bidirectional))
                    logits = model(batch())
                    self.assertEqual(tuple(logits.shape), (2, 2))
                    logits.sum().backward()
                    self.assertTrue(all((torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)))

    def test_rc_round_trip_and_padding(self):
        b = batch()
        rc = reverse_complement(b['ids'], b['mask'])
        self.assertEqual(rc[0].tolist(), [11, 7, 8, 9, 10, 4, 4])
        self.assertTrue(torch.equal(reverse_complement(rc, b['mask']), b['ids']))

    def test_frozen_parameters_do_not_move(self):
        for method in ('frozen', 'partial', 'lora', 'mixers', 'mlps', 'adapters'):
            with self.subTest(method=method):
                cfg = small(method=method)
                model = Model(cfg)
                model.install_adaptation()
                model.set_stage(0)
                model.train()
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                optimizer = make_optimizer(model, cfg)
                torch.nn.functional.cross_entropy(model(batch()), batch()['label']).backward()
                optimizer.step()
                moved = []
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        self.assertTrue(torch.equal(p, before[name]), name)
                    if not torch.equal(p, before[name]):
                        moved.append(name)
                self.assertTrue(any((n.startswith('heads.') for n in moved)))
                if method != 'frozen':
                    self.assertTrue(any((n.startswith('encoder.') for n in moved)), method)

    def test_gradual_stages(self):
        model = Model(small(method='gradual', head_epochs=1, partial_epochs=1))
        counts = []
        for epoch, stage in enumerate(('frozen', 'partial', 'full')):
            model.set_stage(epoch)
            self.assertEqual(model.stage, stage)
            counts.append(sum((p.numel() for p in model.encoder.parameters() if p.requires_grad)))
        self.assertTrue(counts[0] == 0 < counts[1] < counts[2])

    def test_lora_initially_preserves_output(self):
        linear = torch.nn.Linear(7, 9)
        lora = LoRALinear(linear, 4, 8)
        x = torch.randn(2, 5, 7)
        self.assertTrue(torch.equal(linear(x), lora(x)))

    def test_checkpoint_wrappers_and_shape_validation(self):
        model = Model(small())
        source = {'model.' + k.replace('.mixer.', '.mixer.layer.'): v for k, v in model.encoder.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pt'
            torch.save({'state_dict': source}, path)
            target = Model(small())
            load_backbone(target.encoder, path)
            for k, v in model.encoder.state_dict().items():
                self.assertTrue(torch.equal(v, target.encoder.state_dict()[k]))
            source.pop(next(iter(source)))
            torch.save({'state_dict': source}, path)
            with self.assertRaises(ValueError):
                load_backbone(target.encoder, path)

    def test_baselines_and_multihead(self):
        for kind in ('hyena', 'gpt', 'cnn'):
            model = Model(small(model=kind, tasks=['donor', 'acceptor']))
            b = batch()
            b['task'] = torch.tensor([0, 1])
            model(b).sum().backward()
            self.assertTrue(all((h[1].weight.grad is not None for h in model.heads)))

    def test_padding_does_not_change_causal_features(self):
        model = Model(small()).eval()
        b = batch()
        short = model.encoder(b['ids'][:, :5])
        long = model.encoder(b['ids'])
        self.assertTrue(torch.allclose(short, long[:, :5], atol=1e-05))

class AggregationTests(unittest.TestCase):

    def test_mean_max_last_values(self):
        h = torch.tensor([[[1.0, -5.0], [3.0, -2.0], [999.0, 999.0]]])
        mask = torch.tensor([[True, True, False]])
        site = torch.tensor([0])
        expected = {'mean': [2.0, -3.5], 'max': [3.0, -2.0], 'last': [3.0, -2.0], 'mean_max_last': [2.0, -3.5, 3.0, -2.0, 3.0, -2.0]}
        for head, values in expected.items():
            actual = Readout(2, small(head=head))(h, mask, site)
            self.assertTrue(torch.equal(actual, torch.tensor([values])), head)

    def test_padding_invariance_all_heads(self):
        h = torch.randn(2, 6, 16)
        mask = torch.ones(2, 6, dtype=torch.bool)
        site = torch.tensor([2, 4])
        padded = torch.cat([h, torch.full((2, 5, 16), 100000.0)], dim=1)
        padded_mask = torch.cat([mask, torch.zeros(2, 5, dtype=torch.bool)], dim=1)
        for head in ALL_HEADS:
            with self.subTest(head=head):
                pool = Readout(16, small(head=head)).eval()
                a, b = (pool(h, mask, site), pool(padded, padded_mask, site))
                self.assertTrue(torch.allclose(a, b, atol=1e-06), head)

    def test_attention_uniform_when_scores_zero(self):
        h = torch.randn(2, 5, 16)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        mean = Readout(16, small(head='mean'))(h, mask, torch.tensor([-1, -1]))
        for head in ('attention', 'multi_attention'):
            pool = Readout(16, small(head=head, attention_summaries=3))
            torch.nn.init.zeros_(pool.score.weight)
            torch.nn.init.zeros_(pool.score.bias)
            pooled = pool(h, mask, torch.tensor([-1, -1]))
            self.assertTrue(torch.allclose(pooled, mean.repeat(1, 3 if head == 'multi_attention' else 1), atol=1e-06))

    def test_regional_empty_bins(self):
        pool = Readout(1, small(head='regional', regional_bins=4))
        value = pool(torch.tensor([[[2.0], [4.0]]]), torch.ones(1, 2, dtype=torch.bool), torch.tensor([-1]))
        self.assertTrue(torch.equal(value, torch.tensor([[2.0, 0.0, 4.0, 0.0]])))

    def test_site_regions(self):
        pool = Readout(1, small(head='site_regions', local_radius=1))
        h = torch.arange(6.0).reshape(1, 6, 1)
        mask = torch.ones(1, 6, dtype=torch.bool)
        self.assertTrue(torch.equal(pool(h, mask, torch.tensor([2])), torch.tensor([[0.0, 2.0, 4.5]])))
        self.assertTrue(torch.isfinite(pool(h, mask, torch.tensor([0]))).all())
        for site in (-1, 6):
            with self.assertRaises(ValueError):
                pool(h, mask, torch.tensor([site]))
        mask[0, 5] = False
        with self.assertRaises(ValueError):
            pool(h, mask, torch.tensor([5]))

    def test_all_padding_rejected(self):
        for head in ALL_HEADS:
            with self.assertRaises(ValueError):
                Readout(16, small(head=head))(torch.zeros(1, 4, 16), torch.zeros(1, 4, dtype=torch.bool), torch.tensor([0]))

    def test_cached_summaries_keep_projection_trainable(self):
        cfg = small(method='frozen', head='mean_max_last')
        live = Model(cfg)
        live.set_stage(0)
        cached = copy.deepcopy(live)
        b = batch()
        live.train()
        cached.train()
        with torch.no_grad():
            features = cached.features(b)
        opt_live, opt_cached = (make_optimizer(live, cfg), make_optimizer(cached, cfg))
        for model, optimizer, inputs in ((live, opt_live, b), (cached, opt_cached, dict(b, features=features))):
            before = model.projection[0].weight.detach().clone()
            torch.nn.functional.cross_entropy(model(inputs), b['label']).backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.projection[0].weight))
        for (name, a), (_, bparam) in zip(live.named_parameters(), cached.named_parameters()):
            self.assertTrue(torch.allclose(a, bparam, atol=1e-07), name)

    def test_fixed_output_width(self):
        for head in ALL_HEADS:
            model = Model(small(head=head))
            features = model.features(batch())
            self.assertEqual(model.projection(features).shape, (2, 16))

class RegionalAttentionTests(unittest.TestCase):

    def test_hand_calculated_values_and_local_normalization(self):
        h = torch.arange(4.0).reshape(1, 4, 1)
        mask = torch.ones(1, 4, dtype=torch.bool)
        for head, expected in zip(REGIONAL_HEADS, ([2 / 3, 8 / 3], [7 / 12, 31 / 12])):
            pool = Readout(1, small(head=head, regional_bins=2))
            with torch.no_grad():
                pool.regional_query.fill_(math.log(2))
            actual = pool(h, mask, torch.tensor([-1]))
            torch.testing.assert_close(actual, torch.tensor([expected]))
            changed = h.clone()
            changed[:, 2:] += 100
            torch.testing.assert_close(pool(changed, mask, torch.tensor([-1]))[:, :1], actual[:, :1])

    def test_short_inputs_empty_bins_nan_padding_and_gradients(self):
        torch.manual_seed(3)
        h = torch.randn(3, 7, 8, requires_grad=True)
        mask = torch.tensor([[1] * 7, [1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
        site = torch.full((3,), -1)
        for head in REGIONAL_HEADS:
            pool = Readout(8, small(head=head))
            with torch.no_grad():
                pool.regional_query.normal_()
            a = pool(h, mask, site)
            padded = torch.cat([h.detach(), torch.full((3, 4, 8), float('nan'))], 1)
            padded_mask = torch.cat([mask, torch.zeros(3, 4, dtype=torch.bool)], 1)
            torch.testing.assert_close(a, pool(padded, padded_mask, site), atol=1e-06, rtol=1e-05)
            grads = torch.autograd.grad(a.square().sum(), [h] + list(pool.parameters()))
            self.assertTrue(all((torch.isfinite(g).all() for g in grads)))
            self.assertEqual(grads[0][~mask].abs().max().item(), 0)
            self.assertGreater(grads[1].abs().max().item(), 0)
            if head == 'residual_regional_attention':
                self.assertGreater(grads[2].abs().max().item(), 0)
            torch.testing.assert_close(a[2, 8:], torch.zeros(24))

    def test_initialization_matches_regional_model_and_preserves_rng(self):
        for head in REGIONAL_HEADS:
            torch.manual_seed(17)
            baseline = Model(small(head='regional', method='partial'))
            torch.manual_seed(17)
            candidate = Model(small(head=head, method='partial'))
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(baseline(batch()), candidate(batch()), atol=1e-06, rtol=1e-05)
            baseline.set_stage(0)
            candidate.set_stage(0)
            extra = sum((p.numel() for p in candidate.parameters() if p.requires_grad)) - sum((p.numel() for p in baseline.parameters() if p.requires_grad))
            self.assertEqual(extra, 16 + (4 if head == 'residual_regional_attention' else 0))

    def test_initialization_score_gradient_is_available(self):
        torch.manual_seed(19)
        h = torch.randn(2, 24, 16, requires_grad=True)
        mask = torch.ones(2, 24, dtype=torch.bool)
        for head in REGIONAL_HEADS:
            pool = Readout(16, small(head=head))
            pool(h, mask, torch.full((2,), -1)).square().sum().backward()
            self.assertGreater(pool.regional_query.grad.abs().max().item(), 0)
            if head == 'residual_regional_attention':
                torch.testing.assert_close(pool.regional_gate_logits.grad, torch.zeros(4), atol=1e-06, rtol=0)

    def test_actual_tiny_model_counts(self):
        for head, expected in zip(REGIONAL_HEADS, (1081922, 1081926)):
            cfg = load_config(dict(pretrained=False, model_config='', head=head, method='partial', max_length=600))
            model = Model(cfg)
            model.set_stage(0)
            self.assertEqual(sum((p.numel() for p in model.parameters() if p.requires_grad)), expected)
            self.assertEqual(model.projection[0].in_features, 1024)

    def test_learned_heads_cannot_cache_frozen_summaries(self):
        for head in REGIONAL_HEADS:
            with self.assertRaises(ValueError):
                small(head=head, method='frozen', cache_features=True)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = generate(small(), Path(tmp) / 'sweep', ['fixture'], REGIONAL_HEADS, [0], ['frozen'])
            for row in json.loads(path.read_text()):
                self.assertFalse(json.loads(Path(row['config']).read_text())['cache_features'])

class PartialTuningTests(unittest.TestCase):

    def test_partial_updates_only_final_block_and_normalization(self):
        for head in ('mean', 'max', 'last', 'attention', 'regional', 'site_regions', 'regional_attention', 'residual_regional_attention'):
            with self.subTest(head=head):
                cfg = small(method='partial', head=head, local_radius=1)
                model = Model(cfg)
                model.install_adaptation()
                model.set_stage(0)
                model.train()
                before = {n: p.detach().clone() for n, p in model.encoder.named_parameters()}
                optimizer = make_optimizer(model, cfg)
                torch.nn.functional.cross_entropy(model(batch()), batch()['label']).backward()
                optimizer.step()
                moved = []
                for name, p in model.encoder.named_parameters():
                    allowed = name.startswith(('backbone.layers.1.', 'backbone.ln_f.'))
                    self.assertEqual(p.requires_grad, allowed, name)
                    if not allowed:
                        self.assertIsNone(p.grad, name)
                        self.assertTrue(torch.equal(p, before[name]), name)
                    elif not torch.equal(p, before[name]):
                        moved.append(name)
                self.assertTrue(any((n.startswith('backbone.layers.1.') for n in moved)))
                self.assertTrue(any((n.startswith('backbone.ln_f.') for n in moved)))
if __name__ == '__main__':
    unittest.main()
