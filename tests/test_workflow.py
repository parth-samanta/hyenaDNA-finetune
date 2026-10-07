"""Training, checkpoint evaluation, sweep recovery, and repeatable study runs."""
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from filelock import FileLock
from hyenadna.config import load_config
from hyenadna.data import prepare, write_json
from hyenadna.engine import fit, test as evaluate_test
from hyenadna.experiments import generate, run_manifest, report, test_selected
from hyenadna.workflow import load_study, run_study, validate_manifest
from _fixtures import small, rows_fixture
from hyenadna.paper_data import PAPER_IMAGE, PAPER_REVISION, PAPER_SPLIT_PROTOCOL

class WorkflowTests(unittest.TestCase):

    def setUp(self):
        self.original_cwd = Path.cwd()
        self.original_env = {k: os.environ.get(k) for k in ('HF_HOME', 'HF_HUB_CACHE')}
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (self.root / 'configs').mkdir()
        cfg = small(epochs=1, patience=1, max_length=600)
        cfg["model_overrides"]["layer"]["l_max"] = 602
        (self.root / 'configs/base.json').write_text(json.dumps(cfg))
        study = dict(name='fixture', base='configs/base.json', tasks=['splice_sites_acceptor'], heads=['mean', 'regional_attention', 'residual_regional_attention'], seeds=[0], methods=['partial'], lr_trials=1, data_revision=PAPER_REVISION)
        (self.root / 'configs/study.json').write_text(json.dumps(study))

    def tearDown(self):
        os.chdir(self.original_cwd)
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def prepared_data(self, tasks, output, revision):
        self.assertEqual(tasks, ['splice_sites_acceptor'])
        self.assertEqual(revision, PAPER_REVISION)
        rows = [r for r in rows_fixture() if r['task'] == 'splice_sites_acceptor']
        prepare(rows, output)
        write_json(Path(output) / 'source.json', dict(repository=PAPER_IMAGE,
                    revision=revision, split_protocol=PAPER_SPLIT_PROTOCOL))

    def test_new_study_runs_all_steps_and_rerun_preserves_models_and_predictions(self):
        with contextlib.redirect_stdout(io.StringIO()), patch('hyenadna.paper_data.prepare_paper', side_effect=self.prepared_data), patch('hyenadna.download_checkpoint.download') as download:
            destination = run_study(self.root)
        download.assert_not_called()
        manifest = self.root / 'results/fixture/manifest.json'
        rows = json.loads(manifest.read_text())
        saved = {p: p.read_bytes() for row in rows for p in [Path(row['output']) / 'best.pt', Path(row['output']) / 'test_predictions.csv', Path(row['output']) / 'test.json']}
        with contextlib.redirect_stdout(io.StringIO()), patch('hyenadna.engine.fit') as fit, patch('hyenadna.engine.test') as test, patch('hyenadna.paper_data.prepare_paper') as prepare_data:
            self.assertEqual(run_study(self.root), destination)
        fit.assert_not_called()
        test.assert_not_called()
        prepare_data.assert_not_called()
        self.assertTrue(all((p.read_bytes() == value for p, value in saved.items())))
        with (destination / 'comparison.csv').open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)
        self.assertTrue((destination / 'README.md').exists())

    def test_changed_training_controls_stop_before_reusing_results(self):
        study, cfg = load_study(self.root)
        folder = self.root / 'results/fixture'
        manifest = generate(cfg, folder, study['tasks'], study['heads'], study['seeds'], study['methods'])
        cfg['epochs'] = 2
        with self.assertRaisesRegex(ValueError, 'epochs'):
            validate_manifest(manifest, study, cfg)
        changed = json.loads((self.root / 'configs/base.json').read_text())
        changed['epochs'] = 2
        (self.root / 'configs/base.json').write_text(json.dumps(changed))
        with patch('hyenadna.workflow.run_manifest') as runner, patch('hyenadna.workflow.report') as reporter, self.assertRaisesRegex(ValueError, 'epochs'):
            run_study(self.root)
        runner.assert_not_called()
        reporter.assert_not_called()

    def test_workflow_lock_prevents_concurrent_runs(self):
        folder = self.root / 'results/fixture'
        folder.mkdir(parents=True)
        with FileLock(str(folder / '.workflow.lock')):
            with self.assertRaisesRegex(RuntimeError, 'Another workflow'), patch('hyenadna.workflow.generate') as generator:
                run_study(self.root)
            generator.assert_not_called()

    def test_invalid_name_cannot_escape_results_folder(self):
        path = self.root / 'configs/study.json'
        study = json.loads(path.read_text())
        study['name'] = '../outside'
        path.write_text(json.dumps(study))
        with self.assertRaisesRegex(ValueError, 'simple folder'):
            load_study(self.root)

    def test_default_study_uses_original_source_and_exports_split_protocol(self):
        loaded, cfg = load_study(self.root)
        self.assertEqual(loaded['data_revision'], PAPER_REVISION)
        self.assertEqual(Path(cfg['data']), self.root / 'data/prepared/paper_splice_sites_acceptor')
        with contextlib.redirect_stdout(io.StringIO()), patch('hyenadna.paper_data.prepare_paper', side_effect=self.prepared_data) as paper:
            destination = run_study(self.root)
        paper.assert_called_once()
        self.assertEqual(destination, self.root / 'reports/fixture')
        self.assertIn('original paper', (destination / 'README.md').read_text(encoding='utf-8'))
        self.assertIn('same holdout', (destination / 'README.md').read_text(encoding='utf-8'))

    def test_changed_dataset_revision_is_rejected(self):
        path = self.root / 'configs/study.json'
        study = json.loads(path.read_text(encoding='utf-8'))
        study['data_revision'] = 'different'
        path.write_text(json.dumps(study))
        with self.assertRaisesRegex(ValueError, 'pinned revision'):
            load_study(self.root)

class SweepRestartTests(unittest.TestCase):

    def make_manifest(self, root, seeds):
        manifest = generate(small(), root / 'study', ['fixture'], ['mean'], seeds, ['partial'])
        return (manifest, json.loads(manifest.read_text()))

    def test_preserves_interrupted_run_and_skips_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            manifest, rows = self.make_manifest(Path(tmp), [0, 1, 2])
            original_manifest = manifest.read_bytes()
            complete, interrupted, fresh = [Path(r['output']) for r in rows]
            complete.mkdir(parents=True)
            write_json(complete / 'complete.json', {'sentinel': 'keep'})
            original_complete = (complete / 'complete.json').read_bytes()
            interrupted.mkdir(parents=True)
            write_json(interrupted / 'config.json', load_config(rows[1]['config']))
            (interrupted / 'best.pt').write_bytes(b'preserve checkpoint')
            (interrupted / 'history.jsonl').write_bytes(b'preserve history\n')
            calls = []

            def fit(config):
                out = Path(load_config(config)['output'])
                self.assertFalse(out.exists())
                out.mkdir(parents=True)
                write_json(out / 'complete.json', {'new': True})
                calls.append(config)
            with patch('hyenadna.engine.fit', side_effect=fit):
                run_manifest(manifest)
                run_manifest(manifest)
            self.assertEqual(calls, [rows[1]['config'], rows[2]['config']])
            archives = list((manifest.parent / 'interrupted').iterdir())
            self.assertEqual(len(archives), 1)
            self.assertEqual((archives[0] / 'best.pt').read_bytes(), b'preserve checkpoint')
            self.assertEqual((archives[0] / 'history.jsonl').read_bytes(), b'preserve history\n')
            self.assertEqual((complete / 'complete.json').read_bytes(), original_complete)
            self.assertEqual(manifest.read_bytes(), original_manifest)

    def test_refuses_unknown_or_changed_output(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                manifest, rows = self.make_manifest(Path(tmp), [0])
                out = Path(rows[0]['output'])
                out.mkdir(parents=True)
                (out / 'notes.txt').write_text('keep me')
                if changed:
                    cfg = load_config(rows[0]['config'])
                    cfg['head'] = 'max'
                    write_json(out / 'config.json', cfg)
                with patch('hyenadna.engine.fit') as fit, self.assertRaises(FileExistsError):
                    run_manifest(manifest)
                fit.assert_not_called()
                self.assertEqual((out / 'notes.txt').read_text(), 'keep me')

    def test_refuses_concurrent_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self.make_manifest(Path(tmp), [0])
            with FileLock(str(manifest.parent / '.sweep.lock')):
                with self.assertRaisesRegex(RuntimeError, 'Another process'), patch('hyenadna.engine.fit') as fit:
                    run_manifest(manifest)
                fit.assert_not_called()

class TrainingPipelineTests(unittest.TestCase):

    def test_fit_tapt_multitask_test_reload(self):
        for method in ('full', 'lora', 'adapters', 'gradual', 'frozen'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                data = Path(tmp) / 'data'
                prepare(rows_fixture(), data)
                cfg = small(method=method, data=str(data), output=str(Path(tmp) / 'run'), tasks=['promoter_all', 'splice_sites_acceptor'], epochs=3 if method == 'gradual' else 1, head_epochs=1, partial_epochs=1, tapt_epochs=1 if method == 'full' else 0, cache_features=method == 'frozen')
                run = fit(cfg)
                self.assertTrue((run / 'complete.json').exists())
                metrics = evaluate_test(run)
                self.assertEqual(set(metrics), set(cfg['tasks']))
                self.assertTrue((run / 'test_slices.json').exists())
                with self.assertRaises(FileExistsError):
                    evaluate_test(run)

    def test_local_head_fails_without_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = rows_fixture()
            for row in rows:
                row['site'] = -1
            prepare(rows, Path(tmp) / 'data')
            cfg = small(head='local_global', data=str(Path(tmp) / 'data'), output=str(Path(tmp) / 'run'))
            with self.assertRaises(ValueError):
                fit(cfg)

class PartialSweepTests(unittest.TestCase):

    def test_partial_sweep_fit_select_and_test(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            rows = rows_fixture()
            for row in rows:
                row['id'] = row['task'] + ':' + row['id']
                row['task'] = 'fixture'
            prepare(rows, root / 'data')
            cfg = small(data=str(root / 'data'), epochs=1)
            manifest = generate(cfg, root / 'sweep', ['fixture'], ['attention', 'site_regions', 'regional_attention', 'residual_regional_attention'], [0], ['partial'])
            configs = [json.loads(Path(r['config']).read_text()) for r in json.loads(manifest.read_text())]
            self.assertTrue(all((c['method'] == 'partial' and (not c['cache_features']) for c in configs)))
            run_manifest(manifest)
            report(manifest)
            test_selected(manifest)
            report(manifest)
            self.assertEqual(len(json.loads((manifest.parent / 'summary.json').read_text())['test']), 4)
if __name__ == '__main__':
    unittest.main()
