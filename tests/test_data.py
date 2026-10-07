"""Dataset splits, leakage checks, coordinates, and cached dataset reuse."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hyenadna.data import prepare, load_split, subsample
from _fixtures import rows_fixture
from hyenadna.paper_data import prepare_paper, _paper_records, PAPER_REVISION, PAPER_IMAGE

class DataTests(unittest.TestCase):

    def test_duplicate_leakage_and_group_leakage(self):
        for group_only in (False, True):
            rows = rows_fixture()
            a = next((r for r in rows if r['split'] == 'train'))
            b = next((r for r in rows if r['split'] == 'test'))
            if group_only:
                a['group'] = b['group'] = 'homology_cluster_1'
            else:
                b['sequence'] = a['sequence']
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                prepare(rows, tmp)

    def test_subsets_are_nested(self):
        rows = rows_fixture()
        small_set = {(r['task'], r['id']) for r in subsample(rows, 0.25, 55)}
        big_set = {(r['task'], r['id']) for r in subsample(rows, 0.5, 55)}
        self.assertTrue(small_set <= big_set)

    def test_split_generation_and_integrity(self):
        rows = rows_fixture()
        for r in rows:
            if r['split'] == 'val':
                r['split'] = 'train'
                r['id'] = 'extra_' + r['id']
        with tempfile.TemporaryDirectory() as tmp:
            a, b = (Path(tmp) / 'a', Path(tmp) / 'b')
            prepare(copy.deepcopy(rows), a, 0.25)
            prepare(copy.deepcopy(rows), b, 0.25)
            self.assertEqual((a / 'val.csv').read_text(), (b / 'val.csv').read_text())
            with open(a / 'val.csv', 'a') as f:
                f.write('\n')
            with self.assertRaises(ValueError):
                load_split(a, 'val', ['promoter_all'])

class PaperDataTests(unittest.TestCase):
    def test_original_split_preserved_and_prepared_data_reused(self):
        rng = random.Random(311)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for split, n in (("train", 40), ("test", 8)):
                paths[split] = root / (split + ".fasta")
                sequences = [''.join(rng.choices('ACGT', k=600)) for _ in range(n)]
                if split == "train":
                    sequences[1] = sequences[0]
                paths[split].write_text("".join(
                    f">{split}_{i} label_{i % 2}\n{sequences[i]}\n" for i in range(n)))
            with self.assertRaisesRegex(ValueError, "Conflicting labels"):
                prepare(_paper_records(paths["train"], "train") + _paper_records(paths["test"], "test"), root / "strict")
            expected_test = {(r["sequence"], r["label"]) for r in _paper_records(paths["test"], "test")}
            output = root / "prepared"
            with patch("hyenadna.paper_data._paper_files", return_value=paths) as fetch:
                prepare_paper(["splice_sites_acceptor"], output)
                before = {p.name: p.read_bytes() for p in output.iterdir()}
                prepare_paper(["splice_sites_acceptor"], output)
                fetch.assert_called_once()
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            partitions = {split: load_split(output, split, ["splice_sites_acceptor"]) for split in ("train", "val", "test")}
            self.assertEqual({(r["sequence"], r["label"]) for r in partitions["test"]}, expected_test)
            self.assertEqual(len(partitions["train"]), 40)
            self.assertEqual(len(partitions["val"]), 8)
            self.assertEqual([(r["sequence"], r["label"]) for r in partitions["val"]],
                             [(r["sequence"], r["label"]) for r in partitions["test"]])
            contradictory = _paper_records(paths["train"], "train")[0]["sequence"]
            pair_splits = {split for split, rows in partitions.items() if any(r["sequence"] == contradictory for r in rows)}
            self.assertEqual(len(pair_splits), 1)
            self.assertTrue(all(r["site"] == -1 for rows in partitions.values() for r in rows))
            with self.assertRaises(ValueError):
                prepare_paper(["splice_sites_donor"], output)
            with self.assertRaises(ValueError):
                prepare_paper(["splice_sites_acceptor"], output, revision="different")
            with (output / "test.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Prepared data changed"):
                prepare_paper(["splice_sites_acceptor"], output)

    def test_fasta_rejects_invalid_labels_and_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.fasta"
            for content in (">record label_2\n" + "A" * 600 + "\n",
                            ">record label_1\n" + "A" * 600 + "\n>record label_1\n" + "C" * 600 + "\n",
                            ">record label_1\nACGT\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    _paper_records(path, "train")

if __name__ == '__main__':
    unittest.main()
