#!/usr/bin/env python3
"""CPU tests for packed SFT batches: no GPU, no model, no Megatron import.

Packing bugs do not crash -- they silently train one conversation on another's
context, or shift labels by a position. So these check content equivalence
against the unpacked reader over the same cache, not just shapes.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poziomka_data import MMapSFTDataset, OFFSET_DTYPE, TOKEN_DTYPE


def build_cache(root, lengths, seq_length):
    """Write a cache directly: these tests are about the reader, not the preparer."""
    root = Path(root)
    (root / "train").mkdir(parents=True)
    tokens, masks, offsets = [], [], [0]
    for i, length in enumerate(lengths):
        # Distinct, recoverable content per record so cross-contamination is visible.
        ids = np.arange(100 + i * 1000, 100 + i * 1000 + length, dtype=TOKEN_DTYPE) % 32000
        ids[0] = 1  # BOS, as encode_record guarantees
        mask = np.ones(length, dtype=np.uint8)
        mask[0] = 0
        tokens.append(ids)
        masks.append(mask)
        offsets.append(offsets[-1] + length)
    prefix = root / "train" / "shard_00000"
    np.concatenate(tokens).tofile(str(prefix) + ".tokens.bin")
    np.concatenate(masks).tofile(str(prefix) + ".masks.bin")
    np.asarray(offsets, dtype=OFFSET_DTYPE).tofile(str(prefix) + ".offsets.bin")
    shard = dict(split="train", prefix="train/shard_00000", records=len(lengths),
                 tokens=int(offsets[-1]), supervised_tokens=int(sum(l - 1 for l in lengths)))
    manifest = dict(format="poziomka-sft-v2", seq_length=seq_length, packing=False,
                    special_ids={"<s>": 1, "</s>": 2, "<|im_start|>": 3, "<|im_end|>": 4,
                                 "<tool_call>": 5, "</tool_call>": 6},
                    loss_roles=["all"], shards=[shard])
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


class PackingTests(unittest.TestCase):
    SEQ = 128
    LENGTHS = [40, 31, 17, 60, 9, 25, 51, 13, 44, 22, 8, 37]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.manifest = build_cache(self.temporary.name, self.LENGTHS, self.SEQ)

    def tearDown(self):
        self.temporary.cleanup()

    def packed(self, **kwargs):
        return MMapSFTDataset(self.manifest, "train", pack=True, shuffle=False, **kwargs)

    def test_every_record_appears_exactly_once(self):
        dataset = self.packed()
        seen = np.concatenate(dataset.bins)
        self.assertEqual(sorted(seen.tolist()), list(range(len(self.LENGTHS))))

    def test_no_bin_exceeds_the_window(self):
        dataset = self.packed()
        lengths = dataset.usable_lengths()
        for records in dataset.bins:
            self.assertLessEqual(int(lengths[records].sum()), self.SEQ)

    def test_content_matches_the_unpacked_reader(self):
        """The decisive test: same cache, both readers, identical per-record content."""
        packed = self.packed()
        plain = MMapSFTDataset(self.manifest, "train", shuffle=False)
        reference = {}
        for row in range(len(self.LENGTHS)):
            sample = plain[row]
            size = self.LENGTHS[row] - 1
            reference[row] = (sample["tokens"][:size].copy(),
                              sample["labels"][:size].copy(),
                              sample["loss_mask"][:size].copy())
        for index, records in enumerate(packed.bins):
            sample = packed[index]
            cuts = sample["cu_seqlens"]
            for position, record in enumerate(records):
                start, end = int(cuts[position]), int(cuts[position + 1])
                tokens, labels, mask = reference[int(record)]
                self.assertEqual(end - start, len(tokens))
                np.testing.assert_array_equal(sample["tokens"][start:end], tokens)
                np.testing.assert_array_equal(sample["labels"][start:end], labels)
                np.testing.assert_array_equal(sample["loss_mask"][start:end], mask)

    def test_position_ids_restart_per_conversation(self):
        packed = self.packed()
        for index, records in enumerate(packed.bins):
            sample = packed[index]
            cuts = sample["cu_seqlens"]
            for position in range(len(records)):
                start, end = int(cuts[position]), int(cuts[position + 1])
                np.testing.assert_array_equal(sample["position_ids"][start:end],
                                              np.arange(end - start))

    def test_padding_is_masked_and_covered_by_cu_seqlens(self):
        packed = self.packed()
        lengths = packed.usable_lengths()
        for index, records in enumerate(packed.bins):
            sample = packed[index]
            filled = int(lengths[records].sum())
            self.assertEqual(sample["loss_mask"][filled:].sum(), 0.0)
            boundaries = np.unique(sample["cu_seqlens"])
            self.assertEqual(int(boundaries[0]), 0)
            self.assertEqual(int(boundaries[-1]), self.SEQ)

    def test_cu_seqlens_shape_is_fixed_across_samples(self):
        """Needed so the batch can be collated and broadcast across pipeline ranks."""
        packed = self.packed()
        shapes = {packed[i]["cu_seqlens"].shape for i in range(len(packed.bins))}
        self.assertEqual(len(shapes), 1)

    def test_bins_respect_the_subsequence_cap(self):
        packed = self.packed(max_subsequences=3)
        self.assertTrue(all(len(records) <= 3 for records in packed.bins))

    def test_plan_is_deterministic_across_instances(self):
        """Every rank and dataloader worker must derive the same plan independently."""
        first, second = self.packed(), self.packed()
        for a, b in zip(first.bins, second.bins):
            np.testing.assert_array_equal(a, b)

    def test_packing_beats_no_packing(self):
        packed = self.packed()
        self.assertLess(len(packed.bins), len(self.LENGTHS))
        self.assertGreater(packed.packing_efficiency(), 0.5)

    def test_unpacked_reader_is_unchanged(self):
        """Packing must not alter the existing path: run 2's behaviour is the baseline."""
        plain = MMapSFTDataset(self.manifest, "train", shuffle=False)
        sample = plain[0]
        size = self.LENGTHS[0] - 1
        self.assertEqual(len(sample["tokens"]), self.SEQ)
        np.testing.assert_array_equal(sample["position_ids"], np.arange(self.SEQ))
        self.assertEqual(sample["loss_mask"][size:].sum(), 0.0)
        self.assertNotIn("cu_seqlens", sample)


if __name__ == "__main__":
    unittest.main(verbosity=2)
