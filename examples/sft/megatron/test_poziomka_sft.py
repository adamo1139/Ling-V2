#!/usr/bin/env python3
"""CPU-only tests with the real local APT4 tokenizer; never loads model weights."""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from poziomka_data import (FORMAT, SPECIAL_IDS, MMapSFTDataset, encode_record,
                           load_tokenizer, verify_shard)
from prepare_poziomka_sft import initialize_worker, process_shard

HERE = Path(__file__).resolve().parent
TOKENIZER_PATH = None
REFERENCE_TEMPLATE = None


def conversation():
    return {"messages": [
        {"role": "user", "content": "Ile to dwa plus dwa?"},
        {"role": "assistant", "reasoning_content": "Dodaję dwie pary.", "content": "Cztery."},
        {"role": "user", "content": "A trzy plus trzy?"},
        {"role": "assistant", "content": "Sześć."}], "source_id": "fixture:1"}


class SFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = (HERE / "poziomka_chatml.jinja").read_text()
        cls.tokenizer = load_tokenizer(TOKENIZER_PATH, cls.template)
        initialize_worker(TOKENIZER_PATH, cls.template)

    def test_exact_render_and_user_assistant_mask(self):
        row = conversation()
        ids, mask = encode_record(self.tokenizer, row)
        rendered = self.tokenizer.apply_chat_template(row["messages"], tokenize=False)
        self.assertTrue(rendered.startswith("<s><|im_start|>user\n"))
        self.assertNotIn("system", rendered)
        self.assertNotIn("detailed thinking off", rendered)
        self.assertIn("<think>\nDodaję dwie pary.\n</think>\nCztery.", rendered)
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        self.assertEqual(ids.tolist(), encoded["input_ids"])
        # Independently compute expected user+assistant body spans, including EOS.
        spans = []
        for role in ("user", "assistant"):
            marker = f"<|im_start|>{role}\n"
            pos = 0
            while marker in rendered[pos:]:
                start = rendered.index(marker, pos) + len(marker)
                end = rendered.index("<|im_end|>", start) + len("<|im_end|>")
                spans.append((start, end))
                pos = end
        expected = [int(any(a <= start and end <= b and end > start for a, b in spans))
                    for start, end in encoded["offset_mapping"]]
        self.assertEqual(mask.tolist(), expected)
        self.assertEqual(sum((ids == 4) & mask.astype(bool)), 4)
        assistant_ids, assistant_mask = encode_record(self.tokenizer, row, ("assistant",))
        np.testing.assert_array_equal(ids, assistant_ids)
        self.assertEqual(int(assistant_mask[ids == 4].sum()), 2)
        self.assertLess(int(assistant_mask.sum()), int(mask.sum()))

    def test_template_compatibility_and_tools(self):
        row = conversation()
        row["messages"].insert(0, {"role": "system", "content": "Odpowiadaj po polsku."})
        row["tools"] = [{"type": "function", "function": {"name": "licz", "parameters": {}}}]
        text = self.tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
        self.assertEqual(text.count("<|im_start|>system\n"), 1)
        self.assertEqual(text.count("Dostępne narzędzia (JSON):"), 1)
        ids, mask = encode_record(self.tokenizer, row)
        # Parquet's JSON-string tools and null reasoning must produce identical tokens.
        parquet_row = json.loads(json.dumps(row))
        parquet_row["tools"] = json.dumps(row["tools"])
        parquet_row["messages"][0]["reasoning_content"] = None
        other_ids, other_mask = encode_record(self.tokenizer, parquet_row)
        np.testing.assert_array_equal(ids, other_ids)
        np.testing.assert_array_equal(mask, other_mask)
        if REFERENCE_TEMPLATE:
            reference = load_tokenizer(TOKENIZER_PATH, self.template)
            reference.chat_template = Path(REFERENCE_TEMPLATE).read_text()
            self.assertEqual(text, reference.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False))
            self.assertEqual(ids.tolist(), reference.apply_chat_template(row["messages"], tools=row["tools"]))

    def test_tool_call_and_tool_result(self):
        row = {"messages": [
            {"role": "user", "content": "Policz."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "function": {"name": "licz", "arguments": {"x": 2}}}]},
            {"role": "tool", "tool_call_id": "a", "name": "licz", "content": "4"},
            {"role": "assistant", "content": "Wynik: 4."}]}
        ids, mask = encode_record(self.tokenizer, row)
        self.assertTrue(np.all(mask[ids == 5]))
        self.assertTrue(np.all(mask[ids == 6]))
        self.assertEqual(int(mask[ids == 4].sum()), 3)  # User + 2 assistants, not tool.
        if REFERENCE_TEMPLATE:
            reference = load_tokenizer(TOKENIZER_PATH, self.template)
            reference.chat_template = Path(REFERENCE_TEMPLATE).read_text()
            self.assertEqual(ids.tolist(), reference.apply_chat_template(row["messages"]))

    def make_cache(self, root, rows, seq_length=128, policy="truncate"):
        source = root / "input.jsonl"
        source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        shard = process_shard((str(source), str(root), "train", "test", seq_length, policy))
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(dict(format=FORMAT, special_ids=SPECIAL_IDS,
                                            seq_length=seq_length, shards=[shard])))
        return shard, manifest

    def test_shift_padding_shuffle_and_pickle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [conversation() for _ in range(3)]
            for i, row in enumerate(rows):
                row["messages"][-1]["content"] += str(i)
            shard, manifest = self.make_cache(root, rows)
            verify_shard(root, shard, 128)
            ds = MMapSFTDataset(manifest, "train", num_samples=6, shuffle=False)
            ids, mask = encode_record(self.tokenizer, rows[0])
            item = ds[0]
            length = len(ids) - 1
            np.testing.assert_array_equal(item["tokens"][:length], ids[:-1])
            np.testing.assert_array_equal(item["labels"][:length], ids[1:])
            np.testing.assert_array_equal(item["loss_mask"][:length], mask[1:])
            self.assertTrue(np.all(item["tokens"][length:] == 2))
            self.assertFalse(item["loss_mask"][length:].any())
            self.assertEqual(item["labels"].dtype, np.int64)
            self.assertEqual(item["loss_mask"].dtype, np.float32)
            self.assertNotIn("attention_mask", item)
            self.assertEqual(int(item["loss_mask"].sum()), int(mask.sum()))
            # First and last PP stage see identical order, including restart mid-epoch.
            a = MMapSFTDataset(manifest, "train", num_samples=6, seed=42)
            b = MMapSFTDataset(manifest, "train", num_samples=6, seed=42)
            for index in (0, 4, 2, 5):
                np.testing.assert_array_equal(a[index]["tokens"], b[index]["tokens"])
            np.testing.assert_array_equal(pickle.loads(pickle.dumps(ds))[0]["tokens"], item["tokens"])

    def test_long_policies_and_no_fake_eos(self):
        row = {"messages": [{"role": "user", "content": "Hej"},
                            {"role": "assistant", "content": "odpowiedź " * 100}]}
        for policy in ("truncate", "drop", "error"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                if policy == "error":
                    with self.assertRaises(ValueError):
                        self.make_cache(root, [row], seq_length=32, policy=policy)
                    continue
                shard, manifest = self.make_cache(root, [row], seq_length=32, policy=policy)
                verify_shard(root, shard, 32)
                self.assertEqual(shard["records"], int(policy == "truncate"))
                if policy == "truncate":
                    item = MMapSFTDataset(manifest, "train")[0]
                    self.assertNotEqual(int(item["labels"][-1]), 4)
                    self.assertTrue(item["loss_mask"].any())

    def test_no_targets_and_corruption_detection(self):
        row = {"messages": [{"role": "system", "content": "instrukcja " * 100},
                            {"role": "user", "content": "Pytanie"},
                            {"role": "assistant", "content": "Tak."}]}
        with tempfile.TemporaryDirectory() as temporary:
            shard, _ = self.make_cache(Path(temporary), [row], seq_length=16)
            self.assertEqual(shard["records"], 0)
            self.assertEqual(shard["stats"]["dropped_no_targets_records"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard, _ = self.make_cache(root, [conversation()])
            path = root / "test.masks.bin"
            with path.open("r+b") as stream:
                stream.write(b"\x01")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                verify_shard(root, shard, 128)
            with self.assertRaises(ValueError):
                verify_shard(root, shard, 128, check_hashes=False)

    def test_spawn_workers_and_full_verification_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "validation"):
                (root / "input" / split).mkdir(parents=True)
                (root / "input" / split / "chunk_1.jsonl").write_text(json.dumps(conversation()) + "\n")
            command = [sys.executable, str(HERE / "prepare_poziomka_sft.py"),
                       "--input", str(root / "input"), "--output", str(root / "cache"),
                       "--tokenizer", str(TOKENIZER_PATH), "--workers", "2", "--seq-length", "128"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = root / "cache" / "manifest.json"
            result = subprocess.run([sys.executable, str(HERE / "prepare_poziomka_sft.py"),
                                     "--verify", str(manifest)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metadata = json.loads(manifest.read_text())
            self.assertTrue(metadata["verified"])
            self.assertEqual(metadata["loss_roles"], ["user", "assistant"])
            # Output must be protected against accidental reruns/overwrite.
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_shared_model_config_and_converter_guard(self):
        # This only sources an array declaration; never starts torchrun.
        result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s\\n" "${POZIOMKA_MODEL_ARGS[@]}"',
                                 "test", str(HERE / "poziomka_model_args.sh")],
                                capture_output=True, text=True, check=True)
        flags = result.stdout.splitlines()
        self.assertEqual(flags[flags.index("--num-experts") + 1], "128")
        self.assertEqual(flags[flags.index("--moe-router-topk") + 1], "32")
        self.assertEqual(flags[flags.index("--pipeline-model-parallel-size") + 1], "8")
        # Importing converter utilities does not initialize Megatron or CUDA.
        path = HERE.parents[2] / "tools" / "load_hf_save_dcp.py"
        spec = importlib.util.spec_from_file_location("poziomka_converter", path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        config = {
            "num_hidden_layers": 16, "hidden_size": 2048, "intermediate_size": 2048,
            "num_attention_heads": 16, "num_key_value_heads": 4, "num_experts": 128,
            "num_experts_per_tok": 32, "moe_intermediate_size": 320,
            "moe_shared_expert_intermediate_size": 320, "vocab_size": 32000,
            "rope_theta": 84000, "partial_rotary_factor": 0.5, "n_group": 8,
            "topk_group": 2, "routed_scaling_factor": 2.5, "score_function": "sigmoid",
            "rms_norm_eps": 1e-6, "use_qk_norm": True, "tie_word_embeddings": False,
            "first_k_dense_replace": 1, "rope_scaling": None,
        }
        args = SimpleNamespace(
            num_layers=16, hidden_size=2048, ffn_hidden_size=2048, num_attention_heads=16,
            num_query_groups=4, num_experts=128, moe_router_topk=32, moe_ffn_hidden_size=320,
            moe_shared_expert_intermediate_size=320, padded_vocab_size=32000,
            rotary_base=84000, rotary_percent=0.5, moe_router_num_groups=8,
            moe_router_group_topk=2, moe_router_topk_scaling_factor=2.5,
            moe_router_score_function="sigmoid", norm_epsilon=1e-6, qk_layernorm=True,
            untie_embeddings_and_output_weights=True, moe_layer_freq=[0] + [1] * 15,
            use_rope_scaling=False, tensor_model_parallel_size=1,
            expert_model_parallel_size=1, pipeline_model_parallel_size=8,
        )
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "config.json").write_text(json.dumps(config))
            converter.validate_hf_config(temporary, args)
            args.moe_router_topk = 16
            with self.assertRaisesRegex(ValueError, "num_experts_per_tok"):
                converter.validate_hf_config(temporary, args)
        import torch
        weight = torch.arange(16).reshape(8, 2)
        actual = converter.reverse_qkv_weight(weight, 4, 4, 2)
        self.assertTrue(torch.equal(actual, weight[[0, 1, 4, 6, 2, 3, 5, 7]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--reference-template", type=Path)
    args, remaining = parser.parse_known_args()
    TOKENIZER_PATH = args.tokenizer.resolve()
    REFERENCE_TEMPLATE = args.reference_template
    unittest.main(argv=[sys.argv[0]] + remaining)
