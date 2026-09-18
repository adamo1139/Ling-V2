"""APT4 conversation encoding and small, portable, memory-mapped SFT caches.

No Megatron imports: preprocessing and CPU tests don't need the GPU stack.
Stored masks refer to tokens themselves; next-token shifting happens exactly
once in MMapSFTDataset.__getitem__. One conversation per sequence, no packing.
"""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

FORMAT = "poziomka-sft-v2"
TOKEN_DTYPE = np.dtype("<u2")
OFFSET_DTYPE = np.dtype("<u8")
# Packing is a training-time view of the same cache, not a new on-disk format:
# record lengths come from the offsets that are already stored. Bins are capped
# so cu_seqlens has a fixed shape and can be collated and broadcast like any
# other batch tensor; trailing padding entries are trimmed in forward_step.
MAX_SUBSEQUENCES = 32
SPECIAL_IDS = {"<s>": 1, "</s>": 2, "<|im_start|>": 3, "<|im_end|>": 4,
               "<tool_call>": 5, "</tool_call>": 6}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tokenizer(path, template):
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(path), local_files_only=True)
    if len(tokenizer) != 32000:
        raise ValueError("Expected unchanged 32,000-token APT4 vocabulary")
    for token, expected in SPECIAL_IDS.items():
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError(f"Wrong APT4 token ID for {token}")
    tokenizer.bos_token, tokenizer.pad_token, tokenizer.eos_token = "<s>", "</s>", "<|im_end|>"
    tokenizer.chat_template = template
    if "generation" not in template:
        raise ValueError("Chat template must mark loss targets with generation blocks")
    return tokenizer


def encode_record(tokenizer, record, loss_roles=("all",)):
    if not loss_roles or (tuple(loss_roles) != ("all",) and set(loss_roles) - {"user", "assistant"}):
        raise ValueError("Loss roles must be all, or user and/or assistant")
    messages = [{k: v for k, v in m.items() if v is not None} for m in record["messages"]]
    if not messages or messages[0]["role"] == "assistant":
        raise ValueError("Expected a conversation starting with a prompt")
    if any(m["role"] not in ("system", "user", "assistant", "tool") for m in messages):
        raise ValueError("Unknown message role")
    tools = record.get("tools")
    if isinstance(tools, str):  # The Parquet repack stores tools as JSON text.
        tools = json.loads(tools)
    if tuple(loss_roles) == ("all",):
        ids = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=False,
        )
        if len(ids) < 2 or ids[0] != 1:
            raise ValueError("Expected BOS and at least one next-token target")
        # Cached records contain no padding. Literal special-token IDs are real
        # content here; only padding positions added by the dataset are masked.
        mask = np.ones(len(ids), dtype=np.uint8)
        mask[0] = 0
        return np.asarray(ids, dtype=TOKEN_DTYPE), mask
    encoded = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_assistant_tokens_mask=True, loss_roles=list(loss_roles),
    )
    ids = encoded["input_ids"]
    mask = encoded["assistant_masks"]
    if len(ids) != len(mask) or not ids or ids[0] != 1 or mask[0]:
        raise ValueError("Invalid BOS or loss mask")
    # Catches incompatible templates/tokenizer versions, including missing EOS supervision.
    turns = sum(m["role"] in loss_roles for m in messages)
    if not turns or sum(t == 4 and bool(m) for t, m in zip(ids, mask)) != turns:
        raise ValueError("Every selected turn must supervise its im_end token")
    if any(m and t in (1, 2, 3) for t, m in zip(ids, mask)):
        raise ValueError("BOS/PAD/role boundary must never be a loss target")
    return np.asarray(ids, dtype=TOKEN_DTYPE), np.asarray(mask, dtype=np.uint8)


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest["format"] != FORMAT or manifest["special_ids"] != SPECIAL_IDS:
        raise ValueError("Incompatible SFT cache; rebuild with the current preparer (v2)")
    return manifest


def verify_shard(root, shard, seq_length, check_hashes=True):
    """Scan EVERY cached token, mask and record boundary; bounded working memory."""
    root = Path(root)
    count, length = shard["records"], shard["tokens"]
    expected_sizes = {"tokens.bin": length * 2, "masks.bin": length, "offsets.bin": (count + 1) * 8}
    for suffix, size in expected_sizes.items():
        path = root / f'{shard["prefix"]}.{suffix}'
        if path.stat().st_size != size:
            raise ValueError(f"Wrong size: {path}")
        if check_hashes and sha256(path) != shard["sha256"][suffix]:
            raise ValueError(f"Checksum mismatch: {path}")
    offsets = np.memmap(root / f'{shard["prefix"]}.offsets.bin', dtype=OFFSET_DTYPE, mode="r")
    if offsets[0] != 0 or offsets[-1] != length:
        raise ValueError("Invalid offsets")
    if not count:
        if length:
            raise ValueError("Tokens without records")
        return
    ids = np.memmap(root / f'{shard["prefix"]}.tokens.bin', dtype=TOKEN_DTYPE, mode="r")
    masks = np.memmap(root / f'{shard["prefix"]}.masks.bin', dtype=np.uint8, mode="r")
    supervised = 0
    for i in range(count):
        start, end = int(offsets[i]), int(offsets[i + 1])
        if not 2 <= end - start <= seq_length + 1:
            raise ValueError(f"Invalid sequence length at record {i}")
        tokens, mask = ids[start:end], masks[start:end]
        if tokens[0] != 1 or mask[0] or tokens.max() >= 32000 or mask.max() > 1 or not mask[1:].any():
            raise ValueError(f"Invalid tokens/mask at record {i}")
        supervised += int(mask.sum())
    if supervised != shard["supervised_tokens"]:
        raise ValueError("Supervised-token count mismatch")


class MMapSFTDataset:
    """Megatron sampler supplies global indices; do NOT shard a second time by rank."""

    def __init__(self, manifest_path, split, num_samples=None, seed=42, shuffle=True,
                 pack=False, max_subsequences=MAX_SUBSEQUENCES):
        self.root = Path(manifest_path).resolve().parent
        self.manifest = load_manifest(manifest_path)
        self.seq_length = self.manifest["seq_length"]
        self.shards = [s for s in self.manifest["shards"] if s["split"] == split and s["records"]]
        self.ends = np.cumsum([s["records"] for s in self.shards], dtype=np.int64)
        if not len(self.ends):
            raise ValueError(f"No usable records in {split}")
        self.record_count = int(self.ends[-1])
        self.seed, self.shuffle = seed, shuffle
        self.pack, self.max_subsequences = pack, max_subsequences
        self.bins = self._build_bins() if pack else None
        # One sample is a packed window when packing, a single conversation otherwise.
        self.sample_count = len(self.bins) if pack else self.record_count
        self.num_samples = self.sample_count if num_samples is None else int(num_samples)
        self._maps, self._order, self._epoch = OrderedDict(), None, None
        # Size checks are cheap on training startup. Full content verification is offline.
        for shard in self.shards:
            for suffix, size in (("tokens.bin", shard["tokens"] * 2),
                                 ("masks.bin", shard["tokens"]),
                                 ("offsets.bin", (shard["records"] + 1) * 8)):
                if (self.root / f'{shard["prefix"]}.{suffix}').stat().st_size != size:
                    raise ValueError(f"Incomplete cache shard: {shard['prefix']}")

    def usable_lengths(self):
        """Positions each record contributes after the next-token shift: len(ids) - 1."""
        lengths = np.empty(self.record_count, dtype=np.int64)
        position = 0
        for shard in self.shards:
            offsets = np.memmap(self.root / f'{shard["prefix"]}.offsets.bin',
                                dtype=OFFSET_DTYPE, mode="r")
            count = shard["records"]
            lengths[position:position + count] = np.diff(offsets[:count + 1].astype(np.int64)) - 1
            position += count
        return lengths

    def _build_bins(self):
        """Deterministic next-fit packing over a seeded permutation.

        Computed once from lengths alone, so every rank and every dataloader worker
        derives the identical plan without communication. The bin *order* is what
        gets reshuffled per epoch, which keeps the sample count fixed across epochs
        -- repacking per epoch would change the epoch length and break index math.
        """
        lengths = self.usable_lengths()
        if lengths.max(initial=0) > self.seq_length:
            raise ValueError("Record longer than seq_length; cache and seq_length disagree")
        order = np.random.default_rng(self.seed).permutation(self.record_count)
        bins, current, space = [], [], self.seq_length
        for index in order:
            length = int(lengths[index])
            if length > space or len(current) >= self.max_subsequences:
                bins.append(np.asarray(current, dtype=np.int64))
                current, space = [], self.seq_length
            current.append(int(index))
            space -= length
        if current:
            bins.append(np.asarray(current, dtype=np.int64))
        return bins

    def packing_efficiency(self):
        """Real tokens over padded positions; the whole point of packing."""
        return float(self.usable_lengths().sum()) / (len(self.bins) * self.seq_length)

    def __len__(self):
        return self.num_samples

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_maps=OrderedDict(), _order=None, _epoch=None)
        return state

    def _shard_maps(self, shard_id):
        if shard_id not in self._maps:
            prefix = self.root / self.shards[shard_id]["prefix"]
            self._maps[shard_id] = tuple(np.memmap(str(prefix) + suffix, dtype=dtype, mode="r")
                for suffix, dtype in ((".tokens.bin", TOKEN_DTYPE), (".masks.bin", np.uint8),
                                      (".offsets.bin", OFFSET_DTYPE)))
            # A packed bin can touch many shards at once, so keep more of them open.
            if len(self._maps) > (self.max_subsequences if self.pack else 8):
                self._maps.popitem(last=False)
        self._maps.move_to_end(shard_id)
        return self._maps[shard_id]

    def _locate(self, record):
        shard_id = int(np.searchsorted(self.ends, record, side="right"))
        return shard_id, record - (int(self.ends[shard_id - 1]) if shard_id else 0)

    def _shuffled_row(self, index):
        epoch, row = divmod(index, self.sample_count)
        if self.shuffle:
            if self._epoch != epoch:
                self._order = np.random.default_rng(self.seed + epoch).permutation(self.sample_count)
                self._epoch = epoch
            row = int(self._order[row])
        return row

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self._shuffled_row(index)
        # Valid vocabulary IDs even at masked positions: safe for fused TE cross entropy.
        tokens = np.full(self.seq_length, 2, dtype=np.int64)
        labels = np.full(self.seq_length, 2, dtype=np.int64)
        loss_mask = np.zeros(self.seq_length, dtype=np.float32)
        position_ids = np.zeros(self.seq_length, dtype=np.int64)

        if not self.pack:
            shard_id, local_row = self._locate(row)
            ids, masks, offsets = self._shard_maps(shard_id)
            start, end = int(offsets[local_row]), int(offsets[local_row + 1])
            size = end - start - 1
            tokens[:size], labels[:size] = ids[start:end - 1], ids[start + 1:end]
            loss_mask[:size] = masks[start + 1:end]
            # No attention matrix: ordinary causal FlashAttention, right padding.
            position_ids[:] = np.arange(self.seq_length, dtype=np.int64)
            return dict(tokens=tokens, labels=labels, loss_mask=loss_mask,
                        position_ids=position_ids)

        # Packed: conversations laid end to end, each restarting its own positions.
        # Boundaries become cu_seqlens so thd attention never crosses a conversation.
        boundaries = [0]
        filled = 0
        for record in self.bins[row]:
            shard_id, local_row = self._locate(int(record))
            ids, masks, offsets = self._shard_maps(shard_id)
            start, end = int(offsets[local_row]), int(offsets[local_row + 1])
            size = end - start - 1
            window = slice(filled, filled + size)
            tokens[window], labels[window] = ids[start:end - 1], ids[start + 1:end]
            loss_mask[window] = masks[start + 1:end]
            position_ids[window] = np.arange(size, dtype=np.int64)
            filled += size
            boundaries.append(filled)
        if filled < self.seq_length:
            # Trailing padding is its own sub-sequence: it attends only to itself and
            # its loss is masked, so cu_seqlens covers the window with no padded/
            # unpadded special case to get wrong.
            position_ids[filled:] = np.arange(self.seq_length - filled, dtype=np.int64)
            boundaries.append(self.seq_length)
        # Fixed shape so this collates and broadcasts like any other batch tensor;
        # forward_step drops the repeated tail before building PackedSeqParams.
        cu_seqlens = np.full(self.max_subsequences + 2, self.seq_length, dtype=np.int32)
        cu_seqlens[:len(boundaries)] = boundaries
        return dict(tokens=tokens, labels=labels, loss_mask=loss_mask,
                    position_ids=position_ids, cu_seqlens=cu_seqlens)
