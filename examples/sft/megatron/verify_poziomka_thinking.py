#!/usr/bin/env python3
"""Read-only source/audit/cache comparison; does not rerun greedy selection."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from poziomka_data import (OFFSET_DTYPE, TOKEN_DTYPE, encode_record,
                           load_manifest, load_tokenizer, sha256)
from prepare_poziomka_sft import iter_records


def verify(manifest_path, tokenizer_path, per_shard=2, seed=42, source_root=None, shard_prefix=None):
    root = manifest_path.parent
    manifest = load_manifest(manifest_path)
    shards = [s for s in manifest['shards'] if shard_prefix is None or s['prefix'] == shard_prefix]
    if not shards:
        raise ValueError(f'Unknown shard prefix: {shard_prefix}')
    if manifest['long_policy'] != 'remove-reasoning':
        raise ValueError('Expected a greedy reasoning-removal cache')
    template = root / 'chat_template.jinja'
    if sha256(template) != manifest['template_sha256']:
        raise ValueError('Template hash mismatch')
    if sha256(tokenizer_path / 'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer hash mismatch')
    tokenizer = load_tokenizer(tokenizer_path, template.read_text())
    rng = np.random.default_rng(seed)
    totals = dict(records=0, removed_blocks=0, fallback_records=0)
    for shard in shards:
        # Audit rows are source indices; mapping is exact only without drops.
        if any(v for k, v in shard['stats'].items() if k.startswith('dropped_')):
            raise ValueError('Source-to-cache mapping requires a cache without dropped rows')
        audit_path = root / (shard['prefix'] + '.reasoning.jsonl')
        if not audit_path.exists():
            if shard['stats'].get('overlong_records', 0):
                raise ValueError(f'Missing audit: {audit_path}')
            continue
        audits = [json.loads(line) for line in audit_path.read_text().splitlines()]
        if len(audits) != shard['stats'].get('overlong_records', 0):
            raise ValueError(f'Audit count mismatch: {audit_path}')
        selected = {}
        # Sample both successful removals and fallback cases, separately.
        for fallback in (False, True):
            candidates = [a for a in audits if a['removed_blocks'] and
                          a['fallback_truncated'] == fallback]
            count = len(candidates) if per_shard == 0 else min(per_shard, len(candidates))
            for i in rng.choice(len(candidates), count, replace=False):
                a = candidates[int(i)]
                selected[a['row']] = a
        if not selected:
            continue
        source = Path(shard['source'])
        if source_root is not None:
            source = source_root / shard['split'] / source.name
        ids = np.memmap(root / (shard['prefix'] + '.tokens.bin'), dtype=TOKEN_DTYPE, mode='r')
        masks = np.memmap(root / (shard['prefix'] + '.masks.bin'), dtype=np.uint8, mode='r')
        offsets = np.memmap(root / (shard['prefix'] + '.offsets.bin'), dtype=OFFSET_DTYPE, mode='r')
        pending = set(selected)
        for row, original in enumerate(iter_records(source), 1):
            if row not in pending:
                continue
            audit = selected[row]
            if original.get('record_id') != audit['record_id']:
                raise ValueError(f'Source record ID mismatch: {source}:{row}')
            expected = copy.deepcopy(original)
            for block in audit['removed_blocks']:
                message = expected['messages'][block['message_index']]
                if message['role'] != 'assistant' or not message.get('reasoning_content'):
                    raise ValueError(f'Invalid removal: {source}:{row}')
                message['reasoning_content'] = None
                message['content'] = '<think>\n</think>\n' + (message.get('content') or '')
            flags = [bool(m.get('reasoning_content')) for m in expected['messages']
                     if m['role'] == 'assistant']
            profile = 'mixed' if any(flags) and not all(flags) else ('on' if any(flags) else 'off')
            if 'reasoning_profile' in original and audit['reasoning_profile'] != profile:
                raise ValueError(f'Wrong audit profile: {source}:{row}')
            expected_ids, expected_mask = encode_record(tokenizer, expected, manifest['loss_roles'])
            limit = manifest['seq_length'] + 1
            if len(expected_ids) != audit['tokens_before_fallback'] or (len(expected_ids) > limit) != audit['fallback_truncated']:
                raise ValueError(f'Wrong fallback audit: {source}:{row}')
            start, end = int(offsets[row - 1]), int(offsets[row])
            if not np.array_equal(ids[start:end], expected_ids[:limit]) or not np.array_equal(masks[start:end], expected_mask[:limit]):
                raise ValueError(f'Tokens/masks differ from corrected source: {source}:{row}')
            totals['records'] += 1
            totals['removed_blocks'] += len(audit['removed_blocks'])
            totals['fallback_records'] += audit['fallback_truncated']
            pending.remove(row)
            if not pending:
                break
        if pending:
            raise ValueError(f'Source missing audited rows: {source}')
        print(f"Verified {shard['prefix']}: {len(selected)} records", flush=True)
    if not totals['records']:
        raise ValueError('No reasoning-removal records checked')
    print(json.dumps(totals, indent=2))
    return totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, help='Relocated original dataset with train/validation subdirectories')
    parser.add_argument('--per-shard', type=int, default=2, help='Samples per outcome per shard; 0 checks every removal')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shard', help='Only this exact manifest prefix, e.g. train/shard_00000')
    args = parser.parse_args()
    if args.per_shard < 0:
        parser.error('--per-shard must be nonnegative')
    verify(args.manifest, args.tokenizer, args.per_shard, args.seed, args.source_root, args.shard)
