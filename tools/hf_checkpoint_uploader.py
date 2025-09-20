#!/usr/bin/env python3
"""
Incremental Hugging Face Hub uploader for Megatron checkpoints.

Watches a Megatron checkpoint directory and, whenever a new iteration
is saved (iter_XXXXXXX), commits only that iteration folder (and the
latest_checkpointed_iteration.txt tracker) to a Hub repository.

Usage (run in a separate tmux session):

  export HUGGINGFACE_HUB_TOKEN=hf_xxx   # or huggingface-cli login
  export HF_HUB_ENABLE_HF_TRANSFER=1     # optional, faster uploads
  python tools/hf_checkpoint_uploader.py \
      --ckpt-dir poziomka_1 \
      --repo-id your-org/your-repo \
      --revision main \
      --poll-interval 60

Dependencies:
  pip install -U huggingface_hub hf_transfer
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, List

try:
    from huggingface_hub import snapshot_upload, create_repo, HfApi
except Exception as e:
    print("ERROR: huggingface_hub is required. Install via 'pip install -U huggingface_hub'.", file=sys.stderr)
    raise


ITER_DIR_RE = re.compile(r"^iter_(\d{7})$")
LATEST_FILE = "latest_checkpointed_iteration.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental uploader to Hugging Face Hub for Megatron checkpoints")
    p.add_argument("--ckpt-dir", required=True, type=str, help="Path to checkpoint directory (contains iter_XXXXXXX folders)")
    p.add_argument("--repo-id", required=True, type=str, help="Hugging Face repo id, e.g. 'org/name'")
    p.add_argument("--revision", default="main", type=str, help="Branch or revision to push to (default: main)")
    p.add_argument("--poll-interval", default=60, type=int, help="Seconds between checks (default: 60)")
    p.add_argument("--state-file", default=None, type=str, help="Path to state file (default: <ckpt-dir>/.hf_uploader_state.json)")
    p.add_argument("--from-iteration", default=None, type=int, help="If no state is present, start uploading from this iteration (inclusive).")
    p.add_argument("--once", action="store_true", help="Upload the current latest iteration and exit")
    p.add_argument("--no-create", action="store_true", help="Do not attempt to create the repo if it does not exist")
    p.add_argument("--max-retries", default=5, type=int, help="Max retries per upload (default: 5)")
    p.add_argument("--retry-wait", default=15, type=int, help="Seconds to wait between retries (default: 15)")
    p.add_argument("--no-include-latest-file", action="store_true", help=f"Do not include {LATEST_FILE} in commits")
    p.add_argument("--dry-run", action="store_true", help="Do not upload; just print what would happen")
    return p.parse_args()


def load_state(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_state(path: Path, state: dict) -> None:
    try:
        path.write_text(json.dumps(state, indent=2, sort_keys=True))
    except Exception as e:
        print(f"WARN: failed to write state file {path}: {e}")


def get_iter_from_latest_file(ckpt_dir: Path) -> Optional[int]:
    latest_path = ckpt_dir / LATEST_FILE
    if not latest_path.is_file():
        return None
    try:
        content = latest_path.read_text().strip()
        # Megatron writes numeric iteration (e.g., 5000)
        it = int(content)
        return it
    except Exception:
        # fallback: try to parse iter_XXXXXXX
        m = re.match(r"^iter_(\d+)$", content)
        if m:
            return int(m.group(1))
    return None


def list_iter_dirs(ckpt_dir: Path) -> List[int]:
    iters = []
    for p in ckpt_dir.iterdir():
        if not p.is_dir():
            continue
        m = ITER_DIR_RE.match(p.name)
        if m:
            try:
                iters.append(int(m.group(1)))
            except Exception:
                pass
    return sorted(iters)


def format_iter_dir(iteration: int) -> str:
    return f"iter_{iteration:07d}"


def ensure_repo(repo_id: str, revision: str, allow_create: bool = True) -> None:
    api = HfApi()
    try:
        _ = api.repo_info(repo_id, revision=revision)
        return
    except Exception:
        if not allow_create:
            raise
        print(f"Repo {repo_id}@{revision} not found. Creating (or ensuring exists, public)...")
        # Default to a public model repo as requested.
        create_repo(repo_id, repo_type="model", exist_ok=True, private=False)


def upload_one(
    ckpt_dir: Path,
    repo_id: str,
    revision: str,
    iteration: int,
    include_latest_file: bool,
    max_retries: int,
    retry_wait: int,
    dry_run: bool = False,
) -> None:
    iter_dir = format_iter_dir(iteration)
    allow = [f"{iter_dir}/**"]
    if include_latest_file:
        allow.append(LATEST_FILE)
    msg = f"Add {iter_dir}"
    print(f"Uploading {iter_dir} to {repo_id}@{revision} (allow_patterns={allow}) ...")
    if dry_run:
        print("[dry-run] Skipping upload")
        return
    attempt = 0
    last_err: Optional[Exception] = None
    while attempt <= max_retries:
        try:
            snapshot_upload(
                local_dir=str(ckpt_dir),
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=allow,
                revision=revision,
                commit_message=msg,
            )
            print(f"Uploaded {iter_dir} ✓")
            return
        except Exception as e:
            last_err = e
            attempt += 1
            if attempt > max_retries:
                break
            print(f"Upload failed ({e}); retrying in {retry_wait}s ({attempt}/{max_retries})...")
            time.sleep(retry_wait)
    raise RuntimeError(f"Failed to upload {iter_dir} after {max_retries} retries: {last_err}")


def main() -> int:
    args = parse_args()

    ckpt_dir = Path(args.ckpt_dir).absolute()
    if not ckpt_dir.exists():
        print(f"ERROR: ckpt-dir does not exist: {ckpt_dir}", file=sys.stderr)
        return 2

    state_path = Path(args.state_file) if args.state_file else (ckpt_dir / ".hf_uploader_state.json")
    state = load_state(state_path)
    last_uploaded = int(state.get("last_uploaded_iteration", -1))

    # If first run with no state, optionally set from --from-iteration
    if last_uploaded < 0 and args.from_iteration is not None:
        last_uploaded = int(args.from_iteration) - 1

    # Ensure repo exists (if allowed)
    try:
        ensure_repo(args.repo_id, args.revision, allow_create=(not args.no_create))
    except Exception as e:
        print(f"ERROR: cannot access or create repo {args.repo_id}@{args.revision}: {e}", file=sys.stderr)
        return 3

    def find_pending() -> List[int]:
        # Prefer latest file as source-of-truth to avoid racing partial saves
        latest = get_iter_from_latest_file(ckpt_dir)
        candidates: List[int] = []
        if latest is not None and latest > last_uploaded:
            candidates.append(latest)
        elif latest is None:
            # fallback: list dirs and pick those newer than last_uploaded
            for it in list_iter_dirs(ckpt_dir):
                if it > last_uploaded:
                    candidates.append(it)
        return sorted(set(candidates))

    def loop_once() -> bool:
        nonlocal last_uploaded, state
        pending = find_pending()
        if not pending:
            return False
        for it in pending:
            iter_dir = ckpt_dir / format_iter_dir(it)
            if not iter_dir.exists():
                # race: latest points to a dir that hasn't materialized yet
                print(f"Pending {iter_dir} not found yet; will retry")
                return False
            upload_one(
                ckpt_dir=ckpt_dir,
                repo_id=args.repo_id,
                revision=args.revision,
                iteration=it,
                include_latest_file=(not args.no_include_latest_file),
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
                dry_run=args.dry_run,
            )
            last_uploaded = it
            state["last_uploaded_iteration"] = last_uploaded
            save_state(state_path, state)
        return True

    if args.once:
        _ = loop_once()
        return 0

    print(f"Starting HF uploader: ckpt_dir={ckpt_dir}, repo={args.repo_id}@{args.revision}")
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        print("TIP: set HF_HUB_ENABLE_HF_TRANSFER=1 for faster uploads")
    try:
        while True:
            changed = loop_once()
            time.sleep(args.poll_interval if not changed else max(1, args.poll_interval // 4))
    except KeyboardInterrupt:
        print("Interrupted. Exiting.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
