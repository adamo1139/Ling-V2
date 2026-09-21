#!/usr/bin/env python3
"""Validation loss with and without packing, same weights, same cache, no training.

If packing is correct these two numbers must be close: identical conversations,
identical weights, only the grouping differs. A materially higher packed loss
means the packed path corrupts the objective -- which forward-only parity on a
dense 2-layer stand-in would not have caught.

Runs the real model through the real training stack, so MoE, the pipeline and
the loss function are all exercised.

  LOAD_CHECKPOINT=~/projects/pretrain/poziomka_run2_v11_8192 \
  SFT_DATA=~/projects/pretrain/poziomka-sft-cache-v11-16384-truncate \
  bash compare_poziomka_packing_loss.sh
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(packing, args):
    """One short run: load weights, evaluate, never step the optimizer."""
    save = Path(args["scratch"]) / f"probe_pack{packing}"
    subprocess.run(["rm", "-rf", str(save)], check=False)
    env = dict(os.environ)
    env.update(
        SFT_DATA=args["data"], LOAD_CHECKPOINT=args["load"], SAVE_CHECKPOINT=str(save),
        SEQ_LENGTH=str(args["seq_length"]), MAX_POSITION_EMBEDDINGS=str(args["seq_length"]),
        ROTARY_BASE=str(args["rotary_base"]), PACKING=str(packing),
        # TRAIN_ITERS=1 with the evaluation done first; LR 0 so no weight can move.
        TRAIN_ITERS="1", GLOBAL_BATCH_SIZE=str(args["batch"]),
        EVAL_ITERS=str(args["eval_iters"]), EVAL_INTERVAL="1",
        SAVE_INTERVAL="1000000", RESUME="0", LR="0", WARMUP_ITERS="0",
        DATALOADER_WORKERS="2",
    )
    log = Path(args["scratch"]) / f"probe_pack{packing}.log"
    with log.open("w") as handle:
        subprocess.run(["bash", str(HERE / "run_poziomka.sh")],
                       env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    text = log.read_text(errors="replace")
    losses = re.findall(r"validation loss at [^|]*\|\s*lm loss value:\s*([0-9.E+-]+)", text)
    packed_line = re.search(r"^SFT: .*$", text, re.M)
    subprocess.run(["rm", "-rf", str(save)], check=False)
    return (float(losses[-1]) if losses else None,
            packed_line.group(0) if packed_line else "(brak linii SFT:)", log)


def main():
    args = dict(
        data=os.environ.get("SFT_DATA", ""),
        load=os.environ.get("LOAD_CHECKPOINT", ""),
        scratch=os.environ.get("SCRATCH", "/media/nvme_2tb/poziomka_packing_probe"),
        seq_length=int(os.environ.get("SEQ_LENGTH", "16384")),
        rotary_base=int(os.environ.get("ROTARY_BASE", "640000")),
        batch=int(os.environ.get("GLOBAL_BATCH_SIZE", "64")),
        eval_iters=int(os.environ.get("EVAL_ITERS", "20")),
    )
    if not args["data"] or not args["load"]:
        sys.exit("Ustaw SFT_DATA i LOAD_CHECKPOINT")
    Path(args["scratch"]).mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(Path(args["data"]) / "manifest.json"))
    print(f"cache: {manifest['seq_length']} tokenow, polityka {manifest['long_policy']}")
    print(f"wagi:  {args['load']}")
    print(f"eval:  {args['eval_iters']} iteracji x batch {args['batch']}, LR=0\n")

    results = {}
    for packing in (0, 1):
        label = "Z PACKINGIEM" if packing else "BEZ PACKINGU"
        print(f"-- {label} ...", flush=True)
        loss, sft_line, log = run(packing, args)
        results[packing] = loss
        print(f"   {sft_line}")
        print(f"   strata walidacyjna: {loss}   ({log})\n", flush=True)

    plain, packed = results[0], results[1]
    if plain is None or packed is None:
        sys.exit("Nie udalo sie odczytac straty z co najmniej jednego przebiegu; sprawdz logi.")
    delta = packed - plain
    print("=" * 62)
    print(f"bez packingu: {plain:.4f}")
    print(f"z packingiem: {packed:.4f}")
    print(f"roznica:      {delta:+.4f}  ({100 * delta / plain:+.1f}%)")
    print("=" * 62)
    if abs(delta) < 0.02:
        print("Zgodne. Packing NIE psuje funkcji celu -- przyczyny szukamy w danych.")
    elif delta > 0:
        print("Wersja spakowana wyraznie gorsza na tych samych danych i wagach.")
        print("To dowodzi bledu w sciezce packingu, nie w danych.")
    else:
        print("Wersja spakowana lepsza -- nieoczekiwane, sprawdz czy obie ladowaly ten sam cache.")


if __name__ == "__main__":
    main()
