#!/usr/bin/env python3
"""Policz TRAIN_ITERS i LR z gotowego cache'u SFT.

To samo, co robi koncowka `build_polskie_sprawy_v3_cache.sh`, ale na istniejacym
cache'u - bez przebudowywania. Przydaje sie, gdy build przeszedl, a weryfikacja
padla, albo gdy chcesz porownac warianty GBS.

LR liczymy regula pierwiastkowa dla Adama, ta sama co w run 7 i run 10:

    LR = 3e-4 * sqrt(tokeny_SFT_na_krok / 2_097_152)

gdzie 2_097_152 = 8192 x GBS 256 to tokeny na krok z pretreningu, a tokeny SFT
na krok liczymy z PRAWDZIWYCH tokenow, nie z wypelnienia okna: pakowanie nigdy
nie jest stuprocentowe (v3 77,4%, v4-nc 92,5%), wiec branie SEQ_LENGTH x GBS
zawyzaloby LR.

    python3 policz_kroki_i_lr.py ~/projects/pretrain/poziomka-sft-cache-polskie-sprawy-v6-v13-16384
"""

import argparse
import json
import math
import sys
from pathlib import Path

PRETRENING_TOKENY_NA_KROK = 8192 * 256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", type=Path, help="Katalog cache'u z manifest.json")
    parser.add_argument("--gbs", type=int, nargs="+", default=[16, 32, 64])
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from poziomka_data import MMapSFTDataset

    manifest = json.load((args.cache / "manifest.json").open())
    tokeny = manifest["totals"]["train"]["tokens"]
    rekordy = manifest["totals"]["train"]["records"]

    train = MMapSFTDataset(str(args.cache / "manifest.json"), "train", pack=True)
    kubelki = len(train.bins)
    na_sekwencje = tokeny / kubelki

    print(f"cache:     {args.cache}")
    print(f"rekordy:   {rekordy:,}")
    print(f"tokeny:    {tokeny:,}")
    print(f"kubelki:   {kubelki:,}  (pakowanie {train.packing_efficiency():.3f})")
    print(f"prawdziwych tokenow na spakowana sekwencje: {na_sekwencje:,.0f}\n")
    print(f"{'GBS':>5} {'TRAIN_ITERS':>12} {'tokeny/krok':>13} {'LR':>10}")
    for gbs in args.gbs:
        krok = na_sekwencje * gbs
        lr = 3e-4 * math.sqrt(krok / PRETRENING_TOKENY_NA_KROK)
        print(f"{gbs:>5} {math.ceil(kubelki / gbs):>12} {krok:>13,.0f} {lr:>10.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
