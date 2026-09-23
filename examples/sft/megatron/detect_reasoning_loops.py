#!/usr/bin/env python3
"""Wykrywanie zdegenerowanych petli w polu reasoning_content korpusu SFT.

TLO
---
Przebieg SFT "run3" (16384 tokenow, polityka long_policy=truncate) wyprodukowal
model, ktory przy dekodowaniu zachlannym wpada w petle i nie konczy odpowiedzi:

    "Zbieraja miod i miod, Zbieraja miod i miod, Zbieraja miod i miod, ..."

Diagnoza wykluczyla pomiarowo packing, tempo uczenia, zmiane bazy RoPE i konwersje
DCP->HF (log treningowy czysty, grad norm 0.14, wagi ewoluuja gladko). Przyczyna
okazala sie byc w danych: slady rozumowania w korpusie same sa zapetlone.

Pomiar na splicie walidacyjnym v11 (13 401 rozmow, 10 307 wiadomosci z
rozumowaniem), progami domyslnymi tego skryptu:

     2.73% wiadomosci z rozumowaniem zawiera petle
     2.10% rozmow trzeba przez to odrzucic
     8.7%  SLOW rozumowania lezy w zapetlonych wiadomosciach
    najgorszy przypadek: slowo "community" powtorzone 2045 razy pod rzad

Udzial SLOW jest tu wazniejszy niz udzial wiadomosci: zapetlone slady sa z
definicji dlugie, wiec 2.73% wiadomosci wnosi 8.7% masy tekstu. Poniewaz
rozumowanie to ~76.7% tokenow korpusu, mowimy o okolo 7% calej masy
treningowej. Model nauczyl sie dokladnie tego, co mu pokazano.

Kalibracja szla w tym kierunku: zgrubny detektor max_gram >= 10 mierzyl
18.78% wiadomosci i 68.6% slow; pierwsza wersja progow tego skryptu
(absolutny max_gram >= 20, bez detektora znakowego) -- 7.65% i 37.2%.
Probki tuz przy progu pokazaly, ze w obu przypadkach duza czesc to dlugie
zdrowe slady, ktore naturalnie powtarzaja frazy techniczne (w tekstach
33-38 tys. slow najczestszy 4-gram powtarza sie 26-45 razy, czyli do 1%
tekstu); stad skalowany dlugoscia prog max_gram (MAX_GRAM_SHARE) i detektor
znakowy z regula litery. Recenzja probek wykryc max_gram o udziale 0.5-9%
tekstu (szablony weryfikacji, transkrypcje siatek, enumeracje slownikowe)
przesunela prog na 0.10 -- powyzej zostaja juz tylko teksty zdominowane
przez kopiowanie jednej frazy.

Dwie konsekwencje, o ktorych warto pamietac przy filtrowaniu:

1. Zapetlony tekst jest trywialnie przewidywalny, wiec strata TRENINGOWA na nim
   spada. Krzywa uczenia wygladala wzorowo przez caly zepsuty przebieg. Metryka
   straty nie wykryje tego problemu -- filtr musi zadzialac na poziomie danych.
2. Split walidacyjny ma to samo skazenie, wiec walidacja tez nic nie pokazala.
   Po odfiltrowaniu treningu trzeba odfiltrowac rowniez walidacje.

CZEGO SZUKAMY
-------------
Kluczowe rozroznienie: dlugi, zdrowy slad rozumowania NATURALNIE powtarza frazy
("Zatem x >= 2", "Sprawdzmy przypadek"). Degeneracja to co innego -- ten sam
fragment wraca JEDEN PO DRUGIM, wielokrotnie. Dlatego glowna miara jest liczba
pelnych powtorzen cyklu, a nie sam udzial powtorzonych n-gramow.

Liczone sa cztery miary, od najwazniejszej:

  cycles       ile razy ten sam fragment powtorzyl sie bezposrednio po sobie.
               Wykrywane dla kazdego okresu p (p = 1 dla "aaa...", p = 3 dla
               "x y z x y z ..."), wynikiem jest najlepszy okres. To jest
               wlasciwy detektor, bo mierzy powtorzenia CIAGLE.
  max_gram     ile razy wystepuje najczestszy 4-gram. Lapie petle rozproszone po
               tekscie, ktorych detektor okresowy nie zlapie.
  rep4         udzial powtorzonych 4-gramow. Najslabsza miara, bo rosnie sama z
               dlugoscia tekstu -- trzymana tylko jako kontekst, NIE jako prog.
  char_cycles  to samo co cycles, ale na znakach zamiast na slowach. Lapie
               petle bez spacji ("jakijakijakijaki", "aaaaaaaa"), ktore
               podzial na slowa zamienia w jedno "slowo" i przez to traci.
               Towarzysza mu char_run (dlugosc ciagu w znakach) i char_period.
               Liczy sie tylko cykl wolny od spacji, na tekscie ze scisnietymi
               bialymi znakami -- patrz PROGI.
  ws_run       najdluzszy ciag bialych znakow pod rzad (spacje, taby, nowe
               linie). Lapie degeneracje typu nieskonczone "\n\n\n..." albo
               "2000 spacji", ktore sciskanie bialych znakow celowo ukrywa
               przed detektorem znakowym. Prog 64: na v11 zdrowy ogon konczy
               sie na 48 (wciecia kodu, wyrownania), degeneracja zaczyna sie
               od setek.

PROGI
-----
Domyslne wartosci sa skalibrowane na rozkladzie ze splitu walidacyjnego v11
(percentyle po wiadomosciach z rozumowaniem):

    max_gram    p50=3   p75=7   p90=15   p95=24   p99=93   p99.9=644
    slowa       p50=439 p75=1990 p90=6505 p95=12310 p99=26374

MIN_CYCLES = 3 oznacza "ten sam fragment powtorzony po sobie czterokrotnie",
czego zdrowy tekst praktycznie nie robi. MAX_GRAM = 20 to podloga dla krotkich
tekstow; dla dlugich prog rosnie z dlugoscia (MAX_GRAM_SHARE), bo absolutny
prog misfire'uje na dlugich sladach: w tekscie 33-38 tys. slow zdrowe frazy
techniczne powtarzaja sie 26-45 razy i absolutne 20 flagowalo je jako petle
(pomiar: 485 z 828 wykryc na v11 to byly dlugie zdrowe slady). Oba progi warto przed masowym
filtrowaniem zweryfikowac na wlasnym splicie: uruchom z --report i obejrzyj
probki tuz powyzej i tuz ponizej progu (--examples oraz --near-threshold).

Detektor znakowy ma wlasne progi i reguly, wykalibrowane pomiarowo na v11:

    MIN_CHAR_CYCLES = 3           pelne powtorzenia cyklu, jak wyzej
    MIN_CHAR_RUN = 12             ciag dla cyklu wieloznakowego; 12 znakow to
                                  dokladnie "jakijakijakijaki" (4x "jaki")
    MIN_CHAR_RUN_SINGLE = 70      ciag dla cyklu jednoznakowego ("aaaa...",
                                  "-----"); zera w liczbach, blanki "____"
                                  i linie separatorow to legalne powtorzenia
                                  pojedynczego znaku i siegaja ~70 znakow
    MIN_WS_RUN = 64               najdluzszy ciag bialych znakow; na v11
                                  zdrowy tekst konczy sie na 48 znakach
                                  (p99=22), degeneracja "\n\n\n..." zaczyna
                                  sie od setek

Dwie reguly wynikaja z pomiarow na v11. Po pierwsze, biale znaki sa sciskane
do pojedynczej spacji przed analiza: wcicie kodu to 20+ identycznych spacji
pod rzad i bez sciskania wciecia kodu dominowaly wykrycia znakowe. Po drugie,
cykl zawierajacy spacje nie liczy sie wcale: "T H T H T H" albo "8, 0, 8, 0"
to transkrypcja danych, nie degeneracja, a powtorki ze spacja sa domena
detektora slowowego. Liczba "wykryte tylko znakowo" w podsumowaniu pokazuje,
ile wiadomosci dolapalo wlasnie dzieki temu detektorowi; jesli probki tuz
przy progu wygladaja zdrowo, podnies MIN_CHAR_RUN. Uwaga na stylistyczne
"hahahahahahaha" (7x "ha", run 12) -- przechodzi, a "hahahahaha" (5x "ha")
juz nie.

UZYCIE
------
    # rozpoznanie: statystyki i przyklady, nic nie zapisuje
    python3 detect_reasoning_loops.py --input v11_validation --split validation

    # pelny raport per wiadomosc do wyboru progow + lista wiadomosci do odrzucenia
    python3 detect_reasoning_loops.py --input sciezka/do/korpusu --split train \\
        --workers 8 --report raport.jsonl --save-ids wiadomosci_z_petla.tsv

# przegladanie probek: ekscerpty wg dowolnego filtra na polach raportu
    python3 detect_reasoning_loops.py --input poziomka-fun-rp-v11 --split validation \\
        --report raport.jsonl --samples 20 \\
        --sample-filter "decided_by == 'chars' and char_run <= 20"

    Jako biblioteka -- to jest funkcja, ktorej powinien uzywac filtr korpusu:

    from detect_reasoning_loops import detect_loop
    verdict = detect_loop(message["reasoning_content"])
    if verdict.is_loop:
        ...odrzuc rekord...
"""
import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

# Progi domyslne; uzasadnienie w docstringu powyzej.
# MIN_CYCLES: ile pelnych powtorzen cyklu uznajemy za degeneracje. 3 oznacza, ze
# ten sam fragment wystapil po sobie czterokrotnie -- zdrowy tekst tego nie robi.
MIN_CYCLES = 3
# Zabezpieczenie przed krotkim cyklem o malym okresie: "tak, tak, tak, tak" ma
# cycles=3 przy period=1, ale to tylko 4 slowa. Wymagamy tez minimalnej dlugosci.
MIN_RUN_WORDS = 8
MAX_GRAM = 20
# Do jakiego okresu szukamy powtorzen. Petle modeli jezykowych maja zwykle okres
# 1-20 slow (slowo, frazа, zdanie). 50 daje zapas przy akceptowalnym koszcie.
MAX_PERIOD = 50
# Ponizej tylu slow nie ma sensu mowic o petli.
MIN_WORDS = 16

# Detektor znakowy: te same miary co wyzej, ale na znakach zamiast na slowach.
# Lapie petle bez spacji ("jakijakijakijaki", "aaaaaaaa"), ktore podzial na
# slowa zamienia w pojedyncze "slowa" i przez to traci z oczu. Analiza idzie
# po tekscie ze scisnietymi bialymi znakami i liczy tylko cykle wolne od
# spacji -- uzasadnienie obu regul w PROGI ponizej.
MIN_CHAR_CYCLES = 3
# Najkrotszy ciag znakow dla cyklu wieloznakowego. 12 znakow to dokladnie
# jedno "jakijakijakijaki" (4x "jaki").
MIN_CHAR_RUN = 12
# Wyjatek dla okresu 1 (jeden znak wielokrotnie). Pojedyncze znaki powtarzaja
# sie legalnie: zera w liczbach ("0.00000000000001"), blanki "____", linie
# separatorow "-----" (takze w kodzie, np. komentarze ABAP). Petla
# jednoznakowa w zdrowym tekscie tez prawie nie wystepuje, wiec tu wymagamy
# 70 znakow ciagu (71 identycznych znakow pod rzad).
MIN_CHAR_RUN_SINGLE = 70
# Do jakiego okresu szukamy powtorzen znakowych. Cykl dluzszy niz ~16 znakow
# bez spacji to juz powtorzone "slowo" i widzi je detektor slowowy (okres 1).
# Krotszy limit tnie tez koszt liczenia.
MAX_CHAR_PERIOD = 16
# Ponizej tylu znakow (po scisnieciu bialych znakow) petla znakowa sie nie miesci.
MIN_CHARS = 16

# Najdluzszy ciag samych bialych znakow (spacje, taby, nowe linie -- tez moga
# tworzyac petle, np. nieskonczone "\n\n\n..."). Zdrowy tekst ma tu mniej niz
# 50: najglebsze wciecia kodu i wyrownania tabel na v11 koncza sie na 48
# (p99=22). Degeneracja typu "model wypuscil 2000 spacji" zaczyna sie tam,
# gdzie zdrowia juz nie widziano.
MIN_WS_RUN = 64

# Prog max_gram skaluje sie z dlugoscia tekstu: flagujemy, gdy najczestszy
# 4-gram zajmuje co najmniej tyle tekstu (max_gram >= MAX_GRAM_SHARE * slow).
# Domyslnie 0.10 = 10% tekstu; absolutny MAX_GRAM=20 zostaje jako podloga
# dla krotkich tekstow. Kalibracja na v11, probka po probce: zdrowe slady
# 33-38 tys. slow mialy najczestszy 4-gram powtorzony 26-45 razy (do 1%
# tekstu), a wykrycia z udzialem 0.5-9% (szablony weryfikacji, transkrypcje
# siatek, enumeracje slownikowe) po recenzji okazaly sie zdrowe; degeneracja
# rozproszona zaczyna sie, gdy powtorzona fraza zajmuje okolo 10% tekstu.
MAX_GRAM_SHARE = 0.10


@dataclass
class LoopVerdict:
    """Wynik analizy jednego tekstu."""
    is_loop: bool
    decided_by: str      # ktora miara zadecydowala: "cycles"|"chars"|"max_gram"|"ws"|""
    cycles: int          # ile razy cykl powtorzyl sie po sobie; glowna miara
    run_length: int      # dlugosc ciagu dopasowan w slowach
    run_period: int      # okres cyklu (1 = pojedyncze slowo w kolko)
    run_start: int       # indeks slowa, od ktorego zaczyna sie ciag
    max_gram: int        # licznosc najczestszego 4-gramu
    rep4: float          # udzial powtorzonych 4-gramow
    char_cycles: int     # jak cycles, ale na znakach; lapie petle bez spacji
    char_run: int        # dlugosc ciagu znakowego w znakach
    char_period: int     # okres cyklu znakowego
    char_start: int      # indeks znaku, od ktorego zaczyna sie ciag; -1 gdy brak
    ws_run: int          # najdluzszy ciag bialych znakow (spacje, taby, \n)
    ws_start: int        # indeks znaku, od ktorego zaczyna sie ciag; -1 gdy brak
    words: int
    reason: str          # czytelny opis decyzji; "" gdy czysto

    def excerpt(self, text, span=160):
        """Fragment tekstu wokol wykrytej petli -- do ogladania przy kalibracji."""
        if self.decided_by == "chars":
            # char_start indeksuje tekst ze scisnietymi bialymi znakami
            compact = " ".join(text.split())
            begin = max(0, self.char_start - 40)
            return compact[begin:begin + span]
        if self.decided_by == "ws":
            # biale znaki sa niewidzialne, wiec je wizualizujemy
            begin = max(0, self.ws_start - 30)
            fragment = text[begin:self.ws_start + self.ws_run + 30]
            return (fragment.replace(" ", "␣").replace("\t", "⇥")
                    .replace("\n", "⏎").replace("\r", "↵"))
        words = text.split()
        begin = max(0, self.run_start - 8)
        return " ".join(words[begin:begin + span])


def longest_periodic_run(tokens, max_period=MAX_PERIOD):
    """Najdluzszy ciag, w ktorym tokens[i] == tokens[i - p] dla stalego p.

    Dziala na dowolnej sekwencji: na liscie slow (detektor slowowy) i na napisie,
    czyli na pojedynczych znakach (detektor znakowy).

    Dla kazdego okresu p przechodzimy tekst raz i liczymy najdluzszy NIEPRZERWANY
    odcinek dopasowan tokens[i] == tokens[i-p]. Taki odcinek to powtarzajacy sie
    cykl o dlugosci p.

    Wynikiem jest liczba DOPASOWAN (run), nie dlugosc odcinka z doliczonym okresem.
    To rozroznienie jest istotne i bylo zrodlem bledu w pierwszej wersji: przy
    doliczaniu okresu pojedyncze przypadkowe powtorzenie slowa na dystansie 50
    dawalo wynik 51 i przechodzilo kazdy sensowny prog. Efekt: 99.6% korpusu
    oznaczone jako petla. Sam run takiej patologii nie ma -- przypadkowe
    powtorzenie daje run = 1.

    Miara jakosci petli to liczba PELNYCH cykli: cycles = run // period.

        "community community community"  -> period=1, run=2,  cycles=2
        "x y z x y z x y z"              -> period=3, run=6,  cycles=2
        zdanie powtorzone 70 razy        -> period=35, run=2415, cycles=69
        przypadkowe powtorzenie slowa    -> period=p, run=1,  cycles=0

    Zwraca (run, period, start_index) dla okresu o najwiekszej liczbie cykli.
    Przy remisie wygrywa dluzszy run.

    Koszt: O(n * max_period). Dla najdluzszych sladow (~35 tys. slow) i
    max_period=50 to okolo 1.7 mln porownan na wiadomosc. Detektor znakowy
    doklada analogiczny koszt dla n = liczba znakow i max_period = 16.
    """
    best_run, best_period, best_start, best_cycles = 0, 0, 0, 0
    total = len(tokens)
    for period in range(1, min(max_period, total // 2) + 1):
        run = 0
        for i in range(period, total):
            if tokens[i] == tokens[i - period]:
                run += 1
                cycles = run // period
                if (cycles, run) > (best_cycles, best_run):
                    best_run, best_period, best_cycles = run, period, cycles
                    best_start = i - run - period + 1
            else:
                run = 0
    return best_run, best_period, best_start


def ngram_stats(words, n=4):
    """Licznosc najczestszego n-gramu oraz udzial powtorzonych n-gramow.

    Lapie degeneracje rozproszona: tekst, ktory wraca do tej samej frazy
    kilkadziesiat razy w roznych miejscach, nie tworzac ciaglego cyklu.
    """
    if len(words) <= n:
        return 0, 0.0
    grams = Counter(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    total = sum(grams.values())
    return max(grams.values()), 1.0 - len(grams) / total


def longest_whitespace_run(text):
    """Najdluzszy nieprzerwany ciag bialych znakow: (dlugosc, indeks startu).

    Napisy, taby i nowe linie licza sie razem -- degeneracja typu "\n\n\n..."
    albo "model wypuscil 2000 spacji" to ciagle powtorzenia bialych znakow,
    nawet jesli naprzemiennie. Koszt O(n), jedna probka tekstu.
    """
    best = cur = best_start = start = 0
    for i, ch in enumerate(text):
        if ch.isspace():
            if not cur:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best, best_start


def detect_loop(text, min_cycles=MIN_CYCLES, min_run_words=MIN_RUN_WORDS,
                max_gram_threshold=MAX_GRAM, max_gram_share=MAX_GRAM_SHARE,
                max_period=MAX_PERIOD,
                min_char_cycles=MIN_CHAR_CYCLES, min_char_run=MIN_CHAR_RUN,
                min_char_run_single=MIN_CHAR_RUN_SINGLE,
                char_max_period=MAX_CHAR_PERIOD, min_ws_run=MIN_WS_RUN):
    """Glowna funkcja: czy ten tekst jest zdegenerowany.

    To jest punkt wejscia dla filtra korpusu. Zwraca LoopVerdict z pelnym zestawem
    miar, zeby mozna bylo przestawic progi bez ponownego liczenia.

    Kolejnosc decyzji: cykle slowowe, potem 4-gram, na koncu petla znakowa.
    Dzieki temu dolaczenie detektora znakowego NIE zmienia werdyktow, ktore
    detektor slowowy juz wydal -- moze tylko dodac nowe.

    Petla znakowa liczona jest na tekscie ze scisnietymi bialymi znakami
    (" ".join(text.split())) i liczy sie tylko cykl wolny od spacji, ktory ma
    co najmniej min_char_cycles pelnych powtorzen oraz min_char_run znakow
    (dla okresu 1: min_char_run_single). Wciecia kodu, zera w liczbach
    i separatory "-----" to legalne powtorzenia znakow i musza przechodzi
    czysto -- bez tych regul pomiar na v11 dawal zdecydowana wiekszosc
    falszywych alarmow.
    """
    words = text.split() if text else []
    if len(words) < MIN_WORDS and len(text or "") < MIN_CHARS:
        return LoopVerdict(False, "", 0, 0, 0, 0, 0, 0.0, 0, 0, 0, -1, 0, -1,
                           len(words), "")

    if len(words) >= MIN_WORDS:
        run_length, run_period, run_start = longest_periodic_run(words, max_period)
        cycles = run_length // run_period if run_period else 0
        max_gram, rep4 = ngram_stats(words)
    else:
        run_length = run_period = run_start = cycles = 0
        max_gram, rep4 = 0, 0.0

    char_run = char_period = char_cycles = 0
    char_start = -1
    char_unit_clean = False
    compact = " ".join((text or "").split())
    if len(compact) >= MIN_CHARS:
        char_run, char_period, char_start = longest_periodic_run(
            compact, char_max_period)
        char_cycles = char_run // char_period if char_period else 0
        if char_run:
            unit = compact[char_start:char_start + char_period]
            # cykl nie moze zawierac spacji (to transkrypcja danych, nie
            # degeneracja: "T H T H", "8, 0, 8, 0"), a dla okresow > 1 musi
            # tez miec przynajmniej jedna litere -- bez tego lapie powtorzone
            # cyfry z interpunkcja: "[1,1,1,1,1]", "13.13.13.13."
            char_unit_clean = (" " not in unit
                               and (char_period == 1
                                    or any(ch.isalpha() for ch in unit)))
        else:
            char_start = -1

    ws_run, ws_start = longest_whitespace_run(text or "")
    if not ws_run:
        ws_start = -1

    gram_floor = max(max_gram_threshold, max_gram_share * len(words))
    reason = decided_by = ""
    if cycles >= min_cycles and run_length >= min_run_words:
        decided_by = "cycles"
        reason = (f"cykl powtorzony {cycles}x "
                  f"(okres {run_period} slow, ciag {run_length} slow)")
    elif max_gram >= gram_floor:
        decided_by = "max_gram"
        reason = (f"powtorzony 4-gram {max_gram} razy "
                  f"(prog {gram_floor:.0f} przy {len(words)} slowach)")
    elif (char_unit_clean and char_cycles >= min_char_cycles
          and char_run >= (min_char_run_single if char_period == 1
                           else min_char_run)):
        decided_by = "chars"
        reason = (f"petla znakowa: cykl powtorzony {char_cycles}x "
                  f"(okres {char_period} znakow, ciag {char_run} znakow)")
    elif ws_run >= min_ws_run:
        decided_by = "ws"
        reason = f"ciag {ws_run} bialych znakow pod rzad"
    return LoopVerdict(bool(reason), decided_by, cycles, run_length, run_period,
                       run_start, max_gram, rep4, char_cycles, char_run,
                       char_period, char_start, ws_run, ws_start, len(words),
                       reason)


def iter_records(root, split):
    """Czyta parquet albo jsonl, z podkatalogu splitu lub wprost z katalogu."""
    root = Path(root)
    candidates = [root / split, root]
    files = []
    for directory in candidates:
        files = sorted(directory.glob("*.parquet")) or sorted(directory.glob("*.jsonl"))
        if files:
            break
    if not files:
        raise SystemExit(f"Nie znaleziono plikow parquet/jsonl w {root} ani {root / split}")
    for file in files:
        if file.suffix == ".parquet":
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
                yield from batch.to_pylist()
        else:
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def analyse_record(record, thresholds=None):
    """Jeden rekord -> lista wynikow dla kazdej wiadomosci z rozumowaniem.

    Osobna funkcja na poziomie modulu, zeby dala sie uzyc w multiprocessing.Pool
    (przez functools.partial, ktory jest picklowalny).
    """
    thresholds = thresholds or {}
    record_id = record.get("record_id")
    results = []
    for index, message in enumerate(record.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        text = message.get("reasoning_content")
        if not text:
            continue
        verdict = detect_loop(text, **thresholds)
        results.append((record_id, index, verdict,
                        verdict.excerpt(text) if verdict.is_loop else ""))
    return results


def percentiles(values, points=(50, 75, 90, 95, 99, 99.9)):
    if not values:
        return {}
    ordered = sorted(values)
    return {p: ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))] for p in points}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Katalog z parquet/jsonl")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=0, help="0 = caly split")
    parser.add_argument("--min-cycles", type=int, default=MIN_CYCLES)
    parser.add_argument("--max-gram", type=int, default=MAX_GRAM)
    parser.add_argument("--max-gram-share", type=float, default=MAX_GRAM_SHARE,
                        help="Prog max_gram skalowany dlugoscia: flaguj, gdy "
                             "najczestszy 4-gram zajmuje tyle tekstu "
                             "(0 = wylacza skalowanie, sam absolutny prog)")
    parser.add_argument("--min-char-cycles", type=int, default=MIN_CHAR_CYCLES,
                        help="Detektor znakowy: minimalna liczba pelnych cykli")
    parser.add_argument("--min-char-run", type=int, default=MIN_CHAR_RUN,
                        help="Detektor znakowy: minimalna dlugosc ciagu w znakach")
    parser.add_argument("--min-char-run-single", type=int, default=MIN_CHAR_RUN_SINGLE,
                        help="Detektor znakowy: jak wyzej, dla okresu 1 (jeden znak)")
    parser.add_argument("--min-ws-run", type=int, default=MIN_WS_RUN,
                        help="Najdluzszy ciag bialych znakow uznany za petle")
    parser.add_argument("--workers", type=int, default=1,
                        help="Procesy robocze; przy pelnym splicie treningowym ustaw 8+")
    parser.add_argument("--examples", type=int, default=3,
                        help="Ile najgorszych przypadkow wypisac")
    parser.add_argument("--near-threshold", type=int, default=0,
                        help="Ile przypadkow tuz przy progu wypisac (do kalibracji)")
    parser.add_argument("--report", type=Path,
                        help="JSONL z miarami i ekscerptem kazdej wiadomosci; pozwala przestawic progi bez liczenia od nowa")
    parser.add_argument("--save-ids", type=Path,
                        help="TSV: record_id i indeks wiadomosci z petla (po jednym na linie)")
    parser.add_argument("--samples", type=int, default=0,
                        help="Ile ekscerptow probek wypisac; filtr --sample-filter, sortowanie --sample-sort")
    parser.add_argument("--sample-filter", default="is_loop",
                        help="Wyrazenie pythonowe na polach raportu, "
                             "np. \"decided_by == 'chars' and char_run <= 20\"")
    parser.add_argument("--sample-sort", default="",
                        help="Pole liczbowe, po ktorym probki sortuja sie malejaco (np. char_run); "
                             "uwaga: sortowanie trzyma wszystkie pasujace wiersze w pamieci")
    args = parser.parse_args()

    from functools import partial
    thresholds = dict(min_cycles=args.min_cycles, max_gram_threshold=args.max_gram,
                      max_gram_share=args.max_gram_share,
                      min_char_cycles=args.min_char_cycles, min_char_run=args.min_char_run,
                      min_char_run_single=args.min_char_run_single,
                      min_ws_run=args.min_ws_run)
    worker = partial(analyse_record, thresholds=thresholds)

    records = iter_records(args.input, args.split)
    if args.limit:
        import itertools
        records = itertools.islice(records, args.limit)

    if args.workers > 1:
        from multiprocessing import Pool
        pool = Pool(args.workers)
        stream = pool.imap_unordered(worker, records, chunksize=16)
    else:
        pool = None
        stream = map(worker, records)

    looping_ids, worst, near = set(), [], []
    looping_messages = set()
    samples, matched = [], 0
    run_lengths, max_grams, word_counts, cycles_all = [], [], [], []
    char_cycles_all, char_runs_all, ws_runs_all = [], [], []
    messages = loops = conversations = char_only = ws_only = 0
    loop_words = total_words = 0
    report = args.report.open("w", encoding="utf-8") if args.report else None

    if args.samples:
        # wczesna walidacja: zle wyrazenie ma walic od razu, nie w polowie splitu
        probe = dict(is_loop=True, decided_by="chars", cycles=1, run_length=1,
                     run_period=1, run_start=0, max_gram=1, rep4=0.0,
                     char_cycles=1, char_run=1, char_period=1, char_start=0,
                     ws_run=1, ws_start=0, words=1, reason="x", record_id="x",
                     message_index=0, excerpt="x")
        try:
            eval(args.sample_filter, {"__builtins__": {}}, dict(probe))
        except Exception as error:
            raise SystemExit(
                f"--sample-filter {args.sample_filter!r} nie da sie wyliczyc: {error}")

    for results in stream:
        conversations += 1
        for record_id, index, verdict, excerpt in results:
            messages += 1
            cycles_all.append(verdict.cycles)
            run_lengths.append(verdict.run_length)
            max_grams.append(verdict.max_gram)
            word_counts.append(verdict.words)
            char_cycles_all.append(verdict.char_cycles)
            char_runs_all.append(verdict.char_run)
            ws_runs_all.append(verdict.ws_run)
            total_words += verdict.words
            if report or args.samples:
                row = asdict(verdict)
                row.update(record_id=record_id, message_index=index, excerpt=excerpt)
                if report:
                    report.write(json.dumps(row, ensure_ascii=False) + "\n")
                try:
                    hit = bool(eval(args.sample_filter, {"__builtins__": {}}, dict(row)))
                except Exception as error:
                    raise SystemExit(
                        f"--sample-filter {args.sample_filter!r} nie da sie wyliczyc "
                        f"na wierszu {record_id}: {error}")
                if hit:
                    matched += 1
                    # bez sortowania zbieramy tylko tyle, ile trzeba wypisac
                    if args.sample_sort or len(samples) < args.samples:
                        samples.append(row)
            if verdict.is_loop:
                loops += 1
                loop_words += verdict.words
                looping_ids.add(record_id)
                looping_messages.add((record_id, index))
                if verdict.decided_by == "chars":
                    char_only += 1
                elif verdict.decided_by == "ws":
                    ws_only += 1
                worst.append((verdict.cycles, verdict.char_cycles, verdict.max_gram,
                              record_id, excerpt))
            elif (verdict.cycles >= max(1, args.min_cycles - 1)
                  or verdict.char_cycles >= max(1, args.min_char_cycles - 1)):
                near.append((verdict.cycles, verdict.run_length, record_id, verdict.reason))
    if pool:
        pool.close()
        pool.join()
    if report:
        report.close()

    print(f"rozmow przejrzanych:        {conversations:,}")
    print(f"wiadomosci z rozumowaniem:  {messages:,}")
    print(f"z petla:                    {loops:,} "
          f"({100 * loops / max(messages, 1):.2f}% wiadomosci)")
    print(f"wykryte tylko znakowo:      {char_only:,} "
          f"({100 * char_only / max(loops, 1):.1f}% petli)")
    print(f"wykryte przez whitespace:   {ws_only:,}")
    print(f"rekordow do odrzucenia:     {len(looping_ids):,} "
          f"({100 * len(looping_ids) / max(conversations, 1):.2f}% rozmow)")
    print(f"slowa rozumowania w petlach:{loop_words:>12,} z {total_words:,} "
          f"({100 * loop_words / max(total_words, 1):.1f}%)")
    print(f"\nprogi: cycles >= {args.min_cycles}, max_gram >= {args.max_gram} "
          f"(lub >= {args.max_gram_share:.0%} tekstu), "
          f"char_cycles >= {args.min_char_cycles}, char_run >= {args.min_char_run}, "
          f"ws_run >= {args.min_ws_run}\n")

    for name, values in (("cycles", cycles_all), ("run_length", run_lengths), ("max_gram", max_grams),
                         ("slowa", word_counts), ("char_cycles", char_cycles_all),
                         ("char_run", char_runs_all), ("ws_run", ws_runs_all)):
        pct = percentiles(values)
        print(f"  {name:<11} " + "   ".join(f"p{p}={pct[p]:.0f}" for p in sorted(pct)))

    if args.samples:
        if args.sample_sort:
            try:
                samples.sort(key=lambda row: -row[args.sample_sort])
            except (KeyError, TypeError, ValueError):
                raise SystemExit(
                    f"--sample-sort: pole {args.sample_sort!r} nie jest liczba we wierszu raportu")
        shown = samples[:args.samples]
        print(f"\nPROBKI | filtr {args.sample_filter!r}: dopasowan {matched:,}, pokazane {len(shown)}")
        for row in shown:
            print("\n" + "-" * 78)
            print(f"PROBKA | {row['record_id']} #{row['message_index']} | "
                  f"decided_by={row['decided_by']!r} | cykli={row['cycles']} "
                  f"znakow={row['char_cycles']} max_gram={row['max_gram']} slow={row['words']}")
            if row["reason"]:
                print(f"powod: {row['reason']}")
            print((row.get("excerpt") or "")[:600].replace("\n", " ⏎ "))

    worst.sort(key=lambda item: (-item[0], -item[1]))
    for cycles, char_cycles, max_gram, record_id, excerpt in worst[:args.examples]:
        print("\n" + "=" * 78)
        print(f"NAJGORSZE | cykli={cycles} znakow={char_cycles} "
              f"max_gram={max_gram} | {record_id}")
        print("=" * 78)
        print(excerpt[:600].replace("\n", " ⏎ "))

    near.sort(key=lambda item: -item[0])
    for cycles, run_length, record_id, reason in near[:args.near_threshold]:
        print(f"PRZY PROGU | cykli={cycles} ciag={run_length} | {record_id} | {reason}")

    if args.save_ids:
        args.save_ids.write_text(
            "".join(f"{rid}\t{idx}\n" for rid, idx in sorted(looping_messages) if rid),
            encoding="utf-8")
        print(f"\n{len(looping_messages):,} wiadomosci (record_id + indeks) zapisane do {args.save_ids}")
    if args.report:
        print(f"raport per wiadomosc: {args.report}")


if __name__ == "__main__":
    main()
