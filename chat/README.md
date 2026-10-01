# Czat z Poziomką

Rozmowa z modelem postawionym na SGLangu, z promptem renderowanym lokalnie.

```
python3 chat/chat_poziomka.py
python3 chat/chat_poziomka.py --thinking off
```

## Adres serwera

Skopiuj `.env.example` do `.env` i wpisz swój adres:

```
POZIOMKA_BASE_URL=http://192.168.1.26:2300
```

Skrypt szuka `.env` najpierw obok siebie, potem w korzeniu repo. Prawdziwa
zmienna środowiskowa ma pierwszeństwo nad plikiem, a `--base-url` nad obiema.
`.env` jest w `.gitignore` — adres maszyny z GPU to własność stanowiska, nie
kodu.

## Komendy

```
/new                   Nowa rozmowa, ustawienia bez zmian.
/reroll                Powtórz ostatnią odpowiedź.
/thinking on|off|auto  Rozumowanie dla kolejnej odpowiedzi.
/set NAZWA WARTOŚĆ     Zmień ustawienie generowania.
/settings              Pokaż ustawienia i długość historii.
/multiline             Wpisz wiele linii; wyślij przez /end.
/quit                  Wyjście (albo Ctrl-D).
```

Ctrl-C przerywa wpisywanie albo czekanie na odpowiedź; nieudana tura nie wchodzi
do historii.

## Dlaczego prompt renderujemy lokalnie

Przełącznik `enable_thinking` bywa po stronie serwera zawodny: zdarzało się, że
model rozumował mimo `false` — w jednej z serii Poziomki w 52% tur. Dlatego
skrypt składa prompt sam i dokleja prefiks odpowiedzi:

| `--thinking` | prefiks | efekt |
|---|---|---|
| `on` | `<think>\n` | blok otwarty, model go kontynuuje |
| `off` | `<think>\n</think>\n` | blok domknięty z góry, model odpowiada wprost |
| `auto` | brak | decyduje szablon serwera |

`off` jest więc twardsze niż `chat_template_kwargs`: model nie ma czego
kontynuować. Tej metody używa też harness Łodygi.

## Sampling

Domyślne wartości odpowiadają wierszowi `(16k/640k, t0,6 min_p)` z leaderboardu
Łodygi: `temperature 0,6`, `top_p 0,9`, `min_p 0,05`, `frequency_penalty 0,15`,
`top_k` i `repetition_penalty` wyłączone.

**Ten zestaw jest związany z konfiguracją RoPE i nie przenosi się.** Przy RoPE
84000 kierunek temperatury się odwraca: tam najlepsza dla rozumowania była 0,9
(1 pusta tura na 48 prób), a 0,3 najgorsza (5/48); przy RoPE 640000 jest
odwrotnie — przy 0,9 wychodzi 4–9/48 pustych, a najlepiej wypada 0,5–0,6.

Odrzucone na tej konfiguracji, w kolejności od najgorszego: `repetition_penalty`
z zakresem (11–12/48 pustych), `top_k` 20/40 (8–10/48 i więcej urwanych tur),
samo `min_p` bez obniżenia temperatury (6/48). Nie wracaj do nich bez pomiaru.

Wartości zmienisz w locie przez `/set`, np. `/set temperature 0.9`.

## Tokenizer

Domyślnie `poziomka_sft_2026_09_24_hf/iter_0000100` z korzenia repo — potrzebny
jest wyłącznie do renderowania szablonu, nie do liczenia. Szablony serii
2026-09-14, -21 i -24 są bajtowo identyczne (md5 `c303c886`), więc przy tych
seriach wybór nie ma znaczenia. Inny podasz przez `--tokenizer`.
