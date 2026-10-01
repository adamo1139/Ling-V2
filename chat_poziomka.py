#!/usr/bin/env python3
"""Rozmowa z Poziomka na SGLangu, z szablonem renderowanym lokalnie.

Uruchomienie: python3 chat_poziomka.py --help
Rozumowanie jest domyslnie wlaczone; wylacza je --thinking off.

Adres serwera bierzemy z pliku .env obok tego skryptu (POZIOMKA_BASE_URL albo
SGLANG_BASE_URL), bo adres maszyny z GPU jest wlasciwoscia stanowiska, a nie
kodu - wpisany na sztywno wymuszalby edycje pliku na kazdej nowej maszynie
i ladowal w commitach. Prawdziwa zmienna srodowiskowa ma pierwszenstwo nad
.env, a --base-url nad obiema.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

DOMYSLNY_BASE_URL = "http://127.0.0.1:30000"
# Tokenizer i szablon czatu. Szablony serii 2026-09-14, -21 i -24 sa bajtowo
# identyczne (md5 c303c886), wiec do samego renderowania promptu wystarczy
# dowolny z nich; bierzemy najnowszy, zeby nie utrwalac martwej sciezki.
TOKENIZER_PATH = (
    Path(__file__).resolve().parent
    / "poziomka_sft_2026_09_24_hf/iter_0000100"
)
# Sampling dobrany pod serwer z oknem 16384 i RoPE 640000 - ten sam zestaw,
# ktorym mierzylismy wiersze "(16k/640k, t0,6 min_p)" na leaderboardzie Lodygi.
#
# NIE PRZENOSI SIE na RoPE 84000: tam kierunek temperatury jest odwrotny.
# Przy 84000 najlepsza dla rozumowania byla 0,9 (1 pusta tura na 48 prob),
# a 0,3 najgorsza (5/48). Przy 640000 jest na odwrot - przy 0,9 wychodzi
# 4-9/48 pustych, a najlepiej wypada 0,5-0,6. RoPE 640000 rozwleka rozumowanie
# (mediana sladu rosnie z 874 do ~2500 znakow) i model nie domyka <think>.
#
# Co sprawdzone i odrzucone na tej konfiguracji (think, 48 prob):
#   top_k 40 / 20          8-10/48 pustych, WIECEJ urwanych tur
#   rep 1,05 range 512     12/48 pustych
#   rep 1,10 range 256     11/48 pustych
#   min_p 0,05 samo         6/48 pustych
#   min_p 0,05 + t0,6       1/48 pustych, zero urwanych  <- wybrane
#
# frequency_penalty zamiast repetition_penalty: skaluje kare liczba wystapien
# tokenu, zamiast karac jednakowo wszystko, co juz padlo. Poprzednia wartosc
# repetition_penalty 1,01 wypadala GORZEJ niz brak kary, wiec jest wylaczona.
TEMPERATURE = 0.6
TOP_P = 0.9
TOP_K = -1                 # -1 wylacza top_k, tak jak w configach pomiarowych
REPETITION_PENALTY = 1.0   # 1,0 to brak kary
FREQUENCY_PENALTY = 0.15
MIN_P = 0.05
SEED = None


HELP = """Commands:
  /new                         Start a new conversation (keep settings).
  /reroll                      Replace the last reply using current settings.
  /thinking on|off|auto        Set the prefix for the next reply.
  /set NAME VALUE              Change a generation setting; see /settings.
  /settings                    Show current settings and history size.
  /multiline                   Enter multiple lines; finish with /end.
  /help                        Show commands.
  /quit                        Exit (or press Ctrl-D).
Ctrl-C cancels input or a pending request; failed turns are not added to history.
"""
THINKING = {"on": True, "off": False, "auto": None}
CONTEXT_BUFFER = 32  # Leave headroom below the server's context boundary.


def wczytaj_dotenv(path=None):
    """Minimalny czytnik .env; prawdziwe zmienne srodowiskowe maja pierwszenstwo.

    Pliku szukamy obok skryptu, a nie w katalogu roboczym, zeby czat dzialal
    tak samo niezaleznie od tego, skad go uruchomiono.
    """
    path = Path(path) if path else Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        klucz, _, wartosc = line.partition("=")
        os.environ.setdefault(klucz.strip(), wartosc.strip().strip("\"'"))


def base_url_z_srodowiska():
    wczytaj_dotenv()
    for nazwa in ("POZIOMKA_BASE_URL", "SGLANG_BASE_URL"):
        wartosc = os.environ.get(nazwa)
        if wartosc:
            return wartosc
    return DOMYSLNY_BASE_URL


def api_request(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def assistant_prefix(thinking):
    if thinking is True:
        return "<think>\n"
    if thinking is False:
        return "<think>\n</think>\n"
    return ""


def split_reasoning(text, thinking_enabled):
    """Separate raw completions, including a think tag supplied in the prompt."""
    body = text.lstrip()
    if body.startswith("<think>"):
        body = body[len("<think>"):]
    elif thinking_enabled is not True:
        return "", text.strip(), False
    reasoning, closing_tag, answer = body.partition("</think>")
    return reasoning.strip(), answer.strip(), not bool(closing_tag)


def validate_setting(name, value):
    if name == "seed":
        return None if value.lower() == "none" else int(value)
    converters = {
        "max_tokens": int, "temperature": float, "top_p": float,
        "top_k": int, "repetition_penalty": float, "frequency_penalty": float,
        "min_p": float,
    }
    if name not in converters:
        raise ValueError("Unknown setting. Use /settings to see the available names.")
    parsed = converters[name](value)
    valid = {
        "max_tokens": lambda v: v > 0,
        "temperature": lambda v: 0 <= v < float("inf"),
        "top_p": lambda v: 0 < v <= 1,
        "top_k": lambda v: v == -1 or v > 0,
        "repetition_penalty": lambda v: 0 < v < float("inf"),
        "frequency_penalty": lambda v: -2 <= v <= 2,
        "min_p": lambda v: 0 <= v < 1,
    }
    if not valid[name](parsed):
        raise ValueError(f"Invalid value for {name}: {value}")
    return parsed


class Chat:
    def __init__(self, tokenizer, model, settings, thinking, context_length, system=""):
        self.tokenizer = tokenizer
        self.model = model
        self.settings = dict(settings)
        self.thinking = thinking
        self.context_length = context_length
        self.system = system
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.system}] if self.system else []

    def prepare(self, user_text):
        pending = self.messages + [{"role": "user", "content": user_text}]
        return self._prepare(pending)

    def prepare_reroll(self):
        if (len(self.messages) < 2 or self.messages[-1]["role"] != "assistant"
                or self.messages[-2]["role"] != "user"):
            raise ValueError("No assistant reply to reroll yet.")
        # Keep the old reply until accept() commits a successful replacement.
        return self._prepare(self.messages[:-1])

    def _prepare(self, pending):
        kwargs = {} if self.thinking is None else {"enable_thinking": self.thinking}
        prompt = self.tokenizer.apply_chat_template(
            pending, tokenize=False, add_generation_prompt=True, **kwargs
        )
        prefix = assistant_prefix(self.thinking)
        if not prompt.endswith("<|im_start|>assistant\n" + prefix):
            raise ValueError("The local template did not render the requested thinking prefix.")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        available = self.context_length - len(ids) - CONTEXT_BUFFER
        if available <= 0:
            raise ValueError("Conversation fills the context window. Use /new to start again.")
        payload = {k: v for k, v in self.settings.items() if v is not None}
        payload.update(
            model=self.model, prompt=ids, stop=["<|im_end|>"],
            max_tokens=min(self.settings["max_tokens"], available),
        )
        return pending, payload

    def accept(self, pending, result):
        choice = result["choices"][0]
        text = choice["text"]
        if not isinstance(text, str):
            raise ValueError("Server returned a non-text completion.")
        # Reasoning is shown for this turn only; history contains final answers.
        _, answer, _ = split_reasoning(text, self.thinking)
        self.messages = pending + [{"role": "assistant", "content": answer}]
        return choice


def show_reply(text, thinking, choice, usage):
    reasoning, answer, opened = split_reasoning(text, thinking)
    print("\nREASONING\n" + "─" * 50)
    print(reasoning or "(No reasoning generated.)")
    if opened:
        print("(Generation ended before the reasoning block closed.)")
    print("\nASSISTANT ANSWER\n" + "─" * 50)
    print(answer or "(No answer generated.)")
    print(
        f"\n[finish: {choice.get('finish_reason', 'unknown')}; "
        f"prompt tokens: {usage.get('prompt_tokens', '?')}; "
        f"generated: {usage.get('completion_tokens', '?')}]\n"
    )


def read_message():
    text = input("You> ")
    if text.strip() == "/multiline":
        print("Enter your message; /end on its own line sends it. Ctrl-C cancels.")
        lines = []
        while True:
            line = input("... ")
            if line.strip() == "/end":
                return "\n".join(lines), False
            lines.append(line)
    return text, text.startswith("/")


def run_chat(chat, request):
    print(HELP)
    while True:
        try:
            text, command = read_message()
            if not text.strip():
                continue
            if command and text.split() == ["/reroll"]:
                pending, payload = chat.prepare_reroll()
            elif command:
                parts = text.split()
                if parts == ["/quit"] or parts == ["/exit"]:
                    break
                if parts == ["/new"]:
                    chat.reset()
                    print("New conversation started.\n")
                elif parts == ["/help"]:
                    print(HELP)
                elif parts == ["/settings"]:
                    print(json.dumps(chat.settings, indent=2))
                    mode = next(k for k, v in THINKING.items() if v is chat.thinking)
                    print(f"thinking: {mode}; context: {chat.context_length}; "
                          f"history messages: {len(chat.messages)}\n")
                elif len(parts) == 2 and parts[0] == "/thinking" and parts[1] in THINKING:
                    chat.thinking = THINKING[parts[1]]
                    print(f"Thinking: {parts[1]}\n")
                elif len(parts) == 3 and parts[0] == "/set":
                    chat.settings[parts[1]] = validate_setting(parts[1], parts[2])
                    print(f"{parts[1]}: {chat.settings[parts[1]]}\n")
                else:
                    print("Unknown command or arguments. Use /help.\n")
                continue
            else:
                pending, payload = chat.prepare(text)
            print(f"Generating… ({len(payload['prompt'])} prompt tokens; "
                  f"up to {payload['max_tokens']} new tokens)", flush=True)
            result = request("/v1/completions", payload)
            choice = chat.accept(pending, result)
            show_reply(choice["text"], chat.thinking, choice, result.get("usage", {}))
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nCancelled.\n")
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}\n")
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            print(f"Error: {exc}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None,
                        help="Domyslnie POZIOMKA_BASE_URL z .env obok skryptu")
    parser.add_argument("--model", help="Hosted model ID; defaults to the first /v1/models entry.")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--context-length", type=int, help="Server context limit; defaults to local config.")
    parser.add_argument("--system", default="", help="Optional system message, retained by /new.")
    parser.add_argument("--thinking", choices=THINKING, default="on",
                        help="Reasoning mode for new replies (default: on); reasoning is not saved in history.")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Optional reply cap; defaults to the full remaining model context.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY,
                        help="1.0 wylacza; patrz komentarz przy stalych")
    parser.add_argument("--frequency-penalty", type=float, default=FREQUENCY_PENALTY,
                        help="Kara rosnaca z liczba wystapien tokenu; 0 wylacza")
    parser.add_argument("--min-p", type=float, default=MIN_P,
                        help="Prog prawdopodobienstwa wzgledem najlepszego tokenu; 0 wylacza")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.base_url = args.base_url or base_url_z_srodowiska()
    try:
        # Readline supplies editing and input recall without persistent chat files.
        try:
            import readline  # noqa: F401
        except ImportError:
            pass
        from transformers import PreTrainedTokenizerFast

        context_length = args.context_length
        if context_length is None:
            config = json.loads((args.tokenizer / "config.json").read_text())
            context_length = int(config["max_position_embeddings"])
        if context_length < 2:
            raise ValueError("Context length must be at least 2.")
        if args.max_tokens is None:
            args.max_tokens = context_length
        settings = {name: validate_setting(name, str(getattr(args, name))) for name in (
            "max_tokens", "temperature", "top_p", "top_k", "repetition_penalty",
            "frequency_penalty", "min_p", "seed"
        )}
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)
        request = lambda path, payload=None: api_request(args.base_url, path, payload)
        model = args.model or request("/v1/models")["data"][0]["id"]
        chat = Chat(tokenizer, model, settings, THINKING[args.thinking], context_length, args.system)
        print(f"SGLang: {args.base_url}\nModel: {model}\nContext limit: {context_length} "
              f"({CONTEXT_BUFFER} tokens reserved)\n")
        run_chat(chat, request)
    except (ImportError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"Could not start chat: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
