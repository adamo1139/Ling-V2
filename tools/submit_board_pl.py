#!/usr/bin/env python3
"""Put an external HF model's Polish results on the Fabryka model board (PL category).

Takes the JSON output of eval_multiblimp_pl.py and, optionally, an ARC-Easy-PL result: either
the output of Probierz benchmarks/zewnetrzne/arc-easy-pl/arc_easy_pl.py (the protocol behind
axis K of eff-PL; its EU20 set is logged, ddudek is a control and is not) or of
eval_arc_easy_pl.py. It checks that the results cover the full datasets and the weights of
`--hf-repo` at `--revision` (the MultiBLiMP weights sha256 against the Hub's LFS sha256; the
Probierz file has no weights hash, so its revision and parameter count are compared instead),
then creates one SDK container run that logs `board_pl/multiblimp` and `board_pl/arc_easy` at
step 0, opts in to public board metrics and sets `leaderboard/result` (fabryka-track
docs/model-board.md, "External model measured here"). Without `board_pl/byte_ppl` and
`board_pl/eff` the row is listed below the ranked PL rows, sorted by MultiBLiMP-pl.

  FABRYKA_API_KEY=... python3 Ling-V2/tools/submit_board_pl.py \
      --multiblimp multiblimp_pl_poziomka_8_plus_9_linear_merge_fp32.json \
      --arc-easy arc_easy_pl_probierz_merge.json \
      --model-name "Poziomka 8+9 linear merge" --author cpral \
      --hf-repo cpral/poziomka_8_plus_9_linear_merge --revision ec9534e1ed3d8b056431cddf3b742b65e7c99f94

--dry-run checks everything and prints what would be sent without contacting the track API.
"""
import argparse
import hashlib
import json
import os

from huggingface_hub import HfApi

ARC_EU20 = ("Eurolingua/arcx/arc_easy_PL_test.jsonl@f2e35fbebdb572a9653eda706b00083b0892ed43",
            "c6732eeba3b2f3cfda95d65d42fb9d4e0f10edc1b9e6c8582d46c069b6706186")
ARC_QUESTIONS = 2376
PROBIERZ = "Probierz arc_easy_pl 7f9a267b"


def read(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def load_multiblimp(path):
    result = read(path)
    if result.get("task") != "multiblimp_polish":
        raise SystemExit(f"{path}: not an eval_multiblimp_pl.py result")
    if not result.get("full_dataset"):
        raise SystemExit(f"{path}: measured with --limit, not the full dataset")
    return result


def load_arc(path):
    """(accuracy %, model revision or None, parameter count or None, weights sha256 or None, harness)."""
    result = read(path)
    if result.get("task") == "arc_easy_polish":
        if not result.get("full_dataset"):
            raise SystemExit(f"{path}: measured with --limit, not the full dataset")
        return (result["accuracy_percent"], result.get("model_revision"), None,
                result["weights_sha256"], "Ling-V2 eval_arc_easy_pl")
    if result.get("benchmark") != "ARC-Easy-PL":
        raise SystemExit(f"{path}: neither an arc_easy_pl.py nor an eval_arc_easy_pl.py result")
    eu20 = result.get("zestawy", {}).get("EU20")
    if not eu20:
        raise SystemExit(f"{path}: no EU20 set (run arc_easy_pl.py without --zestawy ddudek)")
    if (eu20["dataset"], eu20["sha256"]) != ARC_EU20 or len(eu20["items"]) != ARC_QUESTIONS:
        raise SystemExit(f"{path}: EU20 is not the pinned {ARC_QUESTIONS}-question Eurolingua/arcx test")
    accuracy = round(100 * sum(eu20["items"].values()) / ARC_QUESTIONS, 4)
    return accuracy, result.get("revision"), result.get("params"), None, PROBIERZ


def hub_weights(repo, revision):
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise SystemExit(f"--revision must be the full commit sha; {repo}@{revision} is {info.sha}")
    weights = {s.rfilename: s.lfs.sha256 for s in info.siblings
               if s.rfilename.endswith(".safetensors") and s.lfs}
    total = info.safetensors.total if info.safetensors else None
    return weights, total


def checkpoint_sha256(weights):
    """The file's sha256, or for sharded weights the sha256 of their `sha256sum` manifest
    (lines `<sha256>  <file>`, sorted by file name), so `sha256sum *.safetensors | sha256sum`
    reproduces it."""
    if len(weights) == 1:
        return next(iter(weights.values()))
    manifest = "".join(f"{weights[name]}  {name}\n" for name in sorted(weights))
    return hashlib.sha256(manifest.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multiblimp", required=True, help="eval_multiblimp_pl.py output")
    parser.add_argument("--arc-easy", help="arc_easy_pl.py (Probierz) or eval_arc_easy_pl.py output")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--revision", required=True, help="full HF commit of the evaluated weights")
    parser.add_argument("--n-params", type=int,
                        help="default: the parameter count from the arc_easy_pl.py result, else the Hub's")
    parser.add_argument("--harness-sha", help="commit of the evaluation harness, if one applies")
    parser.add_argument("--trust", choices=("reported", "measured"), default="reported")
    parser.add_argument("--project", default="model-board-pl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hub, hub_params = hub_weights(args.hf_repo, args.revision)

    multiblimp = load_multiblimp(args.multiblimp)
    measured = {os.path.basename(f): digest for f, digest in multiblimp["weights_sha256"].items()}
    if measured != {os.path.basename(f): digest for f, digest in hub.items()}:
        raise SystemExit(f"MultiBLiMP weights {measured} are not {args.hf_repo}@{args.revision}: {hub}")
    if multiblimp.get("model_revision") not in (None, args.revision):
        raise SystemExit(f"MultiBLiMP was measured at revision {multiblimp['model_revision']}")
    if len(hub) > 1:
        print(f"note: {len(hub)} weights files; checkpoint_sha256 is the sha256 of their sha256sum manifest")
    metrics = {"board_pl/multiblimp": multiblimp["accuracy_percent"]}
    n_params = args.n_params
    dtype = multiblimp.get("dtype", "").replace("torch.", "")
    harness = f"MultiBLiMP Ling-V2 {dtype}"

    if args.arc_easy:
        accuracy, revision, params, weights, arc_harness = load_arc(args.arc_easy)
        if revision not in (None, args.revision):
            raise SystemExit(f"ARC-Easy-PL was measured at revision {revision}, not {args.revision}")
        if params is not None and hub_params:
            # The Hub counts every stored tensor, including persistent buffers that are not
            # parameters (e.g. the MoE router's expert_bias), so it may exceed the model's count.
            extra = hub_params - params
            if not 0 <= extra <= hub_params * 1e-4:
                raise SystemExit(f"ARC-Easy-PL model has {params} parameters, the Hub reports {hub_params}")
            if extra:
                print(f"note: the Hub stores {extra} more values than the model's parameters "
                      "(buffers); n_params uses the model's count")
        n_params = n_params or params
        if weights is not None and weights != multiblimp["weights_sha256"]:
            raise SystemExit("ARC-Easy-PL and MultiBLiMP were measured on different weights")
        if revision is None and weights is None:
            print("note: the ARC-Easy-PL result records neither a model revision nor a weights hash; "
                  "only its parameter count was checked")
        metrics["board_pl/arc_easy"] = accuracy
        harness += f"; ARC-PL {arc_harness}"

    n_params = n_params or hub_params
    if not n_params:
        raise SystemExit("No parameter count; pass --n-params")
    leaderboard = {
        "step": 0,
        "checkpoint_sha256": checkpoint_sha256(measured),
        "n_params": n_params,
        "kind": "external",
        "trust": args.trust,
        "model_name": args.model_name,
        "author": args.author,
        "hf_repo": args.hf_repo,
        "revision": args.revision,
        "harness": harness,
        "label": "final checkpoint (result)",
    }
    if args.harness_sha:
        leaderboard["harness_sha"] = args.harness_sha
    for field in ("harness", "model_name", "author"):
        if len(leaderboard[field]) > 80 or any(c in leaderboard[field] for c in "<>"):
            raise SystemExit(f"{field} must be plain text of at most 80 characters: {leaderboard[field]!r}")
    name = f"external: {args.model_name} ({args.author})"
    print(json.dumps({"name": name, "metrics": metrics, "leaderboard/result": leaderboard},
                     indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    if not os.getenv("FABRYKA_API_KEY"):
        raise SystemExit("Set FABRYKA_API_KEY")

    from fabryka import RunClient
    client = RunClient()
    client.init(project=args.project, name=name, config={"parameters": n_params})
    client.log(metrics, step=0)
    client["visibility/public_board_metrics"] = True
    client["visibility/public_note"] = "External model, measured by the submitter; not trained here."
    client["leaderboard/result"] = leaderboard
    run_id = client.run_id
    client.finish(timeout=30)
    print(f"run {run_id} finished: {client.api_url}/run/{run_id}")


if __name__ == "__main__":
    main()
