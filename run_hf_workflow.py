#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end workflow:

  1. Download three Hugging Face dataset repos (network required; optional HF_TOKEN for rate limits).
  2. Locate the merged wealth JSON locally by default: final_dataset_clean_merged_input.json
  3. Optionally call deepseek_think_eval.py in the same directory to run the reasoning-trace
     pipeline on Amazon Bedrock.

Dependencies:
  pip install huggingface_hub boto3

Examples:
  # Download only into ./datasets under this script
  python run_hf_workflow.py

  # Download and run Bedrock pipeline (AWS credentials required; see Bedrock docs)
  python run_hf_workflow.py --run-deepseek --limit 100 --workers 10 --region us-east-1

  # Skip download; run on an existing file
  python run_hf_workflow.py --skip-download --run-deepseek --wealth-json path/to/file.jsonl

Environment:
  HF_TOKEN              optional Hugging Face token
  AWS_REGION / AWS credentials  same credential chain as AWS CLI when using --run-deepseek
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Hugging Face repos (edit if your namespace changes)
# ---------------------------------------------------------------------------

HF_REPOS: Tuple[Tuple[str, str], ...] = (
    ("HaolinRPI/wealth-alpaca-lora-final-dataset-clean", "wealth-alpaca-lora-final-dataset-clean"),
    ("HaolinRPI/codealpaca-20k-final", "codealpaca-20k-final"),
    ("HaolinRPI/chatdoctor-healthcaremagic-merged-input", "chatdoctor-healthcaremagic-merged-input"),
)

# Preferred filenames under the wealth snapshot (first match wins)
WEALTH_MERGED_JSON_NAMES: Tuple[str, ...] = (
    "final_dataset_clean_merged_input.json",
    "final_dataset_clean.json",
)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_data_root() -> Path:
    return script_dir() / "datasets"


def default_output_root() -> Path:
    return script_dir() / "outputs"


def download_repo(repo_id: str, local_dir: Path, *, token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token or os.getenv("HF_TOKEN"),
    )


def download_all(data_root: Path, *, token: Optional[str] = None) -> None:
    for repo_id, folder_name in HF_REPOS:
        download_repo(repo_id, data_root / folder_name, token=token)


def find_wealth_merged_json(data_root: Path) -> Optional[Path]:
    """Find merged JSON under the wealth dataset folder."""
    wealth_dir = data_root / HF_REPOS[0][1]
    if not wealth_dir.is_dir():
        return None
    for name in WEALTH_MERGED_JSON_NAMES:
        p = wealth_dir / name
        if p.is_file():
            return p
    for name in WEALTH_MERGED_JSON_NAMES:
        found = list(wealth_dir.rglob(name))
        if found:
            return found[0]
    return None


def resolve_deepseek_script() -> Path:
    return script_dir() / "deepseek_think_eval.py"


def run_deepseek_pipeline(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    deepseek_script: Path,
    extra_args: Sequence[str],
) -> int:
    if not deepseek_script.is_file():
        print(f"ERROR: not found: {deepseek_script}", file=sys.stderr)
        print("Place deepseek_think_eval.py next to this script.", file=sys.stderr)
        return 1
    cmd: List[str] = [
        sys.executable,
        str(deepseek_script),
        "--input-jsonl",
        str(input_jsonl),
        "--output-jsonl",
        str(output_jsonl),
    ]
    cmd.extend(extra_args)
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download three HF datasets (HaolinRPI) and optionally run deepseek_think_eval (Bedrock)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root folder for snapshots (default: <script_dir>/datasets)",
    )
    p.add_argument("--skip-download", action="store_true", help="Skip HF download; use existing folders")
    p.add_argument("--hf-token", default=None, help="Hugging Face token (or set HF_TOKEN)")
    p.add_argument(
        "--wealth-json",
        type=Path,
        default=None,
        help="Input JSON/JSONL path for the pipeline (default: auto-detect under wealth folder)",
    )
    p.add_argument(
        "--run-deepseek",
        action="store_true",
        help="After download (unless --skip-download), invoke deepseek_think_eval.py",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Pipeline output JSONL (default: <script_dir>/outputs/wealth_trace_output.jsonl)",
    )
    p.add_argument("--deepseek-script", type=Path, default=None, help="Path to deepseek_think_eval.py")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--think-budget", type=int, default=2048)
    p.add_argument("--answer-max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--model",
        default=os.getenv("BEDROCK_MODEL_ID") or "anthropic.claude-3-5-sonnet-20240620-v1:0",
        help="Bedrock model id",
    )
    p.add_argument("--region", "--aws-region", dest="region", default=None, help="AWS region (default: AWS_REGION)")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="With --run-deepseek: print the deepseek command only; no conversion run, no Bedrock calls",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run and not args.run_deepseek:
        print("ERROR: --dry-run requires --run-deepseek.", file=sys.stderr)
        return 2
    data_root = args.data_root or default_data_root()
    token = args.hf_token or os.getenv("HF_TOKEN")

    if not args.skip_download:
        try:
            download_all(data_root, token=token)
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed: {exc}", file=sys.stderr)
            print("Try HF_TOKEN, verify repo IDs, or check network.", file=sys.stderr)
            return 2
    else:
        print("[skip] skipping download; using:", data_root)

    wealth_path = args.wealth_json
    if wealth_path is None:
        found = find_wealth_merged_json(data_root)
        if found is None:
            print(
                f"Could not find {', '.join(WEALTH_MERGED_JSON_NAMES)} under "
                f"{data_root / HF_REPOS[0][1]}; pass --wealth-json explicitly.",
                file=sys.stderr,
            )
            return 3
        wealth_path = found

    if not wealth_path.is_file():
        print(f"File not found: {wealth_path}", file=sys.stderr)
        return 3

    print("[input] using:", wealth_path.resolve())
    print("[hint] CodeAlpaca / ChatDoctor folders:")
    for _, folder in HF_REPOS[1:]:
        d = data_root / folder
        print(f"       - {d}")

    if not args.run_deepseek:
        print("\nNext (optional): configure AWS credentials and Bedrock access, then:")
        print(f'  python "{Path(__file__).name}" --skip-download --run-deepseek --wealth-json "{wealth_path}"')
        return 0

    out_path = args.output_jsonl or (default_output_root() / "wealth_trace_output.jsonl")
    ds_script = args.deepseek_script or resolve_deepseek_script()

    extra: List[str] = [
        "--workers",
        str(args.workers),
        "--think-budget",
        str(args.think_budget),
        "--answer-max-tokens",
        str(args.answer_max_tokens),
        "--temperature",
        str(args.temperature),
        "--model",
        args.model,
        "--offset",
        str(args.offset),
    ]
    if args.limit is not None:
        extra.extend(["--limit", str(args.limit)])
    if args.region:
        extra.extend(["--region", args.region])

    input_for_deepseek = wealth_path
    tmp_ndjson: Optional[Path] = None
    try:
        text_head = wealth_path.read_text(encoding="utf-8")[:512].lstrip()
        if args.dry_run:
            if text_head.startswith("["):
                print(
                    "[dry-run] Input is a JSON array; a full run would write a temporary NDJSON file at "
                    f"{data_root / '.wealth_as_jsonl.tmp.jsonl'} first."
                )
            cmd_preview = [
                sys.executable,
                str(ds_script),
                "--input-jsonl",
                str(wealth_path if not text_head.startswith("[") else data_root / ".wealth_as_jsonl.tmp.jsonl"),
                "--output-jsonl",
                str(out_path),
            ]
            cmd_preview.extend(extra)
            print("[dry-run] Command that would run (for arrays, --input-jsonl is the temp NDJSON path):")
            print("[run]", " ".join(str(x) for x in cmd_preview))
            return 0
        if text_head.startswith("["):
            tmp_ndjson = data_root / ".wealth_as_jsonl.tmp.jsonl"
            print("[convert] JSON array detected; writing temporary NDJSON:", tmp_ndjson)
            _json_array_to_ndjson(wealth_path, tmp_ndjson)
            input_for_deepseek = tmp_ndjson
    except OSError as exc:
        print(f"Failed to read input file: {exc}", file=sys.stderr)
        return 4

    rc = run_deepseek_pipeline(
        input_jsonl=input_for_deepseek,
        output_jsonl=out_path,
        deepseek_script=ds_script,
        extra_args=extra,
    )
    if tmp_ndjson is not None and tmp_ndjson.exists():
        try:
            tmp_ndjson.unlink()
        except OSError:
            pass
    if rc == 0:
        print("[done] output:", out_path.resolve())
    return rc


def _json_array_to_ndjson(src: Path, dst: Path) -> None:
    import json

    raw = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Expected a JSON array at top level")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as f:
        for obj in raw:
            if not isinstance(obj, dict):
                raise ValueError("Array elements must be JSON objects")
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
