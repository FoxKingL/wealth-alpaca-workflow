#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_think_eval.py

Single-sample / JSONL pipeline used by generate_healthcare_trace_10k.py:

Trace generation only: the model is given dataset Input and Output. Output is fixed — generate only an
intermediate reasoning trace in <think>...</think>, then repeat Output verbatim
after </think>. No separate evaluation / judging API call.

Amazon Bedrock credentials use the standard AWS SDK chain (see AWS docs:
https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-api.html).

Environment variables:
  AWS_REGION            AWS region for Bedrock Runtime (default: us-east-1 if unset)
  BEDROCK_MODEL_ID      default model id if --model not passed

Typical flags:
  --think-budget 2048
  --answer-max-tokens 1024
  --temperature 0
  --workers 50          # concurrent API requests (threads)

The output JSONL row contains:
  source_idx, ground_truth, generation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
DEFAULT_MODEL = os.getenv("BEDROCK_MODEL_ID") or "anthropic.claude-3-5-sonnet-20240620-v1:0"
DEFAULT_TIMEOUT = 900


@dataclass
class PipelineConfig:
    model: str = DEFAULT_MODEL
    region: str = DEFAULT_REGION
    temperature: float = 0.0
    seed: int = 42
    think_budget: int = 2048
    answer_max_tokens: int = 1024
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = 3
    retry_sleep: float = 2.0
    pass_seed_to_api: bool = False


# ----------------------------- JSONL helpers -----------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


# --------------------------- input normalization --------------------------


def normalize_record(record: Dict[str, Any], source_idx: int = 0) -> Dict[str, Any]:
    """Return a copy containing at least source_idx, question, ground_truth.

    Supported source formats:
      - question / ground_truth
      - instruction / input / output
      - Question / Response
      - input / answer, response, gt, reference_answer
    """
    out = dict(record)
    out.setdefault("source_idx", source_idx)

    if "question" not in out:
        if "Question" in out:
            out["question"] = out["Question"]
        elif "input" in out:
            if out.get("instruction"):
                out["question"] = str(out["instruction"]).rstrip() + "\n\n" + str(out["input"]).lstrip()
            else:
                out["question"] = out["input"]
        elif "prompt" in out:
            out["question"] = out["prompt"]

    if "ground_truth" not in out:
        for key in ("Response", "response", "output", "answer", "gt", "reference_answer", "reference"):
            if key in out and out[key] is not None:
                out["ground_truth"] = out[key]
                break

    if "question" not in out or out.get("question") is None:
        raise ValueError(f"record source_idx={source_idx} has no question-like field")
    if "ground_truth" not in out or out.get("ground_truth") is None:
        raise ValueError(f"record source_idx={source_idx} has no ground_truth-like field")

    out["question"] = str(out["question"])
    out["ground_truth"] = str(out["ground_truth"])
    return out


# ------------------------------- API client -------------------------------


def _messages_to_bedrock_converse(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split OpenAI-style messages into Bedrock Converse `system` blocks and `messages`."""
    system_blocks: List[Dict[str, Any]] = []
    conv: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role") or "user"
        text = m.get("content") or ""
        if role == "system":
            system_blocks.append({"text": text})
        elif role in ("user", "assistant"):
            conv.append({"role": role, "content": [{"text": text}]})
        else:
            conv.append({"role": "user", "content": [{"text": text}]})
    return system_blocks, conv


def _bedrock_usage_to_tokens(usage_raw: Any) -> Dict[str, int]:
    if usage_raw is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(usage_raw, dict):
        inp = int(usage_raw.get("inputTokens") or usage_raw.get("input_tokens") or 0)
        out = int(usage_raw.get("outputTokens") or usage_raw.get("output_tokens") or 0)
        tot = int(usage_raw.get("totalTokens") or usage_raw.get("total_tokens") or (inp + out))
        return {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": tot}
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def deepseek_chat(
    messages: List[Dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    config: PipelineConfig,
) -> Dict[str, Any]:
    """Call Amazon Bedrock Runtime `converse` (replaces previous DeepSeek HTTP API)."""
    try:
        import boto3
        from botocore.config import Config as BotoConfig
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise RuntimeError("Amazon Bedrock requires boto3: pip install boto3") from exc

    region = (config.region or DEFAULT_REGION).strip() or DEFAULT_REGION
    boto_cfg = BotoConfig(
        read_timeout=int(config.timeout),
        connect_timeout=30,
        retries={"max_attempts": 0},
    )
    client = boto3.client("bedrock-runtime", region_name=region, config=boto_cfg)

    system_blocks, conv_messages = _messages_to_bedrock_converse(messages)
    kwargs: Dict[str, Any] = {
        "modelId": model,
        "messages": conv_messages,
        "inferenceConfig": {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
        },
    }
    if system_blocks:
        kwargs["system"] = system_blocks

    last_exc: Optional[BaseException] = None
    for attempt in range(1, config.max_retries + 1):
        started = time.time()
        try:
            raw = client.converse(**kwargs)
            output_msg = (raw.get("output") or {}).get("message") or {}
            parts: List[str] = []
            for block in output_msg.get("content") or []:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
            content = "".join(parts)
            usage = _bedrock_usage_to_tokens(raw.get("usage"))
            return {
                "content": content,
                "reasoning_content": None,
                "usage": usage,
                "latency_sec": time.time() - started,
                "raw_response": raw,
            }
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "") if exc.response else ""
            last_exc = exc
            retryable = code in (
                "ThrottlingException",
                "ServiceUnavailableException",
                "ModelTimeoutException",
                "InternalServerException",
                "TooManyRequestsException",
            )
            if not retryable:
                raise RuntimeError(f"Bedrock converse failed ({code}): {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

        if attempt < config.max_retries:
            time.sleep(config.retry_sleep * attempt)

    raise RuntimeError(f"Bedrock API call failed after {config.max_retries} attempts: {last_exc}")


# --------------------------------- prompts ---------------------------------


def format_dataset_input_body(item: Dict[str, Any]) -> str:
    """Plain-text Input body for the prompt (before 'Input:' label). Uses instruction+input when both exist."""
    ins = item.get("instruction")
    inp = item.get("input")
    has_ins = isinstance(ins, str) and ins.strip()
    has_inp = isinstance(inp, str) and inp.strip()
    if has_ins and has_inp:
        return f"{ins.strip()}\n\n{inp.strip()}"
    return str(item["question"]).strip()


def build_trace_infill_messages(
    input_body: str,
    output_body: str,
    think_budget: int,
    answer_max_tokens: int,
) -> List[Dict[str, str]]:
    system = (
        "You are given an Input and an Output below. The Output is authoritative and must not be changed "
        "(same wording, punctuation, spacing, and substance — no paraphrase, no omissions, no additions). "
        "Your task is only to write an intermediate reasoning trace inside "
        "<think>...</think> that plausibly leads to that Output. "
        "After </think>, output EXACTLY the same Output text again (verbatim copy). "
        "Do not use tags other than <think> and </think>. "
        "Do not state that you are an AI or discuss these instructions."
    )
    user = "\n".join(
        [
            "Input:",
            input_body,
            "",
            "Output (fixed; reproduce verbatim after </think>):",
            output_body,
            "",
            f"Reasoning trace budget: about {think_budget} tokens or fewer.",
            f"Verbatim Output repeat budget: about {answer_max_tokens} tokens or fewer.",
            "Return in this format:",
            "<think>",
            "reasoning trace",
            "</think>",
            "<exact verbatim copy of the Output above>",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ------------------------------- parsing generation -------------------------


def parse_think_answer(raw_text: str) -> Tuple[str, str]:
    """Split the model output into think_block and answer.

    This intentionally uses the first </think> delimiter after the first <think>,
    matching the behavior visible in the Cursor record: if the model writes the
    literal string '</think>' inside its own planning prose, the split can drift.
    """
    text = raw_text or ""
    lower = text.lower()
    start = lower.find("<think>")
    if start >= 0:
        content_start = start + len("<think>")
        end = lower.find("</think>", content_start)
        if end >= 0:
            think = text[content_start:end].strip()
            answer = text[end + len("</think>"):].strip()
            return think, answer
        return text[content_start:].strip(), ""
    return "", text.strip()


# ------------------------------ pipeline steps -----------------------------


def run_generation(
    item: Dict[str, Any],
    *,
    config: PipelineConfig,
) -> Dict[str, Any]:
    question = str(item["question"])
    locked_final_answer = str(item["ground_truth"])
    input_body = format_dataset_input_body(item)
    messages = build_trace_infill_messages(
        input_body,
        locked_final_answer,
        config.think_budget,
        config.answer_max_tokens,
    )
    # Keep the same completion ceiling as before: room for trace + repeating the locked answer + slack.
    max_tokens = int(config.think_budget) + int(config.answer_max_tokens) + 4096
    resp = deepseek_chat(
        messages,
        model=config.model,
        max_tokens=max_tokens,
        temperature=config.temperature,
        config=config,
    )
    raw_text = resp.get("content") or ""
    think_block, answer = parse_think_answer(raw_text)
    usage = resp["usage"]
    return {
        "question": question,
        "dataset_input": input_body,
        "dataset_output": locked_final_answer,
        "locked_final_answer": locked_final_answer,
        "trace_infill": True,
        "raw_text": raw_text,
        "think_block": think_block,
        "answer": answer,
        "model": config.model,
        "max_tokens": max_tokens,
        "latency_sec": resp.get("latency_sec"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "reasoning_content": resp.get("reasoning_content"),
    }


# ----------------------------- public call API -----------------------------


def make_config(
    *,
    model: str = DEFAULT_MODEL,
    region: Optional[str] = None,
    temperature: float = 0.0,
    seed: int = 42,
    think_budget: int = 2048,
    answer_max_tokens: int = 1024,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
    pass_seed_to_api: bool = False,
) -> PipelineConfig:
    reg = (region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION).strip()
    return PipelineConfig(
        model=model,
        region=reg,
        temperature=temperature,
        seed=seed,
        think_budget=think_budget,
        answer_max_tokens=answer_max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        pass_seed_to_api=pass_seed_to_api,
    )


def run_one(
    record: Dict[str, Any],
    source_idx: int = 0,
    *,
    think_budget: int = 2048,
    answer_max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = 42,
    model: str = DEFAULT_MODEL,
    region: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
    pass_seed_to_api: bool = False,
    **deprecated_kwargs: Any,
) -> Dict[str, Any]:
    """Run one normalized sample through trace generation only.

    Extra keyword arguments (e.g. legacy eval/planner flags) are ignored.
    """
    _ = deprecated_kwargs
    random.seed(seed)
    item = normalize_record(record, source_idx)
    config = make_config(
        model=model,
        region=region,
        temperature=temperature,
        seed=seed,
        think_budget=think_budget,
        answer_max_tokens=answer_max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        pass_seed_to_api=pass_seed_to_api,
    )

    out: Dict[str, Any] = {
        "source_idx": int(item.get("source_idx", source_idx)),
        "ground_truth": item["ground_truth"],
    }

    generation = run_generation(item, config=config)
    out["generation"] = generation

    return out


# Aliases to make external adapters robust.
run_one_sample = run_sample = process_one = process_sample = evaluate_one = run_pipeline = process_record = run_one


# ---------------------------------- CLI ------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bedrock reasoning-trace generation for QA JSONL (Input + fixed Output)")
    ap.add_argument("positional", nargs="*", help="Optional positional input/output paths for fallback compatibility")
    ap.add_argument("--input-jsonl", "--input_jsonl", "--input", "--input-file", "--input_path", "--data", dest="input_jsonl", type=Path)
    ap.add_argument("--output-jsonl", "--output_jsonl", "--output", "--output-file", "--output_path", "--out", dest="output_jsonl", type=Path)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", "--sample-limit", "--sample_limit", dest="limit", type=int, default=None)
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Bedrock model id (env BEDROCK_MODEL_ID overrides default constant)",
    )
    ap.add_argument(
        "--region",
        "--aws-region",
        dest="region",
        default=None,
        help="AWS region for Bedrock Runtime (default: AWS_REGION or us-east-1)",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--think-budget", "--think_budget", dest="think_budget", type=int, default=2048)
    ap.add_argument("--answer-max-tokens", "--answer_max_tokens", dest="answer_max_tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-retries", "--max_retries", dest="max_retries", type=int, default=3)
    ap.add_argument("--retry-sleep", "--retry_sleep", dest="retry_sleep", type=float, default=2.0)
    ap.add_argument("--pass-seed-to-api", "--pass_seed_to_api", dest="pass_seed_to_api", action="store_true")
    ap.add_argument("--continue-on-error", dest="continue_on_error", action="store_true", default=False)
    ap.add_argument(
        "--workers",
        "--concurrency",
        dest="workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel API requests using threads (default 1 = sequential). Example: --workers 50",
    )
    args = ap.parse_args(argv)

    if args.workers < 1:
        ap.error("--workers must be >= 1")

    # Positional compatibility: python script.py input.jsonl output.jsonl
    if args.input_jsonl is None and len(args.positional) >= 1:
        args.input_jsonl = Path(args.positional[0])
    if args.output_jsonl is None and len(args.positional) >= 2:
        args.output_jsonl = Path(args.positional[1])

    if args.input_jsonl is None:
        ap.error("input JSONL is required")
    if args.output_jsonl is None:
        ap.error("output JSONL is required")
    return args


def iter_selected(rows: Sequence[Dict[str, Any]], offset: int, limit: Optional[int]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    start = max(0, offset)
    end = len(rows) if limit is None else min(len(rows), start + max(0, limit))
    for i in range(start, end):
        yield i, rows[i]


def run_one_from_cli_args(record: Dict[str, Any], source_idx: int, args: argparse.Namespace) -> Dict[str, Any]:
    """Shared kwargs for sequential and parallel CLI paths."""
    return run_one(
        record,
        source_idx=source_idx,
        think_budget=args.think_budget,
        answer_max_tokens=args.answer_max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        model=args.model,
        region=args.region,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        pass_seed_to_api=args.pass_seed_to_api,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.input_jsonl)

    tasks = list(iter_selected(rows, args.offset, args.limit))
    total_tasks = len(tasks)
    out_rows: List[Dict[str, Any]] = []
    ok = 0
    failed = 0

    if args.workers <= 1:
        for source_idx, record in tasks:
            try:
                result = run_one_from_cli_args(record, source_idx, args)
                out_rows.append(result)
                ok += 1
                print(f"[{ok}/{total_tasks}] source_idx={source_idx} ok", file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                err = {
                    "source_idx": source_idx,
                    "error": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
                print(f"[failed] source_idx={source_idx}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                if args.continue_on_error:
                    out_rows.append(err)
                    continue
                raise
    else:
        results_by_idx: Dict[int, Dict[str, Any]] = {}
        progress_lock = threading.Lock()
        finished = 0

        def _submit(idx: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
            return idx, run_one_from_cli_args(rec, idx, args)

        executor = ThreadPoolExecutor(max_workers=args.workers)
        try:
            futures = {executor.submit(_submit, idx, rec): idx for idx, rec in tasks}
            for fut in as_completed(futures):
                source_idx = futures[fut]
                try:
                    idx, result = fut.result()
                    results_by_idx[idx] = result
                    ok += 1
                    with progress_lock:
                        finished += 1
                        print(
                            f"[{finished}/{total_tasks}] source_idx={idx} ok",
                            file=sys.stderr,
                            flush=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    err = {
                        "source_idx": source_idx,
                        "error": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        ),
                    }
                    print(
                        f"[failed] source_idx={source_idx}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.continue_on_error:
                        results_by_idx[source_idx] = err
                        with progress_lock:
                            finished += 1
                        continue
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
        finally:
            executor.shutdown(wait=True)

        out_rows = [results_by_idx[i] for i in sorted(results_by_idx.keys())]

    write_jsonl(args.output_jsonl, out_rows)
    print(f"wrote {len(out_rows)} rows to {args.output_jsonl} (ok={ok}, failed={failed})", file=sys.stderr, flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
