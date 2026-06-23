# -*- coding: utf-8 -*-
r"""Train a local keep/not-keep classifier with LLM rerank calibration.

环境变量设置（PowerShell）：

   $env:GEMINI_API_KEY="你的 Gemini API Key"
   $env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
   $env:EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"

常用命令：

1. 使用 02/02b 伪标签训练本地“是否保留”分类器，并对疑难样本做 LLM rerank：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\03_train_pseudo_label_classifier.py

2. 不调用 LLM，只用已有伪标签训练：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\03_train_pseudo_label_classifier.py --skip-llm-rerank

3. 调整 LLM rerank 低置信阈值，例如置信度低于 0.85 的样本全部复核：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\03_train_pseudo_label_classifier.py --rerank-confidence-threshold 0.85

说明：
   这不是完整 fine-tune BERT；当前数据量下更稳妥的是：
   BGE/BERT embedding 固定编码文本 + scikit-learn 分类头。

   训练过程：
   伪标签 -> 本地训练/验证 -> 全量训练高置信筛选器
   -> 置信度低于 0.9 的样本交给 LLM rerank
   -> 输出 final_keep_decisions.jsonl。

   本地模型只学习一件事：文本是否值得保留为成长轨迹证据。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "weibo" / "labels" / "pseudo_labels_train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "models" / "pseudo_label_classifier"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
LOG_PATH = PROJECT_ROOT / "output" / "logs" / "03_train_pseudo_label_classifier.log"


class GeminiRateLimitError(RuntimeError):
    """Raised when Gemini returns rate limit and DeepSeek fallback can be used."""


SYSTEM_INSTRUCTION = """你是成长轨迹证据的 rerank 校准器，不是心理诊断器。

你的任务：
1. 只判断文本是否应保留为成长轨迹证据。
2. 不输出心理诊断。
3. 不因为文本是文学评论、历史叙事、作品解读、审美偏好或社会观察，就自动判为低信号。
4. 文学评论、历史叙事、作品解读、审美选择、社会观察、技术/创作方法反思，
   即使没有直接出现“我”，也可能是用户自我选择、价值结构、审美秩序和社会理解方式的证据。
5. 只输出 JSON 数组，不输出 Markdown。
"""


USER_PROMPT = """请对下面样本做 keep / not_keep rerank 校准。

你会看到：
- 原始 LLM 伪标签 keep_score / keep
- 本地 BERT/BGE 分类器预测的 keep 概率
- 文本原文

请只判断该文本是否应该保留为成长轨迹证据。

返回 JSON 数组，每个元素必须包含：

{
  "id": "...",
  "final_keep": true | false,
  "final_keep_score": 0-5,
  "rerank_reason": "为什么保留或不保留，必须简短",
  "counter_interpretation": "反解释或证据不足说明"
}

判断标准：
- 保留：能体现成长轨迹、自我叙事、价值结构、社会观察、审美选择、关系边界、科研道路、技术/创作方法、支点系统等。
- 不保留：纯转发、纯日常噪声、缺乏个人选择或观察视角、无法作为阶段证据。
- 长文学评论/历史叙事/作品解读不能因“不是直接写我”而自动判低信号。

只返回 JSON 数组。

待校准样本：
"""


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_text(value: Any, limit: int = 900) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def extract_json(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def load_training_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    usable = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if text:
            usable.append(row)
    return usable


def label_from_score(row: dict[str, Any], keep_threshold: float) -> bool:
    return float(row.get("keep_score", 0) or 0) >= keep_threshold


def gemini_generate(prompt: str, args: argparse.Namespace) -> str:
    api_key = get_env("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set.")
    base = get_env("GEMINI_API_BASE", DEFAULT_GEMINI_API_BASE).rstrip("/")
    model = get_env("GEMINI_MODEL", args.model)
    url = f"{base}/v1beta/models/{model}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": args.temperature,
            "maxOutputTokens": args.max_output_tokens,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8", "x-goog-api-key": api_key},
        method="POST",
    )
    for attempt in range(1, args.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            return "\n".join(str(part.get("text") or "") for part in parts).strip()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise GeminiRateLimitError("Gemini rate limit: HTTP 429 Too Many Requests") from exc
            if attempt >= args.retries:
                raise
            delay = args.retry_delay * attempt
            log(f"Gemini rerank failed on attempt {attempt}: {exc}. retrying in {delay:.1f}s...")
            time.sleep(delay)
        except urllib.error.URLError as exc:
            if attempt >= args.retries:
                raise
            delay = args.retry_delay * attempt
            log(f"Gemini rerank failed on attempt {attempt}: {exc}. retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError("Gemini rerank failed.")


def deepseek_generate(prompt: str, args: argparse.Namespace) -> str:
    api_key = get_env("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set, but Gemini hit rate limit.")
    base = get_env("DEEPSEEK_API_BASE", DEFAULT_DEEPSEEK_API_BASE).rstrip("/")
    model = get_env("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_output_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    for attempt in range(1, args.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                return json.dumps(data, ensure_ascii=False)
            return str(((choices[0].get("message") or {}).get("content") or "")).strip()
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            if attempt >= args.retries:
                raise
            delay = args.retry_delay * attempt
            log(f"DeepSeek rerank failed on attempt {attempt}: {exc}. retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError("DeepSeek rerank failed.")


def generate_with_fallback(prompt: str, args: argparse.Namespace) -> tuple[str, str, str]:
    gemini_model = get_env("GEMINI_MODEL", args.model)
    try:
        return gemini_generate(prompt, args), "gemini", gemini_model
    except GeminiRateLimitError as exc:
        deepseek_model = get_env("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        log(f"{exc}. Falling back to DeepSeek model={deepseek_model}")
        return deepseek_generate(prompt, args), "deepseek", deepseek_model


def build_rerank_prompt(rows: list[dict[str, Any]]) -> str:
    compact = []
    for row in rows:
        compact.append(
            {
                "id": row.get("id"),
                "created_at": row.get("created_at"),
                "original_keep_score": row.get("keep_score"),
                "original_keep": row.get("original_keep_label"),
                "bert_keep_probability": row.get("initial_pred_keep_probability"),
                "text": compact_text(row.get("text")),
            }
        )
    return USER_PROMPT + json.dumps(compact, ensure_ascii=False, indent=2)


def select_rerank_candidates(rows: list[dict[str, Any]], confidence_threshold: float) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        prob = float(row.get("initial_pred_keep_probability", 0) or 0)
        confidence = max(prob, 1.0 - prob)
        row["initial_keep_confidence"] = round(float(confidence), 4)
        if confidence < confidence_threshold:
            candidates.append(row)
    candidates.sort(key=lambda row: float(row.get("initial_keep_confidence", 0)))
    return candidates


def run_llm_rerank(rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> dict[str, dict[str, Any]]:
    candidates = select_rerank_candidates(rows, args.rerank_confidence_threshold)
    log(f"LLM rerank candidates={len(candidates)}; confidence_threshold={args.rerank_confidence_threshold}")
    if not candidates:
        return {}

    raw_dir = output_dir / "llm_rerank_raw"
    updates: dict[str, dict[str, Any]] = {}
    for index, batch in enumerate(progress(chunks(candidates, args.rerank_batch_size), desc="LLM rerank"), start=1):
        prompt = build_rerank_prompt(batch)
        text, provider, model = generate_with_fallback(prompt, args)
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"rerank_{index:03d}.txt").write_text(text, encoding="utf-8")
        labels = extract_json(text)
        if not isinstance(labels, list):
            raise ValueError("LLM rerank response is not a JSON array")
        for item in labels:
            row_id = str(item.get("id") or "")
            if not row_id:
                continue
            updates[row_id] = {
                "llm_rerank_provider": provider,
                "llm_rerank_model": model,
                "llm_final_keep": bool(item.get("final_keep")),
                "llm_final_keep_score": float(item.get("final_keep_score", 0) or 0),
                "llm_rerank_reason": item.get("rerank_reason") or "",
                "llm_counter_interpretation": item.get("counter_interpretation") or "",
            }
        log(f"Processed LLM rerank batch {index}, rows={len(batch)}, model={provider}/{model}")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    return updates


def compute_metrics(labels: list[bool], predictions: list[bool]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def validate_classifier(embeddings, labels: list[bool], args: argparse.Namespace) -> tuple[Any, dict[str, float]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    train_x, valid_x, train_y, valid_y = train_test_split(
        embeddings,
        labels,
        test_size=args.validation_ratio,
        random_state=42,
        stratify=labels,
    )
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(train_x, train_y)
    valid_pred = classifier.predict(valid_x)
    return classifier, compute_metrics(valid_y, valid_pred)


def train(args: argparse.Namespace) -> None:
    try:
        from joblib import dump
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install sentence-transformers scikit-learn joblib") from exc

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading pseudo labels: {input_path}")
    rows = load_training_rows(input_path)
    if len(rows) < args.min_samples:
        raise SystemExit(f"Not enough pseudo-labeled samples: {len(rows)} < {args.min_samples}")

    original_labels = [label_from_score(row, args.keep_threshold) for row in rows]
    if len(set(original_labels)) < 2:
        raise SystemExit("Need both keep and not_keep pseudo labels to train keep classifier.")

    texts = [str(row["text"]) for row in rows]
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    log(f"Loading embedding model: {model_name}")
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    log("Training validation keep classifier")
    initial_classifier, initial_metrics = validate_classifier(embeddings, original_labels, args)
    log(f"Validation metrics: {initial_metrics}")
    if initial_metrics["f1"] < args.min_f1 or initial_metrics["recall"] < args.min_recall:
        raise SystemExit(
            "Validation did not pass; model was not written. "
            f"f1={initial_metrics['f1']:.4f} < {args.min_f1} or recall={initial_metrics['recall']:.4f} < {args.min_recall}"
        )

    log("Validation passed. Training final high-confidence keep classifier on all pseudo labels")
    keep_classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    keep_classifier.fit(embeddings, original_labels)
    keep_prob_index = list(keep_classifier.classes_).index(True)
    initial_probs = keep_classifier.predict_proba(embeddings)[:, keep_prob_index]

    working_rows = []
    for row, original_label, keep_prob in zip(rows, original_labels, initial_probs):
        item = dict(row)
        item["original_keep_label"] = bool(original_label)
        item["initial_pred_keep_probability"] = round(float(keep_prob), 4)
        confidence = max(float(keep_prob), 1.0 - float(keep_prob))
        item["initial_keep_confidence"] = round(float(confidence), 4)
        item["bert_keep_label"] = bool(keep_prob >= 0.5)
        item["final_keep_label"] = bool(keep_prob >= 0.5)
        item["final_decision_source"] = "bert"
        working_rows.append(item)

    rerank_updates = {}
    if args.skip_llm_rerank:
        log("Skipped LLM rerank by user option.")
    else:
        rerank_updates = run_llm_rerank(working_rows, args, output_dir)

    for row in working_rows:
        update = rerank_updates.get(str(row.get("id") or ""))
        if not update:
            row["llm_rerank_status"] = "not_selected"
            continue
        row.update(update)
        row["llm_rerank_status"] = "reranked"
        row["final_keep_label"] = bool(update["llm_final_keep"])
        row["final_decision_source"] = "llm"
        row["final_keep_score"] = update.get("llm_final_keep_score")
        row["final_decision_reason"] = update.get("llm_rerank_reason")
        row["final_counter_interpretation"] = update.get("llm_counter_interpretation")

    for row in working_rows:
        if row.get("final_decision_source") == "bert":
            row["final_keep_score"] = row.get("keep_score")
            row["final_decision_reason"] = "BERT/BGE keep classifier confidence >= threshold"
            row["final_counter_interpretation"] = row.get("counter_interpretation") or ""
        row["final_pred_keep_probability"] = row.get("initial_pred_keep_probability")

    reranked_rows = sorted(working_rows, key=lambda row: float(row.get("final_pred_keep_probability", 0)), reverse=True)
    write_jsonl(output_dir / "training_rerank.jsonl", reranked_rows)
    write_jsonl(output_dir / "final_keep_decisions.jsonl", sorted(working_rows, key=lambda row: str(row.get("created_at") or "")))
    dump(keep_classifier, output_dir / "keep_classifier.joblib")
    for stale_name in (
        "inner_action_classifier.joblib",
        "evidence_type_classifier.joblib",
        "evidence_type_binarizer.joblib",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    write_json(
        output_dir / "metadata.json",
        {
            "embedding_model": model_name,
            "task": "keep_binary_classification_with_llm_rerank",
            "training_rows": len(rows),
            "keep_threshold": args.keep_threshold,
            "validation_ratio": args.validation_ratio,
            "validation_metrics": initial_metrics,
            "min_f1": args.min_f1,
            "min_recall": args.min_recall,
            "original_keep_counts": dict(Counter(str(value) for value in original_labels)),
            "final_keep_counts": dict(Counter(str(row.get("final_keep_label")) for row in working_rows)),
            "final_decision_source_counts": dict(Counter(str(row.get("final_decision_source")) for row in working_rows)),
            "llm_reranked_rows": len(rerank_updates),
            "skip_llm_rerank": args.skip_llm_rerank,
            "note": "Embedding encoder is not fine-tuned. BERT/BGE handles high-confidence samples; LLM decisions override low-confidence samples.",
        },
    )
    log(f"Training rows: {len(rows)}")
    log(f"Original keep counts: {dict(Counter(str(value) for value in original_labels))}")
    log(f"Final keep counts: {dict(Counter(str(row.get('final_keep_label')) for row in working_rows))}")
    log(f"Final decision source counts: {dict(Counter(str(row.get('final_decision_source')) for row in working_rows))}")
    log(f"LLM reranked rows: {len(rerank_updates)}")
    log(f"Wrote training rerank: {output_dir / 'training_rerank.jsonl'}")
    log(f"Wrote final decisions: {output_dir / 'final_keep_decisions.jsonl'}")
    log(f"Saved model directory: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a local keep/not-keep classifier with LLM rerank calibration.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--keep-threshold", type=float, default=3.0)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--min-f1", type=float, default=0.55)
    parser.add_argument("--min-recall", type=float, default=0.60)
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--rerank-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--rerank-batch-size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--skip-llm-rerank", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
