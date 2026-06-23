# -*- coding: utf-8 -*-
r"""Gemini first-pass pseudo labeling for the growth archive.

环境变量设置（PowerShell）：

   $env:GEMINI_API_KEY="你的 Google AI Studio API Key"
   $env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"

常用命令：

1. 从全量清洗微博中抽样不超过 30%，用 Gemini 做首轮伪标签：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\02_gemini_first_pass_label.py

2. 检查将要发送给 Gemini 的 prompt，不调用 API：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\02_gemini_first_pass_label.py --dry-run

3. 调整首轮伪标签比例，例如最多 20%：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\02_gemini_first_pass_label.py --max-label-ratio 0.2

4. 断点续跑：
   直接重新运行同一条命令即可。已完成批次会从 output/weibo/labels/label_batches/ 自动恢复。

说明：
   当前架构是：
   全量清洗微博
     -> Gemini 首轮伪标签（默认最多全量 30%）
     -> 本地模型只学习是否保留
     -> 本地模型粗召回
     -> Gemini 复核/rerank
     -> Gemini 最终解释

   本脚本只负责第一步：Gemini 首轮伪标签。
   首轮标注已整合校准原则：文学评论、历史叙事、作品解读、审美选择、
   社会观察和创作方法反思，不应仅因没有直接出现“我”而被判为低信号。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "weibo" / "3666468881_browser_original_clean.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "weibo" / "labels"
DEFAULT_FULL_LABELS = DEFAULT_OUTPUT_DIR / "gemini_first_pass_labels.jsonl"
DEFAULT_PSEUDO_LABELS = DEFAULT_OUTPUT_DIR / "pseudo_labels_train.jsonl"
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
LOG_PATH = PROJECT_ROOT / "output" / "logs" / "02_gemini_first_pass_label.log"


class GeminiRateLimitError(RuntimeError):
    """Raised when Gemini returns a quota/rate-limit response and fallback can be used."""


SYSTEM_INSTRUCTION = """你是成长轨迹档案的首轮伪标签标注器，不是心理诊断器。

任务：
1. 对微博文本生成伪标签。
2. 判断该文本是否值得进入成长轨迹训练集。
3. 给出 keep_score 和反解释，其中只有 keep_score 会用于后续本地模型学习。

硬性约束：
1. 不做临床诊断。
2. 不根据单条文本判断人格。
3. 不把诗性表达直接等同于心理事实。
4. 不把关系对象神化、病理化或工具化。
5. 不因为文本是文学评论、历史叙事、作品解读、审美偏好或社会观察，就自动判为低信号。
6. 技术/创作方法反思即使没有直接出现“我”，也可能是行动方式、价值结构或能力建设证据。
7. 只有文本缺少可解释的经验、价值、行动、关系、主题或阶段信息时，才标为低信号。
8. 只输出 JSON，不输出 Markdown。
"""


USER_PROMPT = """请对下面微博做首轮伪标签。

返回 JSON 数组。每个元素必须包含：

{
  "id": "...",
  "keep_score": 0-5,
  "evidence_type": ["情绪调节"|"自我叙事"|"支点系统"|"关系边界"|"科研道路"|"创作技术"|"文学历史观察"|"社会观察"|"低信号"],
  "inner_action": "困惑|自我安慰|重估|行动|告别|重启|接受|建立支点|意义重构|低信号",
  "growth_signal": "这条文本对成长轨迹有什么信号；证据不足就写证据不足",
  "why_keep": "为什么值得保留；如不值得保留，写不保留理由",
  "counter_interpretation": "反解释或证据不足说明"
}

重要标注原则：
- 文学评论、历史叙事、作品解读、审美偏好、社会观察、技术/创作方法反思，不应仅因没有直接出现“我”而被判为低信号。
- 这类文本可能体现用户的自我选择、价值结构、审美秩序、社会理解方式、创作方法或长期主题。
- 如果文本能够显示持续关注的主题、判断标准、价值取向、行动方法或阶段性问题，即使不是直接自述，也可以保留为成长轨迹证据。
- 如果证据不足，应明确写出证据不足，并给出反解释，不要强行拔高。

keep_score 标准：
- 5：高度适合作为成长轨迹证据，有阶段变化、支点、自我叙事、行动转折。
- 4：有明显解释价值，但需要反解释约束。
- 3：可作为背景材料或弱证据。
- 2：信息较弱，不优先。
- 1：低信号。
- 0：噪声或不适合分析。

不要为了凑标签而过度解释。证据不足时应标为低信号或弱证据。
只返回 JSON 数组。

候选微博：
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_dt(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.max


def compact_text(value: Any, limit: int = 900) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def select_labeling_rows(rows: list[dict[str, Any]], max_ratio: float, long_keep_chars: int) -> list[dict[str, Any]]:
    """Select <= max_ratio rows for Gemini labeling without keyword dictionaries."""
    max_rows = max(1, int(len(rows) * max_ratio))
    normalized = []
    for row in rows:
        item = dict(row)
        item["_dt"] = parse_dt(str(row.get("created_at") or row.get("date") or ""))
        item["_month"] = item["_dt"].strftime("%Y-%m") if item["_dt"] != datetime.max else "unknown"
        item["_length"] = len(str(row.get("text") or ""))
        normalized.append(item)

    selected: dict[str, dict[str, Any]] = {}

    # Long texts are high-information by default, but still capped globally.
    long_rows = sorted(
        [row for row in normalized if row["_length"] >= long_keep_chars],
        key=lambda row: (row["_length"], row["_dt"]),
        reverse=True,
    )
    for row in long_rows[:max_rows]:
        selected[str(row.get("id"))] = row

    # Time-balanced fill: each month contributes its longest remaining texts.
    if len(selected) < max_rows:
        by_month: dict[str, list[dict[str, Any]]] = {}
        for row in normalized:
            by_month.setdefault(row["_month"], []).append(row)
        month_ranked = {
            month: sorted(items, key=lambda row: row["_length"], reverse=True)
            for month, items in by_month.items()
        }
        pointer = 0
        months = sorted(month_ranked)
        while len(selected) < max_rows:
            added = False
            for month in months:
                items = month_ranked[month]
                if pointer >= len(items):
                    continue
                row = items[pointer]
                row_id = str(row.get("id"))
                if row_id not in selected:
                    selected[row_id] = row
                    added = True
                    if len(selected) >= max_rows:
                        break
            if not added:
                break
            pointer += 1

    cleaned = []
    for row in selected.values():
        item = {key: value for key, value in row.items() if not key.startswith("_")}
        cleaned.append(item)
    return sorted(cleaned, key=lambda row: parse_dt(str(row.get("created_at") or row.get("date") or "")))


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


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
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                raise GeminiRateLimitError("Gemini rate limit: HTTP 429 Too Many Requests") from exc
            if attempt >= args.retries:
                raise
            delay = args.retry_delay * attempt
            if isinstance(exc, urllib.error.HTTPError):
                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                if exc.code == 429:
                    delay = max(delay, args.rate_limit_delay)
                elif exc.code in {500, 502, 503, 504}:
                    delay = max(delay, min(args.rate_limit_delay, 20.0))
            log(f"Gemini request failed on attempt {attempt}: {exc}. retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError("Gemini request failed.")


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
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}",
        },
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
            log(f"DeepSeek request failed on attempt {attempt}: {exc}. retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError("DeepSeek request failed.")


def generate_with_fallback(prompt: str, args: argparse.Namespace) -> tuple[str, str, str]:
    gemini_model = get_env("GEMINI_MODEL", args.model)
    try:
        return gemini_generate(prompt, args), "gemini", gemini_model
    except GeminiRateLimitError as exc:
        deepseek_model = get_env("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        log(f"{exc}. Falling back to DeepSeek model={deepseek_model}")
        return deepseek_generate(prompt, args), "deepseek", deepseek_model


def validate_gemini_key(args: argparse.Namespace) -> None:
    log("Checking Gemini API key...")
    try:
        text = gemini_generate("只返回 OK。", args)
        if "OK" not in text.upper():
            raise SystemExit(f"Gemini API key check failed. Response preview: {text[:120]}")
        log(f"Gemini API key is valid. model={get_env('GEMINI_MODEL', args.model)}")
    except GeminiRateLimitError:
        log("Gemini API key exists but Gemini is rate-limited. Checking DeepSeek fallback...")
        text = deepseek_generate("只返回 OK。", args)
        if "OK" not in text.upper():
            raise SystemExit(f"DeepSeek API key check failed. Response preview: {text[:120]}")
        log(f"DeepSeek fallback is valid. model={get_env('DEEPSEEK_MODEL', DEFAULT_DEEPSEEK_MODEL)}")


def load_completed(batch_dir: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    if not batch_dir.is_dir():
        return completed
    for path in sorted(batch_dir.glob("batch_*.jsonl")):
        for row in read_jsonl(path):
            completed[str(row.get("id"))] = row
    return completed


def next_batch_index(batch_dir: Path) -> int:
    indexes = []
    if batch_dir.is_dir():
        for path in batch_dir.glob("batch_*.jsonl"):
            match = re.search(r"batch_(\d+)\.jsonl$", path.name)
            if match:
                indexes.append(int(match.group(1)))
    return max(indexes, default=0) + 1


def merge_labels(batch: list[dict[str, Any]], labels: list[dict[str, Any]], provider: str, model: str) -> list[dict[str, Any]]:
    by_id = {str(item.get("id")): item for item in labels if item.get("id") is not None}
    merged = []
    for row in batch:
        label = by_id.get(str(row.get("id")), {})
        keep_score = float(label.get("keep_score", 0) or 0)
        merged.append(
            {
                "id": row.get("id", ""),
                "created_at": row.get("created_at", ""),
                "date": row.get("date", ""),
                "month": str(row.get("created_at", ""))[:7],
                "text": row.get("text", ""),
                "label_provider": provider,
                "label_model": model,
                "keep_score": keep_score,
                "keep": keep_score >= 3,
                "evidence_type": label.get("evidence_type") or [],
                "inner_action": label.get("inner_action") or "低信号",
                "growth_signal": label.get("growth_signal") or "",
                "why_keep": label.get("why_keep") or "",
                "counter_interpretation": label.get("counter_interpretation") or "",
            }
        )
    return merged


def build_prompt(batch: list[dict[str, Any]]) -> str:
    compact = [
        {
            "id": row.get("id"),
            "created_at": row.get("created_at"),
            "text": compact_text(row.get("text")),
        }
        for row in batch
    ]
    return USER_PROMPT + json.dumps(compact, ensure_ascii=False, indent=2)


def write_markdown(path: Path, rows: list[dict[str, Any]], min_keep: float) -> None:
    kept = [row for row in rows if float(row.get("keep_score", 0) or 0) >= min_keep]
    kept.sort(key=lambda row: (str(row.get("created_at")), -float(row.get("keep_score", 0) or 0)))
    lines = [
        "# Gemini First-Pass Pseudo Labels",
        "",
        f"- 标注数量：{len(rows)}",
        f"- keep_score >= {min_keep}：{len(kept)}",
        "",
    ]
    for row in kept:
        lines.extend(
            [
                f"## {row.get('created_at')}｜{row.get('id')}｜keep={row.get('keep_score')}",
                "",
                f"- label_model：{row.get('label_provider', '')}/{row.get('label_model', '')}",
                f"- evidence_type：{', '.join(row.get('evidence_type') or [])}",
                f"- inner_action：{row.get('inner_action')}",
                f"- growth_signal：{row.get('growth_signal')}",
                f"- why_keep：{row.get('why_keep')}",
                f"- counter_interpretation：{row.get('counter_interpretation')}",
                f"- text：{compact_text(row.get('text'), 500)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def persist_all_labels(args: argparse.Namespace, batch_dir: Path) -> list[dict[str, Any]]:
    all_labeled = list(load_completed(batch_dir).values())
    all_labeled.sort(key=lambda row: parse_dt(str(row.get("created_at") or row.get("date") or "")))
    write_jsonl(Path(args.output_jsonl).resolve(), all_labeled)
    write_jsonl(Path(args.pseudo_labels).resolve(), all_labeled)
    write_markdown(Path(args.output_md).resolve(), all_labeled, args.min_keep_score)
    return all_labeled


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    batch_dir = output_dir / "label_batches"
    raw_dir = output_dir / "raw_responses"
    prompt_dir = output_dir / "label_prompts"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(input_path)
    selected = select_labeling_rows(all_rows, args.max_label_ratio, args.long_keep_chars)
    log(f"Full records={len(all_rows)}; selected for Gemini labeling={len(selected)}; ratio={len(selected)/max(len(all_rows), 1):.2%}")

    completed = load_completed(batch_dir)
    remaining = [row for row in selected if str(row.get("id")) not in completed]
    log(f"Completed labels={len(completed)}; remaining={len(remaining)}")

    if args.dry_run:
        for index, batch in enumerate(chunks(remaining, args.batch_size), start=1):
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / f"batch_{index:03d}.prompt.md").write_text(build_prompt(batch), encoding="utf-8")
        log(f"Wrote dry-run prompts: {prompt_dir}")
        return

    validate_gemini_key(args)

    start_index = next_batch_index(batch_dir)
    if remaining:
        log(f"Next batch file index={start_index}")

    for index, batch in progress(list(enumerate(chunks(remaining, args.batch_size), start=start_index)), desc="Gemini labeling"):
        batch_path = batch_dir / f"batch_{index:03d}.jsonl"
        raw_path = raw_dir / f"batch_{index:03d}.txt"
        prompt = build_prompt(batch)
        labels = None
        last_text = ""
        label_provider = ""
        label_model = ""
        for attempt in range(1, args.parse_retries + 1):
            last_text, label_provider, label_model = generate_with_fallback(prompt, args)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(last_text, encoding="utf-8")
            try:
                labels = extract_json(last_text)
                if not isinstance(labels, list):
                    raise ValueError("Gemini response is not a JSON array")
                break
            except Exception as exc:
                log(f"JSON parse failed for batch {index}, attempt {attempt}: {exc}")
                if attempt >= args.parse_retries:
                    raise
                time.sleep(args.retry_delay * attempt)
        merged = merge_labels(batch, labels or [], label_provider, label_model)
        write_jsonl(batch_path, merged)
        persist_all_labels(args, batch_dir)
        log(f"Processed batch {index}, rows={len(batch)}, model={label_provider}/{label_model}")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    all_labeled = persist_all_labels(args, batch_dir)
    log(f"Wrote labels: {args.output_jsonl}")
    log(f"Wrote training pseudo labels: {args.pseudo_labels}")
    log(f"Wrote markdown: {args.output_md}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini first-pass pseudo labeling with resume support.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_FULL_LABELS))
    parser.add_argument("--pseudo-labels", default=str(DEFAULT_PSEUDO_LABELS))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_DIR / "gemini_first_pass_labels.md"))
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-label-ratio", type=float, default=0.3)
    parser.add_argument("--long-keep-chars", type=int, default=180)
    parser.add_argument("--min-keep-score", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--parse-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--rate-limit-delay", type=float, default=45.0)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
