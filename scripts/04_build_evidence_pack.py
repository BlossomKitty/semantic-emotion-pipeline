# -*- coding: utf-8 -*-
r"""Build evidence pack using the local keep classifier distilled from Gemini labels.

常用命令：

1. 使用 03 训练出的本地“是否保留”分类器进行粗召回：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\04_build_evidence_pack.py

2. 控制每月最多保留多少条证据：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\04_build_evidence_pack.py --per-month 8

3. 调整保留概率阈值：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\04_build_evidence_pack.py --min-keep-prob 0.55

说明：
   这个脚本不做无标签 embedding 主题 query 召回。
   它读取 03_train_pseudo_label_classifier.py 训练出的本地 keep 分类器，
   并优先使用 03 输出的 final_keep_decisions.jsonl：
   output/models/pseudo_label_classifier/

   输出：
   output/weibo/evidence/evidence_candidates.jsonl
   output/weibo/evidence/model_predictions.jsonl
   output/weibo/evidence/evidence_pack.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "weibo" / "3666468881_browser_original_clean.jsonl"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "output" / "models" / "pseudo_label_classifier"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "weibo" / "evidence"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
LOG_PATH = PROJECT_ROOT / "output" / "logs" / "04_build_evidence_pack.log"


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


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


def parse_dt(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.max


def compact_text(value: Any, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def load_model(model_dir: Path):
    try:
        from joblib import load
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install joblib scikit-learn sentence-transformers") from exc

    path = model_dir / "keep_classifier.joblib"
    if not path.is_file():
        raise SystemExit(f"Missing trained keep classifier. Run 03_train_pseudo_label_classifier.py first.\n{path}")
    return load(path)


def load_final_decisions(model_dir: Path) -> dict[str, dict[str, Any]]:
    path = model_dir / "final_keep_decisions.jsonl"
    if not path.is_file():
        return {}
    return {str(row.get("id") or ""): row for row in read_jsonl(path) if row.get("id")}


def build(args: argparse.Namespace) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install sentence-transformers") from exc

    input_path = Path(args.input).resolve()
    model_dir = Path(args.model_dir).resolve()
    log(f"Loading cleaned records: {input_path}")
    rows = [row for row in read_jsonl(input_path) if str(row.get("text") or "").strip()]
    log(f"Loading keep classifier: {model_dir}")
    keep_classifier = load_model(model_dir)
    final_decisions = load_final_decisions(model_dir)
    log(f"Loaded final decisions: {len(final_decisions)}")

    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    log(f"Loading embedding model: {model_name}")
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode([str(row["text"]) for row in rows], normalize_embeddings=True, show_progress_bar=True)

    keep_prob_index = list(keep_classifier.classes_).index(True)
    keep_probs = keep_classifier.predict_proba(embeddings)[:, keep_prob_index]

    enriched = []
    for row, keep_prob in zip(rows, keep_probs):
        created_at = str(row.get("created_at") or row.get("date") or "")
        text = str(row.get("text") or "")
        row_id = str(row.get("id") or "")
        keep_by_length = len(text) >= args.long_keep_chars
        decision = final_decisions.get(row_id)
        decision_source = "bert"
        final_keep = keep_prob >= args.min_keep_prob
        final_keep_reason = "BERT/BGE keep classifier"
        if decision:
            decision_source = str(decision.get("final_decision_source") or "bert")
            final_keep = bool(decision.get("final_keep_label"))
            final_keep_reason = str(decision.get("final_decision_reason") or "")
        if not final_keep and not keep_by_length:
            continue
        enriched.append(
            {
                "id": row_id,
                "created_at": created_at,
                "date": row.get("date", created_at[:10]),
                "month": created_at[:7],
                "text": text,
                "pred_keep_probability": round(float(keep_prob), 4),
                "final_keep": bool(final_keep),
                "final_decision_source": decision_source,
                "final_decision_reason": final_keep_reason,
                "keep_by_length": keep_by_length,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[str(row.get("month") or "unknown")].append(row)

    selected = []
    for _month, items in sorted(grouped.items()):
        ranked = sorted(
            items,
            key=lambda row: (
                bool(row.get("keep_by_length")),
                float(row.get("pred_keep_probability", 0)),
                len(str(row.get("text") or "")),
            ),
            reverse=True,
        )
        selected.extend(ranked[: args.per_month])
    selected.sort(key=lambda row: parse_dt(str(row.get("created_at") or row.get("date") or "")))

    output_dir = Path(args.output_dir).resolve()
    write_jsonl(output_dir / "model_predictions.jsonl", enriched)
    write_jsonl(output_dir / "evidence_candidates.jsonl", selected)
    write_markdown(output_dir / "evidence_pack.md", selected)

    log(f"Full records: {len(rows)}")
    log(f"Model-selected records before monthly cap: {len(enriched)}")
    log(f"Evidence candidates after monthly cap: {len(selected)}")
    log(f"Output: {output_dir}")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 本地模型粗召回证据包",
        "",
        f"- 候选数量：{len(rows)}",
        "- 来源：Gemini 伪标签蒸馏出的本地分类器",
        "- 决策规则：优先使用 03 的 LLM 低置信裁决；没有裁决时使用本地 BERT/BGE keep 模型",
        "",
    ]
    current_month = ""
    for row in rows:
        month = str(row.get("month") or "unknown")
        if month != current_month:
            current_month = month
            lines.append(f"## {month}")
        lines.extend(
            [
                f"- `{row.get('created_at')}` `{row.get('id')}` keep_prob={row.get('pred_keep_probability')}",
                f"  - final_keep: {row.get('final_keep')} source={row.get('final_decision_source')}",
                f"  - keep_by_length: {row.get('keep_by_length')}",
                f"  - text: {compact_text(row.get('text'))}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build evidence pack using local keep classifier.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--per-month", type=int, default=8)
    parser.add_argument("--min-keep-prob", type=float, default=0.5)
    parser.add_argument("--long-keep-chars", type=int, default=180)
    return parser


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
