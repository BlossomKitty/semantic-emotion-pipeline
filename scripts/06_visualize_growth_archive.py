# -*- coding: utf-8 -*-
r"""Visualize the personal growth archive.

环境变量设置（PowerShell）：

   # 情感分析模型，默认使用 IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment。
   # 如已提前下载到本地，也可以把这里改成本地目录。
   $env:SENTIMENT_MODEL="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"

   # 语义地图默认使用 BGE；如已提前下载到本地，也可以把这里改成本地目录。
   $env:EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"

常用命令：

1. 生成全部可视化：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\06_visualize_growth_archive.py

2. 不生成语义地图，只生成轻量图表：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\06_visualize_growth_archive.py --skip-semantic-map

3. 只重新使用已有情感缓存，不重新跑情感模型：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\06_visualize_growth_archive.py --skip-sentiment

依赖：

   pip install pandas plotly scikit-learn transformers torch sentence-transformers tqdm

说明：
   本脚本只做可视化分析，不做诊断。
   情感信号来自 IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment。
   技能/主题可视化使用 TF-IDF + NMF 做无监督主题发现，不使用情感词典。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_JSONL = PROJECT_ROOT / "data" / "processed" / "weibo" / "3666468881_browser_original_clean.jsonl"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "output" / "weibo" / "labels" / "label_batches"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "output" / "weibo" / "evidence" / "model_predictions.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visualizations" / "weibo"
DEFAULT_SENTIMENT_MODEL = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
LOG_PATH = PROJECT_ROOT / "output" / "logs" / "06_visualize_growth_archive.log"


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
    if not path.is_file():
        return rows
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


def parse_dt(value: Any) -> datetime:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.max


def month_of(value: Any) -> str:
    dt = parse_dt(value)
    return dt.strftime("%Y-%m") if dt != datetime.max else "unknown"


def compact_text(value: Any, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def load_clean_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        created_at = str(row.get("created_at") or row.get("date") or "")
        rows.append(
            {
                "id": str(row.get("id") or ""),
                "created_at": created_at,
                "date": str(row.get("date") or created_at[:10]),
                "month": month_of(created_at),
                "year": month_of(created_at)[:4],
                "text": text,
                "text_length": len(text),
            }
        )
    return rows


def load_label_rows(label_dir: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    if not label_dir.is_dir():
        return labels
    for path in sorted(label_dir.glob("batch_*.jsonl")):
        for row in read_jsonl(path):
            row_id = str(row.get("id") or "")
            if row_id:
                labels[row_id] = row
    return labels


def load_prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id") or ""): row for row in read_jsonl(path) if row.get("id")}


def merge_rows(
    clean_rows: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = []
    for row in clean_rows:
        item = dict(row)
        label = labels_by_id.get(row["id"], {})
        prediction = predictions_by_id.get(row["id"], {})
        item.update(
            {
                "keep_score": label.get("keep_score"),
                "keep": label.get("keep"),
                "inner_action": label.get("inner_action") or "",
                "evidence_type": label.get("evidence_type") or [],
                "label_provider": label.get("label_provider") or "",
                "label_model": label.get("label_model") or "",
                "pred_keep_probability": prediction.get("pred_keep_probability"),
                "keep_by_length": prediction.get("keep_by_length"),
            }
        )
        merged.append(item)
    return merged


def normalize_sentiment_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {"sentiment_label": "", "sentiment_score": None, "positive_score": None, "negative_score": None}
    label_to_score = {str(item.get("label") or "").lower(): float(item.get("score") or 0) for item in scores}
    positive = None
    negative = None
    for label, score in label_to_score.items():
        if "pos" in label or "positive" in label or label in {"1", "label_1"}:
            positive = score
        if "neg" in label or "negative" in label or label in {"0", "label_0"}:
            negative = score
    if positive is None and len(scores) == 2:
        sorted_scores = sorted(scores, key=lambda item: str(item.get("label") or ""))
        negative = float(sorted_scores[0].get("score") or 0)
        positive = float(sorted_scores[1].get("score") or 0)
    top = max(scores, key=lambda item: float(item.get("score") or 0))
    return {
        "sentiment_label": top.get("label"),
        "sentiment_score": float(top.get("score") or 0),
        "positive_score": positive,
        "negative_score": negative,
    }


def load_or_run_sentiment(rows: list[dict[str, Any]], cache_path: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    cached = {str(row.get("id") or ""): row for row in read_jsonl(cache_path)}
    if args.skip_sentiment:
        log(f"Skipping sentiment model. Loaded cached sentiment rows={len(cached)}")
        return cached

    missing = [row for row in rows if row["id"] not in cached]
    if not missing:
        log(f"Sentiment cache is complete: {cache_path}")
        return cached

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install transformers torch") from exc

    model_name = os.environ.get("SENTIMENT_MODEL", args.sentiment_model).strip()
    log(f"Loading sentiment model: {model_name}")
    classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None, truncation=True)

    new_rows = []
    for row in progress(missing, desc="Sentiment"):
        result = classifier(compact_text(row["text"], args.sentiment_max_chars))
        scores = result[0] if result and isinstance(result[0], list) else result
        normalized = normalize_sentiment_scores(scores)
        normalized.update(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "month": row["month"],
                "sentiment_model": model_name,
            }
        )
        cached[row["id"]] = normalized
        new_rows.append(normalized)
        if len(new_rows) % args.cache_flush_every == 0:
            write_jsonl(cache_path, list(cached.values()))

    write_jsonl(cache_path, list(cached.values()))
    log(f"Wrote sentiment cache: {cache_path}")
    return cached


def add_sentiment(rows: list[dict[str, Any]], sentiment_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    for row in rows:
        item = dict(row)
        sentiment = sentiment_by_id.get(row["id"], {})
        item.update(
            {
                "sentiment_label": sentiment.get("sentiment_label"),
                "sentiment_score": sentiment.get("sentiment_score"),
                "positive_score": sentiment.get("positive_score"),
                "negative_score": sentiment.get("negative_score"),
                "sentiment_model": sentiment.get("sentiment_model"),
            }
        )
        merged.append(item)
    return merged


def as_dataframe(rows: list[dict[str, Any]]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install pandas") from exc
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["is_labeled"] = df["keep_score"].notna()
    df["is_kept_label"] = df["keep_score"].fillna(-1).astype(float) >= 3
    df["is_long"] = df["text_length"].fillna(0).astype(int) >= 180
    df["positive_score"] = pd.to_numeric(df["positive_score"], errors="coerce")
    df["keep_score"] = pd.to_numeric(df["keep_score"], errors="coerce")
    df["pred_keep_probability"] = pd.to_numeric(df["pred_keep_probability"], errors="coerce")
    return df


def write_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    log(f"Wrote chart: {path}")


def build_density_chart(df, output_dir: Path) -> None:
    import plotly.graph_objects as go

    monthly = (
        df.groupby("month", dropna=False)
        .agg(
            posts=("id", "count"),
            long_posts=("is_long", "sum"),
            labeled=("is_labeled", "sum"),
            kept=("is_kept_label", "sum"),
        )
        .reset_index()
    )
    fig = go.Figure()
    for column, name in (
        ("posts", "微博数量"),
        ("long_posts", "长文本数量"),
        ("labeled", "已伪标签数量"),
        ("kept", "成长证据数量"),
    ):
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly[column], mode="lines+markers", name=name))
    fig.update_layout(title="事件密度与成长证据密度", xaxis_title="月份", yaxis_title="数量", hovermode="x unified")
    write_plot(fig, output_dir / "01_event_density.html")
    monthly.to_csv(output_dir / "event_density_monthly.csv", index=False, encoding="utf-8-sig")


def build_sentiment_chart(df, output_dir: Path) -> None:
    import plotly.graph_objects as go

    valid = df.dropna(subset=["positive_score"])
    if valid.empty:
        log("No sentiment scores available; skipped sentiment chart.")
        return
    monthly = (
        valid.groupby("month", dropna=False)
        .agg(
            positive_mean=("positive_score", "mean"),
            positive_median=("positive_score", "median"),
            posts=("id", "count"),
        )
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["positive_mean"], mode="lines+markers", name="月均正向分"))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["positive_median"], mode="lines+markers", name="月中位正向分"))
    fig.update_layout(title="情感趋势：中文 RoBERTa 情感分", xaxis_title="月份", yaxis_title="positive_score", hovermode="x unified")
    write_plot(fig, output_dir / "02_sentiment_timeline.html")
    monthly.to_csv(output_dir / "sentiment_monthly.csv", index=False, encoding="utf-8-sig")


def build_keep_chart(df, output_dir: Path) -> None:
    import plotly.graph_objects as go

    labeled = df.dropna(subset=["keep_score"])
    if labeled.empty:
        log("No Gemini labels available; skipped keep score chart.")
        return
    monthly = (
        labeled.groupby("month", dropna=False)
        .agg(
            keep_score_mean=("keep_score", "mean"),
            kept=("is_kept_label", "sum"),
            labeled=("id", "count"),
        )
        .reset_index()
    )
    monthly["kept_ratio"] = monthly["kept"] / monthly["labeled"].clip(lower=1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["kept"], name="保留证据数"))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["keep_score_mean"], mode="lines+markers", name="平均 keep_score", yaxis="y2"))
    fig.update_layout(
        title="成长证据保留趋势",
        xaxis_title="月份",
        yaxis=dict(title="保留证据数"),
        yaxis2=dict(title="平均 keep_score", overlaying="y", side="right"),
        hovermode="x unified",
    )
    write_plot(fig, output_dir / "03_keep_score_timeline.html")
    monthly.to_csv(output_dir / "keep_score_monthly.csv", index=False, encoding="utf-8-sig")


def build_action_and_type_charts(df, output_dir: Path) -> None:
    import pandas as pd
    import plotly.express as px

    labeled = df[df["inner_action"].fillna("") != ""]
    if not labeled.empty:
        action_month = labeled.groupby(["month", "inner_action"]).size().reset_index(name="count")
        fig = px.bar(action_month, x="month", y="count", color="inner_action", title="心路动作分布（来自 Gemini 伪标签）")
        write_plot(fig, output_dir / "04_inner_action_distribution.html")
        action_month.to_csv(output_dir / "inner_action_monthly.csv", index=False, encoding="utf-8-sig")

    type_rows = []
    for row in df.to_dict("records"):
        for evidence_type in row.get("evidence_type") or []:
            type_rows.append({"month": row.get("month"), "evidence_type": evidence_type})
    if type_rows:
        type_df = pd.DataFrame(type_rows)
        type_month = type_df.groupby(["month", "evidence_type"]).size().reset_index(name="count")
        fig = px.bar(type_month, x="month", y="count", color="evidence_type", title="成长证据类型分布（来自 Gemini 伪标签）")
        write_plot(fig, output_dir / "05_evidence_type_distribution.html")
        type_month.to_csv(output_dir / "evidence_type_monthly.csv", index=False, encoding="utf-8-sig")


def build_topic_skill_chart(df, output_dir: Path, args: argparse.Namespace) -> None:
    try:
        import pandas as pd
        import plotly.express as px
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install pandas plotly scikit-learn") from exc

    source = df[df["text"].fillna("").str.len() > 0].copy()
    if source.empty:
        return
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_df=0.75, max_features=args.topic_max_features)
    matrix = vectorizer.fit_transform(source["text"].tolist())
    topic_count = min(args.topic_count, max(2, matrix.shape[0] // 10), matrix.shape[1] - 1)
    if topic_count < 2:
        log("Not enough text features for topic/skill chart.")
        return
    nmf = NMF(n_components=topic_count, random_state=42, init="nndsvda", max_iter=500)
    weights = nmf.fit_transform(matrix)
    feature_names = vectorizer.get_feature_names_out()
    topic_names = []
    for topic_idx, component in enumerate(nmf.components_):
        top_terms = [feature_names[index] for index in component.argsort()[-8:][::-1]]
        topic_names.append(f"T{topic_idx + 1}: {' / '.join(top_terms[:4])}")
    source["topic"] = [topic_names[index] for index in weights.argmax(axis=1)]
    topic_month = source.groupby(["month", "topic"]).size().reset_index(name="count")
    fig = px.density_heatmap(topic_month, x="month", y="topic", z="count", title="主题/技能线索热力图（TF-IDF + NMF，无监督）")
    write_plot(fig, output_dir / "06_topic_skill_heatmap.html")
    topic_month.to_csv(output_dir / "topic_skill_monthly.csv", index=False, encoding="utf-8-sig")

    topic_summary = pd.DataFrame({"topic": topic_names})
    topic_summary.to_csv(output_dir / "topic_skill_terms.csv", index=False, encoding="utf-8-sig")


def build_semantic_map(df, output_dir: Path, args: argparse.Namespace) -> None:
    if args.skip_semantic_map:
        log("Skipped semantic map.")
        return
    try:
        import plotly.express as px
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install sentence-transformers scikit-learn plotly") from exc

    source = df[df["text"].fillna("").str.len() > 0].copy()
    if source.empty:
        return
    model_name = os.environ.get("EMBEDDING_MODEL", args.embedding_model).strip()
    log(f"Loading embedding model for semantic map: {model_name}")
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(source["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    source["x"] = coords[:, 0]
    source["y"] = coords[:, 1]
    source["hover_text"] = source["created_at"].fillna("") + "<br>" + source["text"].map(lambda text: compact_text(text, 160))
    color_column = "positive_score" if source["positive_score"].notna().any() else "year"
    fig = px.scatter(
        source,
        x="x",
        y="y",
        color=color_column,
        hover_name="hover_text",
        size="text_length",
        size_max=12,
        title="语义空间地图（BGE embedding + PCA）",
    )
    write_plot(fig, output_dir / "07_semantic_map.html")
    source[["id", "created_at", "month", "x", "y", "positive_score", "keep_score", "text"]].to_csv(
        output_dir / "semantic_map_points.csv", index=False, encoding="utf-8-sig"
    )


def build_overview_markdown(output_dir: Path, rows: list[dict[str, Any]], labels_by_id: dict[str, Any], sentiment_count: int) -> None:
    lines = [
        "# Weibo Growth Archive Visualization",
        "",
        f"- 清洗微博数量：{len(rows)}",
        f"- 已有 Gemini/DeepSeek 伪标签数量：{len(labels_by_id)}",
        f"- 已有情感分数数量：{sentiment_count}",
        "",
        "## 输出图表",
        "",
        "- `01_event_density.html`：事件密度与成长证据密度",
        "- `02_sentiment_timeline.html`：情感趋势",
        "- `03_keep_score_timeline.html`：成长证据保留趋势",
        "- `04_inner_action_distribution.html`：心路动作分布",
        "- `05_evidence_type_distribution.html`：成长证据类型分布",
        "- `06_topic_skill_heatmap.html`：主题/技能线索热力图",
        "- `07_semantic_map.html`：语义空间地图",
        "",
        "## 方法边界",
        "",
        "- 情感趋势来自中文 RoBERTa 情感模型，只能作为情感极性信号，不等同于心理状态。",
        "- 心路动作和证据类型来自 LLM 伪标签，不由本地 BERT 预测。",
        "- 主题/技能线索来自 TF-IDF + NMF 的无监督发现，需要人工命名和校准。",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = Path(args.input).resolve()
    label_dir = Path(args.labels_dir).resolve()
    prediction_path = Path(args.predictions).resolve()

    log(f"Loading cleaned records: {clean_path}")
    clean_rows = load_clean_rows(clean_path)
    log(f"Clean records={len(clean_rows)}")

    labels_by_id = load_label_rows(label_dir)
    predictions_by_id = load_prediction_rows(prediction_path)
    log(f"Label rows={len(labels_by_id)}; prediction rows={len(predictions_by_id)}")

    rows = merge_rows(clean_rows, labels_by_id, predictions_by_id)
    sentiment_cache = output_dir / "sentiment_scores.jsonl"
    sentiment_by_id = load_or_run_sentiment(rows, sentiment_cache, args)
    rows = add_sentiment(rows, sentiment_by_id)
    write_jsonl(output_dir / "visualization_records.jsonl", rows)

    df = as_dataframe(rows)
    df.to_csv(output_dir / "visualization_records.csv", index=False, encoding="utf-8-sig")

    build_density_chart(df, output_dir)
    build_sentiment_chart(df, output_dir)
    build_keep_chart(df, output_dir)
    build_action_and_type_charts(df, output_dir)
    build_topic_skill_chart(df, output_dir, args)
    build_semantic_map(df, output_dir, args)
    build_overview_markdown(output_dir, rows, labels_by_id, len(sentiment_by_id))
    log(f"Visualization output: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Weibo growth archive signals.")
    parser.add_argument("--input", default=str(DEFAULT_CLEAN_JSONL))
    parser.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sentiment-model", default=DEFAULT_SENTIMENT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--sentiment-max-chars", type=int, default=480)
    parser.add_argument("--cache-flush-every", type=int, default=50)
    parser.add_argument("--topic-count", type=int, default=12)
    parser.add_argument("--topic-max-features", type=int, default=4000)
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--skip-semantic-map", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
