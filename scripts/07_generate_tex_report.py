# -*- coding: utf-8 -*-
r"""Generate a XeLaTeX PDF report from pipeline outputs.

常用命令：

   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\07_generate_tex_report.py

然后编译：

   xelatex -interaction=nonstopmode -output-directory D:\Users\18905\Desktop\资料整理\0.自我复盘系统\output\reports\weibo_growth D:\Users\18905\Desktop\资料整理\0.自我复盘系统\output\reports\weibo_growth\weibo_growth_report.tex
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "reports" / "weibo_growth"
FIG_DIR = OUTPUT_DIR / "figures"
VIS_DIR = PROJECT_ROOT / "output" / "visualizations" / "weibo"
LABELS_PATH = PROJECT_ROOT / "output" / "weibo" / "labels" / "pseudo_labels_train.jsonl"
FINAL_DECISIONS_PATH = PROJECT_ROOT / "output" / "models" / "pseudo_label_classifier" / "final_keep_decisions.jsonl"
EVIDENCE_PATH = PROJECT_ROOT / "output" / "weibo" / "evidence" / "evidence_candidates.jsonl"
MODEL_METADATA_PATH = PROJECT_ROOT / "output" / "models" / "pseudo_label_classifier" / "metadata.json"
RAW_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "weibo" / "3666468881_browser_original_clean.jsonl"
LOG_PATH = PROJECT_ROOT / "output" / "logs" / "07_generate_tex_report.log"


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def tex_escape(text: Any) -> str:
    text = str(text or "")
    text = "".join(
        ch
        for ch in text
        if ord(ch) <= 0xFFFF and unicodedata.category(ch) not in {"So", "Cs"}
    )
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def compact_text(text: Any, limit: int = 220) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value if len(value) <= limit else value[:limit].rstrip() + "……"


def format_metric(value: Any) -> str:
    try:
        if value == "":
            return ""
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    font_candidates = [
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttf"),
        Path(r"C:\Windows\Fonts\SimSun.ttc"),
    ]
    selected_font = None
    for font_path in font_candidates:
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            selected_font = FontProperties(fname=str(font_path)).get_name()
            break
    if selected_font:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
        log(f"Using matplotlib CJK font: {selected_font}")
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        log("No CJK font file found for matplotlib; Chinese glyphs may be missing.")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10


def bold_axes(ax: Any) -> None:
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontweight("bold")


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    log(f"Wrote figure: {path}")


def load_data() -> dict[str, Any]:
    data = {
        "raw": pd.DataFrame(read_jsonl(RAW_CLEAN_PATH)),
        "labels": pd.DataFrame(read_jsonl(LABELS_PATH)),
        "final": pd.DataFrame(read_jsonl(FINAL_DECISIONS_PATH)),
        "evidence": pd.DataFrame(read_jsonl(EVIDENCE_PATH)),
        "records": pd.read_csv(VIS_DIR / "visualization_records.csv"),
        "event_density": pd.read_csv(VIS_DIR / "event_density_monthly.csv"),
        "sentiment_monthly": pd.read_csv(VIS_DIR / "sentiment_monthly.csv"),
        "keep_monthly": pd.read_csv(VIS_DIR / "keep_score_monthly.csv"),
        "inner_action": pd.read_csv(VIS_DIR / "inner_action_monthly.csv"),
        "evidence_type": pd.read_csv(VIS_DIR / "evidence_type_monthly.csv"),
        "metadata": read_json(MODEL_METADATA_PATH),
    }
    for key in ("raw", "labels", "final", "evidence", "records"):
        if not data[key].empty and "created_at" in data[key].columns:
            data[key]["created_at_dt"] = pd.to_datetime(data[key]["created_at"], errors="coerce")
            data[key]["year"] = data[key]["created_at_dt"].dt.year
            data[key]["month"] = data[key]["created_at_dt"].dt.strftime("%Y-%m")
    return data


def plot_event_density(data: dict[str, Any]) -> Path:
    df = data["event_density"].copy()
    path = FIG_DIR / "event_density.png"
    plt.figure(figsize=(12, 4.8))
    for col, label in [("posts", "微博总量"), ("long_posts", "长文本"), ("labeled", "已伪标签"), ("kept", "成长证据")]:
        if col in df.columns:
            plt.plot(df["month"], df[col], marker="o", linewidth=1.5, markersize=3, label=label)
    plt.xticks(range(0, len(df), max(1, len(df) // 14)), df["month"].iloc[:: max(1, len(df) // 14)], rotation=45, ha="right")
    plt.title("事件密度与成长证据密度")
    plt.xlabel("月份")
    plt.ylabel("数量")
    plt.legend()
    savefig(path)
    return path


def plot_sentiment(data: dict[str, Any]) -> Path:
    df = data["sentiment_monthly"].copy()
    path = FIG_DIR / "sentiment_timeline.png"
    plt.figure(figsize=(12, 4.8))
    plt.plot(df["month"], df["positive_mean"], marker="o", linewidth=1.6, markersize=3, label="月均正向分")
    plt.plot(df["month"], df["positive_median"], marker="s", linewidth=1.2, markersize=2.5, label="月中位正向分")
    plt.xticks(range(0, len(df), max(1, len(df) // 14)), df["month"].iloc[:: max(1, len(df) // 14)], rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title("情感趋势：中文 RoBERTa 情感分")
    plt.xlabel("月份")
    plt.ylabel("positive score")
    plt.legend()
    savefig(path)
    return path


def plot_keep(data: dict[str, Any]) -> Path:
    df = data["keep_monthly"].copy()
    path = FIG_DIR / "keep_score_timeline.png"
    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    ax1.bar(df["month"], df["kept"], color="#4C78A8", alpha=0.75, label="保留证据数")
    ax1.set_ylabel("保留证据数")
    ax2 = ax1.twinx()
    ax2.plot(df["month"], df["keep_score_mean"], color="#E45756", marker="o", linewidth=1.5, markersize=3, label="平均 keep_score")
    ax2.set_ylabel("平均 keep_score")
    step = max(1, len(df) // 14)
    ax1.set_xticks(range(0, len(df), step))
    ax1.set_xticklabels(df["month"].iloc[::step], rotation=45, ha="right")
    ax1.set_title("成长证据保留趋势")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    savefig(path)
    return path


def plot_evidence_type(data: dict[str, Any]) -> Path:
    df = data["evidence_type"].copy()
    path = FIG_DIR / "evidence_type_top.png"
    total = df.groupby("evidence_type")["count"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(9, 5.2))
    sns.barplot(x=total.values, y=total.index, palette="Blues_r")
    plt.title("成长证据类型 Top 10")
    plt.xlabel("计数")
    plt.ylabel("")
    savefig(path)
    return path


def plot_inner_action(data: dict[str, Any]) -> Path:
    df = data["inner_action"].copy()
    path = FIG_DIR / "inner_action_top.png"
    total = df.groupby("inner_action")["count"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(9, 5.2))
    sns.barplot(x=total.values, y=total.index, palette="Greens_r")
    plt.title("心路动作 Top 10")
    plt.xlabel("计数")
    plt.ylabel("")
    savefig(path)
    return path


def plot_decision_sources(data: dict[str, Any]) -> Path:
    df = data["final"].copy()
    path = FIG_DIR / "decision_sources.png"
    counts = df["final_decision_source"].fillna("unknown").value_counts()
    plt.figure(figsize=(6.5, 4.8))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("最终 keep 决策来源")
    savefig(path)
    return path


def plot_semantic(data: dict[str, Any]) -> Path:
    sem_path = VIS_DIR / "semantic_map_points.csv"
    df = pd.read_csv(sem_path)
    path = FIG_DIR / "semantic_map.png"
    sample = df.copy()
    plt.figure(figsize=(9.5, 7))
    sc = plt.scatter(sample["x"], sample["y"], c=sample["positive_score"], cmap="viridis", s=12, alpha=0.72)
    plt.colorbar(sc, label="positive score")
    plt.title("语义空间地图：BGE embedding + PCA")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    savefig(path)
    return path


THEME_DEFINITIONS = {
    "科研与承担": ["逃避", "不逃避", "论文", "申博", "读博", "申请", "开题", "导师", "小目标"],
    "关系支点": ["边界", "朋友", "见面", "雪瑞", "甜宝", "心安", "孤独", "闺蜜", "袁宝"],
    "城市迁移": ["上海", "魔都", "武汉", "香港", "黄山", "城市", "母校", "离沪", "回珈"],
    "社会结构": ["社会", "课堂", "排名", "竞争", "独生子女", "附近", "市场", "规则", "福利", "阶层"],
    "文学历史": ["晚唐", "历史", "白居易", "金庸", "资治通鉴", "吴道子", "文学", "小说", "人类学"],
    "创作技术": ["剪", "视频", "AE", "创作", "代码", "AI", "技术", "PR", "调色"],
}


THEME_COLORS = {
    "科研与承担": "#4C78A8",
    "关系支点": "#F58518",
    "城市迁移": "#54A24B",
    "社会结构": "#B279A2",
    "文学历史": "#9D755D",
    "创作技术": "#E45756",
}


def assign_theme(text: Any) -> str:
    value = str(text or "")
    scores = {
        theme: sum(1 for term in terms if term in value)
        for theme, terms in THEME_DEFINITIONS.items()
    }
    best_theme, best_score = max(scores.items(), key=lambda item: item[1])
    return best_theme if best_score > 0 else "日常自我叙事"


def add_evidence_themes(data: dict[str, Any]) -> pd.DataFrame:
    df = data["evidence"].copy()
    if df.empty:
        return df
    df["theme"] = df["text"].fillna("").map(assign_theme)
    dt = pd.to_datetime(df["created_at"], errors="coerce")
    df["year"] = dt.dt.year
    df["period"] = pd.cut(
        df["year"],
        bins=[2012, 2019, 2022, 2024, 2026],
        labels=["2013-2019", "2020-2022", "2023-2024", "2025-2026"],
    )
    return df


def plot_theme_timeline(data: dict[str, Any]) -> Path:
    df = add_evidence_themes(data)
    path = FIG_DIR / "growth_theme_timeline.png"
    theme_order = list(THEME_DEFINITIONS)
    yearly = (
        df[df["theme"].isin(theme_order)]
        .pivot_table(index="year", columns="theme", values="id", aggfunc="count", fill_value=0)
        .reindex(columns=theme_order, fill_value=0)
        .sort_index()
    )
    ax = yearly.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 5.8),
        color=[THEME_COLORS[t] for t in theme_order],
        width=0.82,
    )
    ax.set_title("成长主线证据的年度分布")
    ax.set_xlabel("年份")
    ax.set_ylabel("证据条数")
    ax.legend(ncol=3, fontsize=9, frameon=True)
    bold_axes(ax)
    savefig(path)
    return path


def plot_phase_theme_heatmap(data: dict[str, Any]) -> Path:
    df = add_evidence_themes(data)
    path = FIG_DIR / "growth_phase_heatmap.png"
    theme_order = list(THEME_DEFINITIONS)
    heat = (
        df[df["theme"].isin(theme_order)]
        .pivot_table(index="theme", columns="period", values="id", aggfunc="count", fill_value=0, observed=False)
        .reindex(index=theme_order)
    )
    plt.figure(figsize=(9.5, 5.4))
    ax = sns.heatmap(
        heat,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "证据条数"},
        annot_kws={"fontweight": "bold"},
    )
    plt.title("不同阶段的主线重心")
    plt.xlabel("阶段")
    plt.ylabel("")
    bold_axes(ax)
    if ax.collections and ax.collections[0].colorbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_fontweight("bold")
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight("bold")
    savefig(path)
    return path


def plot_key_turning_points(data: dict[str, Any]) -> Path:
    df = data["evidence"].copy()
    path = FIG_DIR / "growth_turning_points.png"
    selected_ids = [
        "4392795365810121",
        "4315283336435542",
        "4522779103974084",
        "4976488422310362",
        "5065461006602574",
        "5237025907409200",
        "5312375302392111",
    ]
    labels = {
        "4392795365810121": "逃避会缩小选择",
        "4315283336435542": "课堂/竞争进入自我反思",
        "4522779103974084": "毕业视频与勇气",
        "4976488422310362": "不逃避和亲力亲为",
        "5065461006602574": "离沪与城市老友",
        "5237025907409200": "gap year 的结构反思",
        "5312375302392111": "多建立一些支点",
    }
    rows = df[df["id"].isin(selected_ids)].copy()
    rows["created_at_dt"] = pd.to_datetime(rows["created_at"], errors="coerce")
    rows = rows.sort_values("created_at_dt")
    plt.figure(figsize=(12, 3.8))
    y = [1] * len(rows)
    colors = [THEME_COLORS.get(assign_theme(text), "#777777") for text in rows["text"]]
    plt.scatter(rows["created_at_dt"], y, s=115, color=colors, zorder=3)
    plt.hlines(1, rows["created_at_dt"].min(), rows["created_at_dt"].max(), color="#BBBBBB", linewidth=2, zorder=1)
    for idx, row in enumerate(rows.to_dict("records")):
        offset = 0.16 if idx % 2 == 0 else -0.22
        va = "bottom" if offset > 0 else "top"
        plt.text(
            row["created_at_dt"],
            1 + offset,
            labels.get(row["id"], ""),
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )
    plt.yticks([])
    plt.ylim(0.45, 1.55)
    plt.title("关键转折证据节点")
    plt.xlabel("时间")
    plt.gca().spines[["left", "right", "top"]].set_visible(False)
    bold_axes(plt.gca())
    savefig(path)
    return path


def plot_narrative_transition_blueprint(data: dict[str, Any]) -> Path:
    path = FIG_DIR / "growth_narrative_blueprint.png"
    nodes = [
        ("2018-2019", "竞争经验\n进入自我反思", "评价系统 -> 自我要求"),
        ("2019", "逃避后果\n转为行动边界", "焦虑 -> 小目标"),
        ("2020-2022", "创作和关系\n分流压力", "单一压力 -> 复数支点"),
        ("2023-2024", "亲历和承担\n形成道路语法", "等待 -> 进入现场"),
        ("2024", "城市离别\n改写身份容器", "学校节点 -> 地方依恋"),
        ("2025-2026", "结构意识\n扩展选择理解", "个人努力 -> 社会现场"),
    ]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_axis_off()
    xs = [0.08, 0.25, 0.42, 0.59, 0.76, 0.92]
    y = 0.58
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D", "#E45756"]
    for idx, ((period, title, shift), x, color) in enumerate(zip(nodes, xs, colors)):
        ax.text(
            x,
            y,
            f"{period}\n{title}\n{shift}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            linespacing=1.45,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.45,rounding_size=0.08",
                "facecolor": "#FFFFFF",
                "edgecolor": color,
                "linewidth": 2.2,
            },
            transform=ax.transAxes,
        )
        if idx < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.075, y),
                xytext=(x + 0.075, y),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#666666"},
            )
    ax.text(
        0.5,
        0.18,
        "叙事蓝图：主题、行动者、意图和隐含行动召唤随节点迁移",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )
    savefig(path)
    return path


def top_examples(data: dict[str, Any], terms: list[str], limit: int = 3) -> list[dict[str, Any]]:
    df = data["evidence"].copy()
    if df.empty:
        df = data["labels"].copy()
    if df.empty:
        return []
    mask = pd.Series(False, index=df.index)
    for term in terms:
        mask = mask | df["text"].fillna("").str.contains(term, regex=False)
    selected = df[mask].copy()
    if "pred_keep_probability" in selected.columns:
        selected = selected.sort_values("pred_keep_probability", ascending=False)
    elif "keep_score" in selected.columns:
        selected = selected.sort_values("keep_score", ascending=False)
    return selected.head(limit).to_dict("records")


def representative_quotes(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "科研、申请与承担责任": top_examples(data, ["逃避", "不逃避", "论文", "申博", "读博", "申请", "开题"], 6),
        "关系、边界与支点系统": top_examples(data, ["边界", "朋友", "见面", "雪瑞", "甜宝", "心安", "孤独"], 6),
        "城市、迁移与地方依恋": top_examples(data, ["上海", "魔都", "武汉", "香港", "黄山", "城市", "母校"], 6),
        "社会结构与个人道路": top_examples(data, ["社会", "课堂", "排名", "竞争", "独生子女", "附近", "市场", "规则", "福利"], 6),
        "文学、历史与价值秩序": top_examples(data, ["晚唐", "历史", "白居易", "金庸", "资治通鉴", "吴道子", "文学", "人类学"], 6),
        "创作、技术与能量管理": top_examples(data, ["剪", "视频", "AE", "创作", "代码", "AI", "技术", "本职", "勤勉"], 6),
    }


def summary_numbers(data: dict[str, Any]) -> dict[str, Any]:
    raw = data["raw"]
    labels = data["labels"]
    final = data["final"]
    evidence = data["evidence"]
    records = data["records"]
    start = pd.to_datetime(raw["created_at"], errors="coerce").min()
    end = pd.to_datetime(raw["created_at"], errors="coerce").max()
    final_sources = final["final_decision_source"].fillna("unknown").value_counts().to_dict() if not final.empty else {}
    final_keep = final["final_keep_label"].fillna(False).astype(bool).value_counts().to_dict() if not final.empty else {}
    return {
        "raw_count": len(raw),
        "label_count": len(labels),
        "final_count": len(final),
        "evidence_count": len(evidence),
        "start": start.strftime("%Y-%m-%d") if not pd.isna(start) else "",
        "end": end.strftime("%Y-%m-%d") if not pd.isna(end) else "",
        "final_sources": final_sources,
        "final_keep": final_keep,
        "sentiment_mean": records["positive_score"].mean() if "positive_score" in records else math.nan,
        "sentiment_std": records["positive_score"].std() if "positive_score" in records else math.nan,
    }


def latex_quote_block(title: str, rows: list[dict[str, Any]]) -> str:
    parts = [rf"\subsection{{{tex_escape(title)}}}"]
    if not rows:
        parts.append("暂无足够证据。")
        return "\n".join(parts)
    for row in rows:
        date = row.get("created_at") or row.get("date") or ""
        text = compact_text(row.get("text"), 260)
        parts.append(r"\begin{quote}")
        parts.append(rf"\textbf{{{tex_escape(date)}}}\\")
        parts.append(tex_escape(text))
        parts.append(r"\end{quote}")
    return "\n".join(parts)


def markdown_to_plain(text: Any) -> str:
    """Strip Markdown markers before embedding generated text in TeX."""
    value = str(text or "")
    value = re.sub(r"<!--.*?-->", "", value, flags=re.S)
    value = re.sub(r"```.*?```", "", value, flags=re.S)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"\*([^*]+)\*", r"\1", value)
    value = re.sub(r"^#{1,6}\s*", "", value, flags=re.M)
    value = re.sub(r"^\s*[-*]\s+", "", value, flags=re.M)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def evidence_table(rows: list[dict[str, Any]], limit: int = 3) -> str:
    parts = []
    for row in rows[:limit]:
        date = row.get("created_at") or row.get("date") or ""
        reason = row.get("final_decision_reason", "")
        text = compact_text(row.get("text"), 210)
        human_reason = compact_text(reason, 80)
        if re.search(r"(BERT|BGE|classifier|confidence|threshold|keep_score|pred_keep)", human_reason, re.I):
            human_reason = ""
        parts.append(r"\begin{quote}")
        if human_reason:
            parts.append(rf"\textbf{{{tex_escape(date)}}} \quad {tex_escape(human_reason)}\\")
        else:
            parts.append(rf"\textbf{{{tex_escape(date)}}}\\")
        parts.append(tex_escape(text))
        parts.append(r"\end{quote}")
    return "\n".join(parts)


def section_block(title: str, claim: str, rows: list[dict[str, Any]], interpretation: str, counter: str, observe: str) -> str:
    return rf"""
\subsection{{{tex_escape(title)}}}
\textbf{{判断：}}{tex_escape(claim)}

\textbf{{证据：}}
{evidence_table(rows, 3)}

\textbf{{解释：}}{tex_escape(interpretation)}

\textbf{{反解释：}}{tex_escape(counter)}

\textbf{{后续观察：}}{tex_escape(observe)}
"""


def transition_block(title: str, nodes: str, analysis: str, counter: str, observe: str) -> str:
    return rf"""
\subsection{{{tex_escape(title)}}}
{tex_escape(nodes)}

{tex_escape(analysis)}

\textbf{{反解释：}}{tex_escape(counter)}

\textbf{{后续观察：}}{tex_escape(observe)}
"""


def build_social_psychology_sections(quotes: dict[str, list[dict[str, Any]]]) -> str:
    blocks = [
        section_block(
            "从逃避到亲历：责任感不是口号，而是行动边界的扩大",
            "04 证据包里反复出现的不是单纯的拖延，而是先用回避降低压力，再意识到回避会缩小选择空间，最后把任务拆回可执行的小目标。",
            quotes["科研、申请与承担责任"],
            "用认知行为取向看，这是一条从情绪回避到问题聚焦应对的路径。2019 年的表述强调逃避的后果，2023 年以后开始强调亲力亲为、不找中介、不强求结果、每天做好眼前事。变化的关键不是变得完全不焦虑，而是在焦虑存在时仍能增加信息搜集、申请、联系导师、修正论文等具体动作。",
            "这些文本也可能只是阶段性自我鼓励，不能证明稳定人格变化。论文、申请和读博本身就是高压任务，回避并不必然说明能力不足，也可能是资源、信息和支持不足时的合理暂停。",
            "继续记录重大任务中是否出现更早的信息搜集、更清楚的时间边界，以及暂停后能否回到行动。"
        ),
        section_block(
            "关系从救援想象转向支点网络",
            "关系文本的重点不是依赖某个对象，而是不断学习如何把朋友、母亲、同伴、互联网好友转化为不同强度的支点。",
            quotes["关系、边界与支点系统"],
            "从依恋理论的温和视角看，文本中有被看见、被陪伴、被理解的需求，也有边界和分寸感。2025 到 2026 年的证据更接近支点网络：不要求每段关系承担全部情绪功能，而是让交流推动思考，让旧友提供心安，让互联网关系逐步进入现实生活。这比把某个人神化成救援者更稳定。",
            "关系记述往往发生在离别、重逢或孤独时刻，情绪浓度会被放大。不能据此推断日常关系模式，也不能把朋友关系解释成固定依恋类型。",
            "观察后续文本里支点是否更多元，独处时是否仍能维持节奏，以及关系表达是否能同时容纳亲密和现实边界。"
        ),
        section_block(
            "地方依恋：城市不是背景，而是身份转换的容器",
            "上海、黄山、香港、武汉等地点在证据包中不是旅游坐标，而是承载不同阶段自我感的地方装置。",
            quotes["城市、迁移与地方依恋"],
            "环境心理学和地方依恋可以解释这条线。上海承载大学、创作、朋友和最初职业想象；黄山连接起点和身体性的熟悉；香港带来远距离吸引和耗电感；武汉则在回珈叙事中提供重新落地的感觉。迁移带来的不是简单怀旧，而是重新分配安全感、自由感和未来感。",
            "城市叙事也可能受具体事件影响，例如毕业、见友、旅行、天气和居住条件。不能把某座城市固定解释为唯一归属。",
            "继续观察新地点是否从陌生耗电转为可行动空间，以及旧地点是否从执念变成可携带的记忆资源。"
        ),
        section_block(
            "社会学视角：个人困境被放回制度和代际结构中理解",
            "较有力量的部分在于，文本不只说我难受，而是把压力放进课堂排名、教育竞争、社会流动、市场供需、独生子女和附近等结构里重读。",
            quotes["社会结构与个人道路"],
            "这条线需要社会学而不只是心理学。2018 年课堂排名与竞争的文本说明外部评价如何进入自我要求；2020 年教育生态和阶层固化的观察把个人焦虑放进资源分配；2025 年 gap year 的反思直接写到个人成长需要被市场供需和社会价值判断淹没；2026 年关于杀马特、附近、独生子女和互联网联结的文本，则把孤独、自我保护和兴趣共同体放在代际处境中理解。",
            "结构解释不能替代个人行动。把困难完全归因于社会系统，可能会削弱对可控变量的辨认。但证据中同时有小目标、本职工作、勤勉自洽等行动取向，因此更像结构意识和行动意识并存。",
            "后续可观察文本是否能继续在结构理解和具体行动之间来回切换，而不是停留在宏大归因。"
        ),
        section_block(
            "历史和文学不是装饰，而是意义建构的核心工具",
            "文学、历史、剧评和人物小纪在证据包中承担了价值建模功能，帮助把个人经验放进更长的时间尺度。",
            quotes["文学、历史与价值秩序"],
            "叙事心理学中的 meaning-making 在这里很明显。白居易、晚唐、金庸、资治通鉴、女性角色中的义爱礼，并不是单纯兴趣清单，而是用来组织价值秩序的材料：何为克制，何为悲悯，何为风骨，何为孤独真实，何为在历史长河中保护心力。它们让现实挫折不只停在情绪层面，而能被改写成道路、尺度和风格。",
            "文学化表达有时会美化痛苦，也可能让现实问题延后处理。尤其长篇历史叙事未必都直接对应个人成长，04 证据包也保留了部分 final_keep=False 的反例，说明不是所有好文本都应进入复盘。",
            "后续可区分三类文本：纯审美文本、借审美调节情绪的文本、把审美转化为行动原则的文本。第三类最值得复盘。"
        ),
        section_block(
            "创作和技术：能量管理比兴趣标签更关键",
            "剪辑、写作、AE、AI 和技术观察反复出现，真正重要的是它们暴露了不同任务对心力的消耗和补给方式。",
            quotes["创作、技术与能量管理"],
            "从自我决定理论看，创作活动给出更强的自主感、胜任感和即时反馈。证据里多次出现学习或论文低能量、剪视频高专注的对比；后期又出现 AE tips、AI 灵魂问题、机器人避障方法等技术反思。这说明技能线不只是逃避学业，也可能是恢复能量、建立方法感、形成研究直觉的入口。",
            "创作高能量不等于所有创作都应优先于科研任务。它也可能成为压力下的替代性满足。关键不在于压制创作，而在于把创作里的流程感、反馈感和完成感迁移到论文、课程和研究任务中。",
            "继续观察创作活动是否带来可复用的方法，例如脚本、模板、素材管理、定期产出和研究问题转化。"
        ),
    ]
    return "\n".join(blocks)


def build_formal_skill_sections(quotes: dict[str, list[dict[str, Any]]]) -> str:
    blocks = [
        transition_block(
            "第一节点：竞争经验进入自我反思",
            "2018 到 2019 年的节点，核心变化不是“变得焦虑”，而是开始把课堂、排名、虚荣、接纳这些经验放到同一个叙事框架里。竞争不再只是外部评价，而成为理解自我要求和他人目光的入口。",
            "叙事蓝图中，这一阶段的主要 actors 是课堂、同辈、排名和被看见的自己。叙事 intent 是解释为什么外部评价会牵动自我感；隐含的 call to action 还比较温和，更多是看清和接纳，而不是立即改变。这个节点为后面的责任感转向打下基础：只有先意识到评价系统如何进入自我叙事，后面才可能区分“我真正要做什么”和“我只是在回应评价”。",
            "课堂和竞争文本也可能只是青春期或校园环境中的常见感受，不必然构成长线转折。它的解释力来自后续文本不断回到选择、道路和行动边界，而不是来自单条表达本身。",
            "后续应看评价压力是否仍被写成唯一尺度，还是逐步被科研、创作、城市和关系等更多尺度稀释。"
        ),
        transition_block(
            "第二节点：从逃避后果到行动边界",
            "2019 年“逃避会缩小选择”的表达，把问题从情绪层面推进到行动层面。此前压力主要来自评价和不确定；到这一节点，叙事开始承认回避本身会改变未来可选项。",
            "这一变化很关键。narrative intent 从安顿情绪转为恢复主动性，key claim 也从“我很难受”变成“如果继续回避，选择空间会变小”。心理学上可以理解为认知行为取向中的问题重构：焦虑不是被否认，而是被转译成任务管理问题。后面关于论文、申请、读博和工作的文本，基本都沿着这条线展开：不是要求自己立刻强大，而是把不确定拆成阶段、小目标和亲历过程。",
            "这一节点仍可能只是清醒时刻，并不保证行动持续。成长报告不能把一次反思直接写成人格变化。",
            "后续判断要看它是否转化为更早收集信息、更具体拆分任务、更愿意面对反馈。"
        ),
        transition_block(
            "第三节点：毕业、创作和关系把压力重新分流",
            "2020 到 2022 年，文本不再只围绕学业压力打转。毕业视频、创作、关系温暖、边界感和生活细节一起出现，说明叙事的支点开始分散。",
            "这个阶段的 narrative actors 变多了：作品、朋友、家人、剪辑流程、城市日常都进入叙事。它们共同完成一件事，把原本集中在成绩或论文上的压力分流到可感知的连接和可完成的作品中。从自我决定理论看，创作提供胜任感，关系提供稳定感，生活细节提供连续感。这里不是“靠兴趣逃避学业”，更像是在给高压任务之外建立恢复系统。",
            "创作和关系也可能成为短期转移注意力的方式。如果它们没有反哺现实任务，只提供片刻兴奋或安慰，就不能被解释为稳定支点。",
            "后续应看这些支点是否在重大转换期仍有效，尤其是读博申请、离开城市和 gap year 等节点。"
        ),
        transition_block(
            "第四节点：不逃避和亲力亲为成为新的道路语法",
            "2023 到 2024 年，叙事中的核心词从安慰、等待和喜欢，转向不逃避、亲力亲为、申请、读博、论文、能力和未来规划。这里出现了更明确的道路语法。",
            "叙事蓝图下，这一节点的 intent 是把人生阶段重新组织成可执行路径。actors 也发生变化：导师、论文、申请制度、城市、视频作品和自我能力共同进入同一张图。相比前一阶段的支点分流，这一阶段更强调进入现场。换句话说，支点不只是为了让人舒服一点，而是为了让人能回到任务、承担选择、接受结果的不完全可控。",
            "这也可能带来新的风险：责任感被写得太满时，容易变成对自己的过度要求。报告不能把“亲力亲为”浪漫化。",
            "后续应观察责任感是否和边界感一起出现，例如知道什么时候求助、什么时候停下、什么时候降低强求。"
        ),
        transition_block(
            "第五节点：城市离别把身份从学校叙事中移出",
            "2024 年离沪和城市书写，是一个明显的空间转向。城市不再只是事件发生地，而被写成老友、容器和见证者。",
            "地方依恋视角能解释这个节点：当学校、论文或申请无法单独承载身份时，城市提供另一种连续性。它让个人不只属于某个评价系统，也属于走过的街区、关系、气味、季节和记忆。这个变化使叙事从“我要完成什么”扩展为“我曾经在哪里生活过，又如何带着这些经验进入下一阶段”。",
            "城市叙事容易被怀旧放大。把城市写成老友，不等于现实支持系统已经稳定，也不能替代对新环境的实际建设。",
            "后续应看迁移之后是否形成新的生活路线、关系连接和行动节奏。若只有回望，没有新连接，地方依恋会停留在怀旧。"
        ),
        transition_block(
            "第六节点：gap year 和结构意识把个人道路放回社会现场",
            "2025 到 2026 年，gap year、市场供需、社会价值判断、历史长河和“多建立一些支点”集中出现。叙事从个人努力进一步转向结构理解。",
            "这不是放弃个人责任，而是把责任放回现实条件里理解。社会学视角下，教育竞争、市场评价、代际期待和城市流动共同塑造选择空间。这个节点的 key claim 可以概括为：个人不能只靠意志解释处境，但仍需要在结构中建立可行动的支点。与 2019 年相比，这里的“选择空间”已经不只是逃避造成的个人后果，也包括制度和市场给出的外部边界。",
            "结构意识如果过度使用，也可能滑向无力感，把所有困难都解释成外部问题。成长报告需要同时保留行动线和结构线。",
            "后续最值得观察的是支点系统是否真正复数化：学业、工作、朋友、城市、创作、历史阅读和身体节奏能否各自承担有限功能，而不是由单一目标承担全部意义。"
        ),
    ]
    return "\n".join(blocks)


def build_tex(data: dict[str, Any], figures: dict[str, Path]) -> str:
    nums = summary_numbers(data)
    quotes = representative_quotes(data)
    fig = {key: value.relative_to(OUTPUT_DIR).as_posix() for key, value in figures.items()}

    social_psychology_sections = build_formal_skill_sections(quotes)

    return rf"""
\documentclass[UTF8,zihao=-4]{{ctexart}}
\usepackage[a4paper,margin=2.2cm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{enumitem}}
\usepackage{{setspace}}
\usepackage{{indentfirst}}
\usepackage{{caption}}
\usepackage{{fancyhdr}}
\hypersetup{{hidelinks}}
\setlist{{nosep,leftmargin=2em}}
\setstretch{{1.16}}
\setlength{{\parindent}}{{2em}}
\setlength{{\parskip}}{{0.25em}}
\setlength{{\headheight}}{{15pt}}
\setmainfont{{Times New Roman}}
\setCJKmainfont[AutoFakeBold=2.5,AutoFakeSlant=0.2]{{Microsoft YaHei}}
\setCJKsansfont[AutoFakeBold=2.5,AutoFakeSlant=0.2]{{Microsoft YaHei}}
\setCJKmonofont{{Microsoft YaHei}}
\captionsetup{{font=small,labelfont=bf,textfont=bf,skip=8pt}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{个人成长轨迹分析报告}}
\fancyhead[R]{{基于 04 证据包}}
\fancyfoot[C]{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\ctexset{{
  section={{format=\centering\zihao{{3}}\bfseries,beforeskip=1.6em,afterskip=1.0em}},
  subsection={{format=\zihao{{4}}\bfseries,beforeskip=1.1em,afterskip=0.55em}}
}}
\title{{个人成长轨迹分析报告}}
\author{{基于公开微博证据包}}
\date{{{datetime.now():%Y-%m-%d}}}

\begin{{document}}
\maketitle
\tableofcontents
\newpage

\section{{摘要}}
这是一份面向用户复盘的成长分析报告。报告使用非诊断性的心理学与社会学解释框架，关注“文本如何组织经验、关系、城市、学业、创作和社会处境”，不根据单条表达判断人格或心理问题。

本报告回读 {tex_escape(nums['start'])} 至 {tex_escape(nums['end'])} 的公开微博材料，以 04 生成的 {nums['evidence_count']} 条候选证据为主要依据。主线判断集中在六个方面：科研与承担、关系支点、城市迁移、社会结构、文学历史、创作技术。正文由 04 证据包、三张主线可视化和本仓库 prompts 共同约束，不调用 05 解释脚本生成正文。

\section{{成长主线总览}}
\subsection{{主线证据如何分布}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\linewidth]{{{fig['theme_timeline']}}}
\caption{{成长主线证据的年度分布}}
\end{{figure}}

这张图展示成长叙事的重心如何随年份移动。早期材料更多呈现审美兴趣、关系温暖和自我鼓励；2019 到 2020 年开始出现论文、毕业、创作和情绪调节的交织；2023 以后，读博、城市迁移、gap year、社会结构意识和支点系统成为更清晰的主线。

\subsection{{不同阶段的重心变化}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\linewidth]{{{fig['phase_heatmap']}}}
\caption{{不同阶段的主线重心}}
\end{{figure}}

阶段热力图显示，成长不是单线条地“越来越好”，而是多个主题反复回潮：关系和城市提供情感定位，科研与创作提供行动压力和能力线索，文学历史提供意义框架，社会结构意识则让个人困境不再只被理解成个人失败。

\subsection{{关键转折节点}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\linewidth]{{{fig['turning_points']}}}
\caption{{关键转折证据节点}}
\end{{figure}}

这些节点不是“人生结论”，而是复盘入口。它们共同指向一个变化：从把压力体验为模糊的焦虑，逐渐转向把压力拆解为可行动的小目标、可解释的社会结构、可携带的关系支点，以及可持续的意义资源。

\subsection{{叙事节点如何连续推进}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\linewidth]{{{fig['narrative_blueprint']}}}
\caption{{成长叙事的节点变迁蓝图}}
\end{{figure}}

这张图把前面的统计分布转为叙事路线。报告后文不再按证据类别展开，而是按节点推进：每个节点都说明上一阶段的问题如何被重新组织，新的叙事对象是什么，隐含的下一步行动是什么。

\section{{节点变迁分析}}
本节合并使用四类分析提示：成长档案要求判断绑定证据并保留反解释；叙事自我关注“如何讲述自己”；心理反思只使用非诊断框架；narrative blueprint 则把每个节点拆成主题、行动者、叙事意图、隐含行动召唤和关键主张。正文不罗列长证据，而是解释节点之间如何迁移。

{social_psychology_sections}

\section{{结论与后续观察}}
\begin{{enumerate}}
  \item 后续最值得观察的不是情绪分数，而是“支点系统”是否继续扩展：交流、城市、创作、学业和历史阅读能否各自承担有限但稳定的功能。
  \item 科研和论文线应关注行动颗粒度：是否更早搜集信息、拆分任务、降低结果执念，并把创作中的流程感迁移到研究任务里。
  \item 文学历史线应继续区分纯审美、情绪调节和行动原则。能转化为现实判断的文本，才是成长报告中的强证据。
  \item 社会结构意识是这份材料的深层价值。它让个人困境不被简化为个人失败，但仍需要和可控行动相互校正。
  \item 这份报告不追求给出最终人格画像，而是保留一张可继续更新的成长地图。
\end{{enumerate}}

\end{{document}}
"""


def main() -> None:
    ensure_dirs()
    setup_plot_style()
    data = load_data()
    figures = {
        "theme_timeline": plot_theme_timeline(data),
        "phase_heatmap": plot_phase_theme_heatmap(data),
        "turning_points": plot_key_turning_points(data),
        "narrative_blueprint": plot_narrative_transition_blueprint(data),
    }
    tex = build_tex(data, figures)
    tex_path = OUTPUT_DIR / "weibo_growth_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    log(f"Wrote TeX report: {tex_path}")


if __name__ == "__main__":
    main()
