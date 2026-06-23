# -*- coding: utf-8 -*-
r"""Use Gemini to generate evidence-constrained growth archive interpretations.

环境变量设置（PowerShell）：

1. 设置 Gemini API Key：
   $env:GEMINI_API_KEY="你的 Google AI Studio API Key"
   $env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"

常用命令：

1. 基于本地模型证据包测试 prompt，不真正调用 API：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\05_gemini_growth_interpreter.py --scope reranked-evidence --dry-run

2. 基于本地模型证据包调用 Gemini 生成解释报告：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\05_gemini_growth_interpreter.py --scope reranked-evidence

3. 如后续重新生成月度证据文件，可批量生成最近 3 个月的解释报告：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\05_gemini_growth_interpreter.py --scope monthly --limit 3

4. 如后续重新生成成长总图证据文件，可生成成长总图解释层：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\05_gemini_growth_interpreter.py --scope growth-map

5. 如后续重新生成转折点证据文件，可生成转折点解释层：
   python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\05_gemini_growth_interpreter.py --scope turning-points

说明：
   这个脚本不重新清洗微博，也不调用 CPED。
   它读取 output/weibo/evidence/ 中由本地伪标签分类器生成的证据包，
   再调用 Gemini 生成“有证据、有反解释、不诊断”的解释报告。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "output" / "weibo"
DEFAULT_PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_OUTPUT_DIR = DEFAULT_REPORT_DIR / "gemini"
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
LOG_PATH = PROJECT_ROOT / "output" / "logs" / "05_gemini_growth_interpreter.log"


class GeminiRateLimitError(RuntimeError):
    """Raised when Gemini returns a quota/rate-limit response and fallback can be used."""


SYSTEM_INSTRUCTION = """你是一个成长轨迹档案解释器，不是心理诊断器。

硬性约束：
1. 不做临床诊断。
2. 不根据单条文本判断人格。
3. 不输出“用户是某种依恋类型”这类结论。
4. 所有解释必须绑定原文证据。
5. 每个强解释必须给出反解释。
6. 优先分析阶段性模式，而不是标签化用户。
7. 诗性表达不能直接等同于心理事实。
8. 关系对象不能被神化、病理化或工具化。
9. 输出目标是帮助复盘，不是制造焦虑。
10. 如果证据不足，必须明确说“证据不足”。
"""


REPORT_PROMPT = """请基于下面的规则生成解释报告。

你会收到一份由本地伪标签分类器生成的证据包。它不是最终结论，只是材料。
请输出 Markdown，结构如下：

# 解释报告

## 1. 阶段主问题
用 2-4 条说明这个阶段最反复出现的问题。每条必须引用证据。

## 2. 主要支点
分析哪些人、地点、作品、目标、技能或行动承担了支点功能。

## 3. 情绪背景
不要只说正向/负向，要解释情绪与事件、等待、关系、创作、科研等对象的关系。

## 4. 心路动作
分析文本中更像“困惑、安慰、重估、行动、告别、重启、接受、建立支点”的哪些过程。

## 5. 自我叙事变化
只在有连续证据时写；证据不足时不要强行总结。

## 6. 心理学解释层
只做温和解释。可以使用意义重构、情绪调节、认知模式、关系边界、价值动机等框架。

## 7. 可能风险与反解释
每个风险提醒必须包含：
- 可能风险
- 证据
- 反解释
- 温和建议

## 8. 可继续观察的问题
列出后续值得观察的 3-6 个问题。

要求：
- 不要夸赞式总结。
- 不要心理诊断。
- 不要把材料讲得太满。
- 不要编造原文没有的信息。
"""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def gemini_generate(prompt: str, args: argparse.Namespace) -> str:
    api_key = get_env("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set. Set it in PowerShell first.")

    base = get_env("GEMINI_API_BASE", DEFAULT_GEMINI_API_BASE).rstrip("/")
    model = get_env("GEMINI_MODEL", args.model).strip()
    url = f"{base}/v1beta/models/{model}:generateContent"
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}],
        },
        "contents": [
            {
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": args.temperature,
            "maxOutputTokens": args.max_output_tokens,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    for attempt in range(1, args.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            return extract_text(response_data)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                raise GeminiRateLimitError(f"Gemini rate limit: HTTP 429 Too Many Requests. {body[:300]}") from exc
            if attempt >= args.retries:
                raise RuntimeError(f"Gemini API HTTP {exc.code}: {body}") from exc
            time.sleep(args.retry_delay * attempt)
        except urllib.error.URLError as exc:
            if attempt >= args.retries:
                raise RuntimeError(f"Gemini API request failed: {exc}") from exc
            time.sleep(args.retry_delay * attempt)
    raise RuntimeError("Gemini API request failed after retries.")


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
                return json.dumps(data, ensure_ascii=False, indent=2)
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


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


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


def extract_text(response_data: dict[str, Any]) -> str:
    candidates = response_data.get("candidates") or []
    if not candidates:
        return json.dumps(response_data, ensure_ascii=False, indent=2)
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    texts = [str(part.get("text") or "") for part in parts if part.get("text")]
    return "\n".join(texts).strip()


def build_prompt(title: str, evidence_markdown: str, extra_context: str = "") -> str:
    pieces = [
        REPORT_PROMPT,
        "",
        f"# 待解释材料：{title}",
        "",
        "## 额外上下文",
        extra_context.strip() or "无",
        "",
        "## 证据索引",
        evidence_markdown.strip(),
    ]
    return "\n".join(pieces)


def monthly_files(report_dir: Path, period: str = "", limit: int = 0) -> list[Path]:
    files = sorted((report_dir / "monthly").glob("*.md"))
    if period:
        files = [path for path in files if path.stem == period]
    if limit:
        files = files[-limit:]
    return files


def run_one(title: str, evidence: str, output_path: Path, args: argparse.Namespace, extra_context: str = "") -> None:
    prompt = build_prompt(title, evidence, extra_context)
    if args.dry_run:
        write_text(output_path.with_suffix(".prompt.md"), prompt)
        log(f"Wrote dry-run prompt: {output_path.with_suffix('.prompt.md')}")
        return
    validate_gemini_key(args)
    result, provider, model = generate_with_fallback(prompt, args)
    write_text(output_path, f"<!-- model: {provider}/{model} -->\n\n{result}")
    log(f"Wrote report: {output_path}; model={provider}/{model}")


def run_monthly(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir).resolve()
    output_dir = Path(args.output_dir).resolve() / "monthly"
    files = monthly_files(report_dir, args.period, args.limit)
    if not files:
        raise SystemExit("No monthly files matched.")
    extra_context = load_prompt_context(args)
    for path in files:
        title = f"月度报告 {path.stem}"
        output_path = output_dir / f"{path.stem}.md"
        run_one(title, read_text(path), output_path, args, extra_context)
        if not args.dry_run and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


def run_growth_map(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    path = report_dir / "growth_map.md"
    run_one("成长总图解释层", read_text(path), output_dir / "growth_map_interpretation.md", args, load_prompt_context(args))


def run_turning_points(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    path = report_dir / "turning_points.md"
    run_one("转折点解释层", read_text(path), output_dir / "turning_points_interpretation.md", args, load_prompt_context(args))


def run_reranked_evidence(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    candidates = [
        report_dir / "evidence" / "evidence_reranked.md",
        report_dir / "evidence" / "evidence_pack.md",
    ]
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise SystemExit(
            "Missing evidence file. Run 04_build_evidence_pack.py first.\n"
            + "\n".join(str(item) for item in candidates)
        )
    title = "成长证据包解释层"
    if path.name == "evidence_reranked.md":
        title = "Gemini rerank 后的成长证据包解释层"
    run_one(title, read_text(path), output_dir / "reranked_evidence_interpretation.md", args, load_prompt_context(args))


def load_prompt_context(args: argparse.Namespace) -> str:
    prompts_dir = Path(args.prompts_dir).resolve()
    parts = []
    for name in ("growth_archive.md", "narrative_self.md", "psychological_reflection.md", "narrative_blueprint.md"):
        path = prompts_dir / name
        if path.is_file():
            parts.append(f"## {name}\n{read_text(path)}")
    return "\n\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Gemini-based interpretations for the growth archive.")
    parser.add_argument("--scope", choices=["monthly", "growth-map", "turning-points", "reranked-evidence"], default="reranked-evidence")
    parser.add_argument("--period", default="", help="Monthly period like 2026-06")
    parser.add_argument("--limit", type=int, default=0, help="For monthly scope, process the latest N months")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--prompts-dir", default=str(DEFAULT_PROMPTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.scope == "monthly":
        run_monthly(args)
    elif args.scope == "growth-map":
        run_growth_map(args)
    elif args.scope == "turning-points":
        run_turning_points(args)
    elif args.scope == "reranked-evidence":
        run_reranked_evidence(args)
    else:
        raise SystemExit(f"Unsupported scope: {args.scope}")


if __name__ == "__main__":
    main()
