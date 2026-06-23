# 自我复盘系统

这个项目用于从个人微博文本中构建成长轨迹档案。它不是心理诊断系统，也不是简单的 positive / negative 情绪分类器。目标是用时间序列、伪标签、本地筛选、LLM 解释和可视化，辅助观察：

- 情绪状态与情绪波动
- 成长证据与心路动作
- 文学/历史/作品解读中的价值结构与社会观察
- 技能、创作、科研、关系、城市等长期主题
- 阶段变化和可继续观察的问题

## 可复用报告 skill

本仓库已沉淀一套可复用的用户成长报告生成 skill：

```text
skills/growth-report-pipeline/SKILL.md
```

它用于把 `04` 生成的证据包、仓库内 prompts 和主线可视化整理成正式的个人成长分析报告。该 skill 明确要求正文以 `04` 证据包为准，不调用 `05` 解释脚本撰写正文；报告组织方式以“节点变迁”为主，而不是罗列证据。

## 环境变量

PowerShell:

```powershell
$env:GEMINI_API_KEY="你的 Gemini API Key"
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
$env:EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
$env:SENTIMENT_MODEL="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
```

代码默认：

```text
Gemini: gemini-3.5-flash
DeepSeek for 02/03 fallback: deepseek-v4-flash
DeepSeek for 05 fallback: deepseek-v4-pro
Sentiment: IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment
Embedding: BAAI/bge-small-zh-v1.5
```

## 依赖

```powershell
pip install pandas plotly scikit-learn joblib tqdm transformers torch sentence-transformers
```

浏览器抓取需要：

```powershell
pip install playwright
python -m playwright install chromium
```

## 运行顺序

### 01 抓取与清洗微博

```powershell
python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\01_weibo_browser_crawl.py --delay-ms 20000
```

输出：

```text
data/raw/weibo_browser/
data/processed/weibo/3666468881_browser_original_clean.jsonl
data/processed/weibo/3666468881_browser_original_clean.txt
```

### 02 首轮 LLM 伪标签

```powershell
python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\02_gemini_first_pass_label.py
```

Gemini 是首选；Gemini 触发 429 时自动 fallback 到 DeepSeek Flash。

输出：

```text
output/weibo/labels/label_batches/
output/weibo/labels/gemini_first_pass_labels.jsonl
output/weibo/labels/pseudo_labels_train.jsonl
output/weibo/labels/gemini_first_pass_labels.md
```

### 02b 局部校对伪标签

```powershell
python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\02b_calibrate_pseudo_labels.py
```

它只校对已标注结果中低分、低信号、长文本样本。文学评论、历史叙事、作品解读、审美偏好、社会观察和创作方法反思，不会因为没有直接出现“我”而自动判为低信号。

会直接覆盖：

```text
output/weibo/labels/label_batches/*.jsonl
output/weibo/labels/gemini_first_pass_labels.jsonl
output/weibo/labels/pseudo_labels_train.jsonl
output/weibo/labels/gemini_first_pass_labels.md
output/weibo/labels/gemini_first_pass_labels.txt
```

### 03 训练本地 keep 模型并处理低置信样本

```powershell
python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\03_train_pseudo_label_classifier.py
```

职责：

```text
1. 用 02/02b 的伪标签训练本地 keep 模型
2. 验证 precision / recall / f1
3. 对 confidence < 0.9 的样本调用 LLM rerank
4. 高置信样本使用 BERT/BGE 结果
5. 低置信样本直接使用 LLM 裁决
6. 输出 final_keep_decisions.jsonl
```

输出：

```text
output/models/pseudo_label_classifier/keep_classifier.joblib
output/models/pseudo_label_classifier/final_keep_decisions.jsonl
output/models/pseudo_label_classifier/training_rerank.jsonl
output/models/pseudo_label_classifier/metadata.json
```

### 04 构建证据包

```powershell
python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\04_build_evidence_pack.py
```

`04` 的作用是把全量清洗微博转换成给 `05` 使用的证据包。

它会：

```text
1. 读取全量清洗微博
2. 读取 03 训练出的 keep_classifier.joblib
3. 读取 03 输出的 final_keep_decisions.jsonl
4. 对全量微博计算 keep 概率
5. 如果某条微博有 LLM 低置信裁决，优先使用 LLM 裁决
6. 如果没有 LLM 裁决，使用本地 BERT/BGE keep 模型
7. 按月份限制每月最多保留若干条证据
8. 输出 evidence_pack.md 给 05 解释层使用
```

输出：

```text
output/weibo/evidence/model_predictions.jsonl
output/weibo/evidence/evidence_candidates.jsonl
output/weibo/evidence/evidence_pack.md
```

### 05 生成成长解释报告

```powershell
python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\05_gemini_growth_interpreter.py
```

Gemini 是首选；Gemini 429 时 fallback 到 DeepSeek Pro。

输出：

```text
output/weibo/gemini/reranked_evidence_interpretation.md
```

### 06 可视化分析

```powershell
python D:\Users\18905\Desktop\资料整理\0.自我复盘系统\scripts\06_visualize_growth_archive.py
```

输出：

```text
output/visualizations/weibo/
```

包括事件密度、情感趋势、成长证据趋势、心路动作分布、证据类型分布、主题/技能热力图和语义空间图。

## 方法边界

- 本系统不做临床诊断。
- 本地 BERT/BGE 模型只判断是否保留，不负责心理解释。
- LLM 解释必须绑定证据，并保留反解释。
- 情感模型只提供情感极性信号，不等于稳定心理状态。
- 文学评论、历史叙事、作品解读和社会观察可以是成长轨迹证据。

## 日志

所有脚本都会打印带时间戳的日志，并写入：

```text
output/logs/
```

例如：

```text
output/logs/02_gemini_first_pass_label.log
output/logs/03_train_pseudo_label_classifier.log
output/logs/04_build_evidence_pack.log
```
