# DeepSeek 新闻风险标注

入口脚本：`annotate_deepseek_risk.py`

该流程只读取 `data/processed_news/{x,guardian}` 中已经拆分好的数据，不重新分句，也不修改输入 CSV。标题和正文子句是两类独立 target：标题只使用标题文本；子句可以使用所属完整句子作为 context，但 context 只能帮助解析指代、否定、条件和语义关系。

## 三个输出维度

程序严格按以下顺序标注：

1. `topic_relevance`：`0` 表示无关，`0.2` 表示明确讨论笼统 AI 风险但无法归入五类，`1` 表示能明确归入五类中的至少一类。`0.2` 是内容权重，不是模型置信度。
2. `risk_type`：`topic_relevance=1` 时从五类风险中多选；`0.2` 时为 `[]`；`0` 时为 `null`。
3. `risk_direction`：`topic_relevance=0.2` 或 `1` 时判断风险表达倾向，而非时间变化。肯定或警示风险为 `1`，否认、弱化或明确缓解风险为 `-1`，中性、不清楚或混合为 `0`。

`topic_relevance=0` 时，程序校验并强制要求 `risk_type=null`、`risk_direction=null` 和 `direction_status=not_applicable`。`topic_relevance=0.2` 时，程序强制要求 `risk_type=[]`，但仍需给出合法的方向、理由和证据。明确、完整的笼统 AI 风险不因无法细分类别自动进入复核；证据不足的文本仍暂记为 0，并设置 `needs_review=true`。API、JSON 解析或格式校验失败只写入错误日志，不会伪装成不相关记录。

当前 prompt 版本为 `topic_risk_stance_v11_2026-10-02`。缓存键包含 target、context、unit type、模型、API 参数、prompt 版本和 prompt 哈希，因此不会复用 v10 或更早版本产生的结果。

## 配置

密钥从 `.env` 或系统环境变量 `DEEPSEEK_API_KEY` 读取，脚本不会打印或写出密钥。也可以配置：

- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`

模型名在每次联网运行前通过 `/models` 核验。若项目 `.env` 中的旧模型名已不可用，请使用命令行 `--model` 指定当前模型；不要在代码中硬编码密钥。

## 运行

只检查输入和抽样，不调用 API：

```powershell
.\.venv\Scripts\python.exe annotate_deepseek_risk.py --mode trial --model deepseek-v4-pro --sample-size 200 --dry-run
```

只运行合成边界案例：

```powershell
.\.venv\Scripts\python.exe annotate_deepseek_risk.py --mode boundary --model deepseek-v4-pro
```

运行试标：最多 200 个标题和 200 个非标题子句，X 与 Guardian 各占一半，并附带边界案例：

```powershell
.\.venv\Scripts\python.exe annotate_deepseek_risk.py --mode trial --model deepseek-v4-pro --sample-size 200
```

断点恢复会自动读取逐条结果日志和缓存；完全离线地从现有进度重新导出可使用 `--offline-resume`。只有人工检查试标结果后，才运行全量：

```powershell
.\.venv\Scripts\python.exe annotate_deepseek_risk.py --mode full --confirm-full --model deepseek-v4-pro
```

只标注全部正文子句、不调用任何标题时，使用隔离的专用模式：

```powershell
.\.venv\Scripts\python.exe annotate_deepseek_risk.py --mode full-clauses --confirm-full --model deepseek-v4-pro --dry-run
.\.venv\Scripts\python.exe annotate_deepseek_risk.py --mode full-clauses --confirm-full --model deepseek-v4-pro
```

`full-clauses` 在调用 API 前会硬性校验所有 target 均为非标题子句，输出保存在 `data/deepseek_risk_annotations/full-clauses/`。当前输入共 57,034 条正文子句（X 13,284；Guardian 43,750），正式执行前应核对余额和预计 token 成本。

大规模运行可用 `--max-new-requests` 限制单次新增请求量；每批完成后状态为 `in_progress`，重复执行同一命令会从逐条进度与缓存继续。例如每批最多新增 500 个唯一请求：

```powershell
.\.venv\Scripts\python.exe annotate_deepseek_risk.py --mode full-clauses --confirm-full --model deepseek-v4-pro --max-new-requests 500
```

只标注 X 平台子句时使用平台过滤；结果与双平台任务隔离保存在
`data/deepseek_risk_annotations/full-clauses-x/`，全局缓存仍会复用：

```powershell
.\.venv\Scripts\python.exe annotate_deepseek_risk.py --mode full-clauses --platform x --confirm-full --model deepseek-v4-pro --max-new-requests 500
```

全部批次完成后状态变为 `complete`。若尚有未标注记录，`--offline-resume` 会拒绝伪造完整输出。

## 输出

试标结果保存在 `data/deepseek_risk_annotations/trial/`：

- `x/headline_annotations.csv`、`x/clause_annotations.csv`
- `guardian/headline_annotations.csv`、`guardian/clause_annotations.csv`
- 两个平台各自的 `review_queue.csv`
- `trial_preview.csv`
- `boundary_tests.csv`
- `trial_summary.json`、`run_manifest.json`
- `progress/results.jsonl`：逐条成功结果，用于断点续跑
- `errors.jsonl`：API、解析和校验失败

缓存位于 `data/deepseek_risk_annotations/cache/<prompt_version>/`。输出保留原有文档 ID、句子 ID、子句 ID、时间戳、原文、context 和 `unit_type`，并记录配置模型、实际模型、参数、prompt 版本、标注时间和 token 用量。流程不计算综合指数。
