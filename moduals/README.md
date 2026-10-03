# 原始文本到 FinBERT 输入的可复用模块

这个目录封装了项目中从 Guardian HTML 到 FinBERT 输入张量的流程，并且明确停在模型调用之前。

## 流程

```text
HTML + 标题
  -> 删除网页噪声并做最小化规范化
  -> spaCy 分句
  -> 依存句法规则切分 clause
  -> 分别建立 title / clause 文本记录
  -> FinBERT tokenizer
  -> 超长文本切为重叠窗口（正文容量 510，重叠 64）
  -> 添加 [CLS] / [SEP]
  -> 动态 padding
  -> input_ids + attention_mask + token_type_ids
  -> 到此停止，尚未调用 FinBERT
```

## 每个步骤对应一个 Python 文件

1. `html_extraction.py`：删除网页噪声、最小化规范化并把标题放到正文前面。
2. `sentence_splitting.py`：使用 spaCy 分句并保留字符偏移。
3. `clause_splitting.py`：按依存句法规则保守地切分 clause。
4. `record_building.py`：建立彼此独立的 title 和 clause 文本记录。
5. `finbert_tokenization.py`：加载固定版本 tokenizer，把字符串变成正文 token ID。
6. `windowing.py`：对超长序列建立重叠窗口并加入 `[CLS]`、`[SEP]`。
7. `tensor_building.py`：动态 padding，生成传给 FinBERT 的三个整数张量。

`pipeline.py` 只负责依次调用以上七个步骤；`types.py` 保存步骤间传递的数据结构；
`__init__.py` 提供统一导入接口。

这里直接复用 `src/preprocess_financial_news.py` 和
`src/finbert_vectorization.py` 的生产规则，避免复制两套逐渐不一致的预处理逻辑。

## 最小示例

```python
from moduals import load_tokenizer, prepare_article_for_finbert

tokenizer = load_tokenizer()
prepared = prepare_article_for_finbert(
    html_content="<p>Oil prices rose, but equities declined.</p>",
    title="Markets react to conflict",
    document_id="guardian/example",
    tokenizer=tokenizer,
)

# 这三个整数张量就是外部传给 FinBERT 的内容。
print(prepared.finbert.model_inputs["input_ids"].shape)
print(prepared.finbert.model_inputs["attention_mask"].shape)
print(prepared.finbert.model_inputs["token_type_ids"].shape)

# 本模块有意不执行下面这一步：
# output = finbert_model(**prepared.finbert.model_inputs)
```

标题和 clause 会作为独立记录输入，二者不会被拼接。若文本超过 510 个正文 token，
一个记录会对应多个窗口；`window_record_indices` 可以把每个窗口映射回原记录。
