# RAG 评测使用说明

这套评测分两类：

- `eval_retrieval.py`：检索消融，比较纯向量、向量+BM25+RRF、Query Expansion、Reranker。
- `eval_multiturn.py`：多轮追问改写，比较原始 query 与改写 query 的 Recall@K。

## 1. 先跑粗评估

样例数据里的 `relevant_chunk_ids` 为空时，脚本会用 `expected_sources` 做来源级评估，适合快速得到第一版结果。

```bash
python eval/eval_retrieval.py --queries eval/retrieval_cases.jsonl --top-k 5 -v
python eval/eval_multiturn.py --queries eval/multiturn_cases.jsonl --top-k 5 -v
```

如果当前环境没有 BGE 依赖或模型，可先只跑线上 lightweight 配置；需要重新跑 BGE 时再补齐 `FlagEmbedding` 和本地模型目录：

```bash
python eval/eval_retrieval.py --queries eval/retrieval_cases.jsonl --configs hybrid_expanded,full --top-k 5
python eval/eval_retrieval.py --queries eval/retrieval_cases.jsonl --configs bge --bge-model models/bge-reranker-v2-m3 --top-k 5
```

## 2. 做 chunk 级标注

用标注辅助脚本查看候选 chunk：

```bash
python eval/annotate_retrieval.py --query "雷达方程的具体表达式是什么？" --type formula --top-k 20
```

把输出里的正确 `chunk_id` 填入 `eval/retrieval_cases.jsonl` 或 `eval/multiturn_cases.jsonl` 的 `relevant_chunk_ids`，脚本会自动按 chunk 级指标计算。

## 3. 建议样本量

- 检索评测：至少 30 条，覆盖 definition、term、formula、semantic、comparison。
- 多轮评测：至少 15 组，覆盖 anaphora、demonstrative、ellipsis、short_question。

简历里建议同时写绝对值和相对提升，例如：

> 构建 30 条人工标注 RAG 检索评测集，完成纯向量、BM25+RRF、Query Expansion、Reranker 消融实验，Recall@5 从 X% 提升至 Y%，MRR 从 A% 提升至 B%。
