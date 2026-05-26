# rag-agent

一个简易的本地 RAG Agent 项目骨架，用于接入个人知识库文件，并通过本地 Ollama 模型完成检索增强问答。

当前版本已经预留了上传、加载、切分、向量化、检索和问答的代码结构。因为你还没有上传知识库文件，程序会在没有数据时给出友好提示。

## 安装依赖

建议先创建虚拟环境：

```bash
conda create -p ./.conda python=3.11
conda activate ./.conda
pip install -r requirements.txt
```

## 配置环境变量

复制示例环境变量文件：

```bash
cp .env.example .env
```

默认配置使用本地 Ollama：

```bash
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3:8b

EMBEDDING_PROVIDER=ollama
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3
```

首次运行前请先安装 Ollama 并下载模型：

```bash
ollama pull qwen3:8b
ollama pull bge-m3
```

## 运行项目

如果已经把知识库文件放入 `data/`，可以先构建向量索引：

```bash
python build_index.py
```

当前索引会保存到 `vector_store/` 目录中的本地 Chroma 数据库。如果还没有上传知识库文件，可以先跳过这一步。

```bash
python main.py
```

进入命令行交互后，输入问题即可提问；输入 `exit` 退出。

也可以启动本地网页界面：

```bash
python web_app.py
```

然后打开浏览器访问：

```text
http://127.0.0.1:5001
```

网页侧边栏支持上传 `.txt`、`.pdf`、`.docx`、`.pptx` 文件。上传后后端会保存到 `data/`，并自动触发增量索引构建；索引完成后可以直接在页面中提问。

网页侧边栏还提供文档管理面板，可以查看已上传文件的索引状态、文本块数量和 hash 摘要，并支持删除文件后自动更新向量索引。

网页聊天会使用后端 SQLite 会话历史。前端只保存当前 `conversation_id`，每次提问时由后端读取最近 10 条消息作为短期上下文，再结合问题改写和知识库检索生成回答。刷新页面后，同一浏览器会自动恢复当前会话消息；本地会话库保存在 `conversation_store.sqlite3`，不会提交到 Git。

## 添加知识库文件

后续可以把个人知识库原始文件放入 `data/` 文件夹。

当前支持读取 `.txt`、`.pdf`、`.docx`、`.pptx` 文件。你可以放入类似：

```text
data/my_notes.txt
data/project_docs.txt
data/product_manual.pdf
data/meeting_notes.docx
data/training_slides.pptx
```

如果新增了文件，重新运行 `python build_index.py` 后会更新本地 Chroma 索引。

当前索引构建支持增量更新：程序会在 `vector_store/index_manifest.json` 中记录每个文件的 hash 和文本块 id。再次运行 `python build_index.py` 时，只会处理新增、修改或删除的文件。若需要强制重建整个索引，可以运行：

```bash
python build_index.py --rebuild
```

## 知识库切分策略

索引构建时会先保留 PDF 页码或 PPT 幻灯片页码，再识别“第 X 讲/章/节”、多级数字标题和中文序号标题等章节结构。程序会按章节聚合资料内容，并使用 Parent-Child Chunking：章节级 parent section 用于回答上下文，较小的 child chunk 用于向量检索。

每个 child 文本块会写入来源文件、章节标题、起止页码、parent id、章节序号和块序号等 metadata，便于检索命中后回溯到完整 parent section，并在回答时展示来源、章节和页码。

## 检索与精排策略

问答时会使用 Hybrid Search 进行混合召回：一方面从 Chroma 向量库召回语义相似文本块，另一方面使用本地 BM25 关键词检索召回专业术语、公式名和章节标题相关文本块，并通过 RRF 融合两路结果。融合后的候选文本块会再经过轻量级 reranker 二阶段精排。reranker 会综合原始召回分、问题与正文的词面匹配度、问题与文件名/章节标题的匹配度，以及课程讲次线索，最后选择最相关的文本块注入 Prompt。对于“前言/课程安排”这类目录型资料，会在精排阶段做轻量降权，避免其挤占正式讲义内容。

BM25 关键词召回前会先做确定性的查询扩展：系统会根据课程领域词表补充同义词、专业术语和讲次线索，例如“门限/阈值/检测门限”、“测速/多普勒”、“干扰/对抗/ECCM”等。扩展查询只用于关键词召回，不额外调用大模型，也不会改变用户原始问题的生成表达。

## 多轮问题改写

针对“它、这个、上述、前面”等多轮追问，系统会先从后端会话库读取最近窗口，再判断当前问题是否依赖历史上下文。完整问题会直接检索；疑似追问才调用 LLM 改写为适合检索的独立问题，再使用改写后的问题进行向量召回和 reranker 精排。如果改写失败，会自动回退到规则式问题拼接，保证问答流程可用。

## 可追溯回答生成

最终注入模型的上下文会按 `[资料 1]`、`[资料 2]` 等编号组织，并包含来源文件、章节和页码。Prompt 会要求模型在回答正文中只引用这些资料编号，不直接编造参考来源名称或页码；后处理会检查回答中的资料编号是否来自本轮检索结果，如果模型没有给出有效引用，会自动追加本轮检索资料编号作为兜底依据。网页端的“参考来源”会显示相同编号，方便把回答正文和来源列表对应起来。

## RAG 效果评估

项目提供了一个轻量评估集和评估脚本，用于检查检索来源命中率、回答来源命中率和关键词覆盖率：

```bash
python eval/run_eval.py
```

默认模式只评估检索和来源命中。如果希望同时调用大模型生成回答并评估关键词覆盖，可以运行：

```bash
python eval/run_eval.py --answer
```

如果想对比关闭 LLM 查询改写后的效果，可以运行：

```bash
python eval/run_eval.py --disable-rewrite
```

## 项目结构

```text
zxl-agent/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── build_index.py
├── main.py
├── web_app.py
├── config.py
├── eval/
│   ├── qa_eval.json
│   └── run_eval.py
├── templates/
│   └── index.html
├── data/
│   └── .gitkeep
├── vector_store/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── document_loader.py
    ├── section_splitter.py
    ├── text_splitter.py
    ├── embedding.py
    ├── langchain_ollama.py
    ├── retriever.py
    ├── reranker.py
    ├── query_expander.py
    ├── model_provider.py
    ├── query_rewriter.py
    ├── zhipu_llm.py
    └── rag_agent.py
```
