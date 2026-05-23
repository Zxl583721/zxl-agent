# rag-agent

一个简易的本地 RAG Agent 项目骨架，用于后续接入个人知识库文件，并通过智谱 AI 大模型完成检索增强问答。

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

然后编辑 `.env`，填入你的智谱 AI API Key：

```bash
ZHIPUAI_API_KEY=your_api_key_here
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

索引构建时会先保留 PDF 页码或 PPT 幻灯片页码，再识别“第 X 讲/章/节”、多级数字标题和中文序号标题等章节结构。程序会按章节聚合资料内容，并在章节内部使用递归文本切分生成向量块。

每个文本块会写入来源文件、章节标题、起止页码、章节序号和块序号等 metadata，便于检索结果在回答时展示来源、章节和页码。

## 检索与精排策略

问答时会先从 Chroma 向量库召回更多候选文本块，再通过轻量级 reranker 进行二阶段精排。reranker 会综合原始向量相似度、问题与正文的词面匹配度，以及问题与文件名/章节标题的匹配度，最后选择最相关的文本块注入 Prompt。

## 多轮问题改写

针对“它、这个、上述、前面”等多轮追问，系统会先调用 LLM 将当前问题改写为适合检索的独立问题，再使用改写后的问题进行向量召回和 reranker 精排。如果改写失败，会自动回退到规则式问题拼接，保证问答流程可用。

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
    ├── retriever.py
    ├── reranker.py
    ├── query_rewriter.py
    ├── zhipu_llm.py
    └── rag_agent.py
```
