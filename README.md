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

如果已经把 `.txt` 知识库文件放入 `data/`，可以先构建向量索引：

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
http://127.0.0.1:5000
```

## 添加知识库文件

后续可以把个人知识库原始文件放入 `data/` 文件夹。

当前基础代码先支持读取 `.txt` 文件。你可以放入类似：

```text
data/my_notes.txt
data/project_docs.txt
```

后续可以在 `src/document_loader.py` 中扩展 PDF、Word、Markdown 等格式的加载逻辑。

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
    ├── text_splitter.py
    ├── embedding.py
    ├── retriever.py
    ├── zhipu_llm.py
    └── rag_agent.py
```
