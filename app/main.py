from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.routes import router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="本地知识库 RAG Agent",
        description=(
            "面向本地知识库问答的 FastAPI 服务，支持健康检查、普通问答、"
            "流式问答、知识库文件上传、文档列表和后台任务状态查询。"
        ),
        version="0.1.0",
        contact={"name": settings.app_name},
        openapi_tags=[
            {"name": "系统状态", "description": "检查 API、数据库、Redis 和检索服务状态。"},
            {"name": "智能问答", "description": "基于知识库或通用模式进行问答。"},
            {"name": "知识库", "description": "上传、索引和查看知识库文档。"},
            {"name": "任务", "description": "查看异步后台任务的执行状态。"},
        ],
    )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>本地知识库 RAG Agent</title>
    <style>
      :root {
        color-scheme: light;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f6f7f9;
        color: #1f2933;
      }
      body {
        margin: 0;
        min-height: 100vh;
      }
      main {
        width: min(1080px, calc(100vw - 40px));
        margin: 0 auto;
        padding: 28px 0 40px;
      }
      h1 {
        margin: 0 0 12px;
        font-size: 30px;
        line-height: 1.25;
      }
      p {
        margin: 0 0 20px;
        color: #52606d;
        line-height: 1.7;
      }
      .panel {
        background: #ffffff;
        border: 1px solid #dfe3e8;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 18px 48px rgba(31, 41, 51, 0.08);
      }
      .actions {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 16px 0 0;
      }
      a {
        color: #0969da;
        text-decoration: none;
      }
      button, .button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 40px;
        padding: 0 16px;
        border-radius: 6px;
        border: 1px solid #c9d2dc;
        background: #f8fafc;
        color: #1f2933;
        font-weight: 600;
        cursor: pointer;
        font-size: 14px;
      }
      .primary {
        border-color: #0969da;
        background: #0969da;
        color: #ffffff;
      }
      .primary:disabled {
        cursor: not-allowed;
        opacity: 0.6;
      }
      .layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 320px;
        gap: 18px;
        margin-top: 18px;
      }
      .chat {
        display: grid;
        gap: 14px;
      }
      label {
        display: block;
        margin-bottom: 8px;
        color: #52606d;
        font-weight: 700;
        font-size: 14px;
      }
      textarea, select {
        width: 100%;
        box-sizing: border-box;
        border: 1px solid #c9d2dc;
        border-radius: 6px;
        background: #ffffff;
        color: #1f2933;
        font: inherit;
      }
      textarea {
        min-height: 132px;
        resize: vertical;
        padding: 12px;
        line-height: 1.6;
      }
      select {
        min-height: 40px;
        padding: 0 10px;
      }
      .toolbar {
        display: grid;
        grid-template-columns: 220px 1fr auto;
        gap: 12px;
        align-items: end;
      }
      .answer {
        min-height: 220px;
        white-space: pre-wrap;
        line-height: 1.75;
        background: #fbfcfe;
        border: 1px solid #dfe3e8;
        border-radius: 8px;
        padding: 16px;
      }
      .muted {
        color: #7b8794;
      }
      .status {
        color: #52606d;
        font-size: 14px;
      }
      .sources {
        display: grid;
        gap: 10px;
        margin-top: 12px;
      }
      .source {
        border: 1px solid #e5eaf0;
        border-radius: 6px;
        padding: 10px;
        background: #fbfcfe;
        font-size: 14px;
        line-height: 1.6;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
        font-size: 14px;
      }
      th, td {
        border-top: 1px solid #e5eaf0;
        padding: 12px 8px;
        text-align: left;
        vertical-align: top;
      }
      th {
        color: #52606d;
        font-weight: 700;
      }
      code {
        font-family: "SFMono-Regular", Consolas, monospace;
        background: #edf2f7;
        border-radius: 4px;
        padding: 2px 5px;
      }
      @media (max-width: 860px) {
        .layout, .toolbar {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <section class="panel">
        <h1>本地知识库 RAG Agent</h1>
        <p>在下面输入问题，系统会优先根据已经建立的知识库回答，并在回答后显示参考来源。</p>
        <div class="actions">
          <a class="button" href="/api/knowledge/documents" target="_blank" rel="noreferrer">查看知识库文档</a>
          <a class="button" href="/docs" target="_blank" rel="noreferrer">查看接口文档</a>
          <a class="button" href="/health" target="_blank" rel="noreferrer">检查服务状态</a>
        </div>
      </section>

      <div class="layout">
        <section class="panel chat">
          <div>
            <label for="question">输入你的问题</label>
            <textarea id="question" placeholder="例如：雷达测距精度主要受哪些因素影响？"></textarea>
          </div>
          <div class="toolbar">
            <div>
              <label for="mode">问答模式</label>
              <select id="mode">
                <option value="knowledge" selected>知识库问答</option>
                <option value="general">通用聊天</option>
              </select>
            </div>
            <div class="status" id="status">知识库状态检测中...</div>
            <button class="primary" id="askButton" type="button">发送问题</button>
          </div>
          <div>
            <label>回答</label>
            <div class="answer muted" id="answer">回答会显示在这里。</div>
          </div>
          <div>
            <label>参考来源</label>
            <div class="sources" id="sources">
              <div class="source muted">如果回答引用了知识库文档，来源会显示在这里。</div>
            </div>
          </div>
        </section>

        <aside class="panel">
          <h2 style="margin: 0 0 12px; font-size: 20px;">当前服务</h2>
          <p id="kbSummary">正在读取知识库信息...</p>
          <table>
            <tbody>
              <tr><td><code>/api/chat</code></td><td>问答接口</td></tr>
              <tr><td><code>/api/knowledge/documents</code></td><td>文档列表</td></tr>
              <tr><td><code>/health</code></td><td>健康检查</td></tr>
            </tbody>
          </table>
        </aside>
      </div>
    </main>
    <script>
      const questionInput = document.querySelector("#question");
      const modeInput = document.querySelector("#mode");
      const askButton = document.querySelector("#askButton");
      const answerBox = document.querySelector("#answer");
      const sourcesBox = document.querySelector("#sources");
      const statusBox = document.querySelector("#status");
      const kbSummary = document.querySelector("#kbSummary");
      let conversationId = null;

      function setAnswer(text, muted = false) {
        answerBox.textContent = text;
        answerBox.classList.toggle("muted", muted);
      }

      function renderSources(sources) {
        sourcesBox.innerHTML = "";
        if (!sources || sources.length === 0) {
          sourcesBox.innerHTML = '<div class="source muted">本次回答没有返回参考来源。</div>';
          return;
        }
        for (const source of sources) {
          const name = source.source || "未命名文档";
          const page = source.page_range || source.start_page || "";
          const chapter = source.chapter ? `章节：${source.chapter}` : "";
          const pageText = page ? `页码：${page}` : "";
          const detail = [chapter, pageText].filter(Boolean).join("，");
          const item = document.createElement("div");
          item.className = "source";
          item.textContent = detail ? `${name}（${detail}）` : name;
          sourcesBox.appendChild(item);
        }
      }

      async function loadStatus() {
        try {
          const [healthResponse, docsResponse] = await Promise.all([
            fetch("/health"),
            fetch("/api/knowledge/documents"),
          ]);
          const health = await healthResponse.json();
          const docs = await docsResponse.json();
          const count = Array.isArray(docs.documents) ? docs.documents.length : 0;
          statusBox.textContent = health.ready ? "知识库已就绪" : "知识库未完全就绪";
          kbSummary.textContent = `${health.message || "服务已启动"} 当前文档数量：${count}。`;
        } catch (error) {
          statusBox.textContent = "状态读取失败";
          kbSummary.textContent = "暂时无法读取知识库状态，请检查容器日志。";
        }
      }

      async function askQuestion() {
        const question = questionInput.value.trim();
        if (!question) {
          questionInput.focus();
          setAnswer("请先输入一个问题。", true);
          return;
        }
        askButton.disabled = true;
        statusBox.textContent = "正在生成回答...";
        setAnswer("正在思考，请稍等。", true);
        sourcesBox.innerHTML = '<div class="source muted">正在检索参考来源...</div>';
        try {
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              question,
              mode: modeInput.value,
              conversation_id: conversationId,
              user_id: 1,
              knowledge_base_id: 1,
            }),
          });
          const data = await response.json();
          if (!response.ok) {
            const message = data.detail?.message || data.detail || "请求失败，请查看服务日志。";
            throw new Error(typeof message === "string" ? message : JSON.stringify(message));
          }
          conversationId = data.conversation_id || conversationId;
          setAnswer(data.answer || "没有返回回答。");
          renderSources(data.sources || []);
          statusBox.textContent = data.cache_hit ? "已从缓存返回" : "回答完成";
        } catch (error) {
          setAnswer(`生成失败：${error.message}`, true);
          renderSources([]);
          statusBox.textContent = "生成失败";
        } finally {
          askButton.disabled = false;
        }
      }

      askButton.addEventListener("click", askQuestion);
      questionInput.addEventListener("keydown", (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
          askQuestion();
        }
      });
      loadStatus();
    </script>
  </body>
</html>
"""

    application.include_router(router)
    return application


app = create_app()
