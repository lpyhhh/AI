
# BioJarvis 前端部署手册

这是一个专为生物信息学 Agent 设计的高性能前端。

## 如何在本地运行？

1. **安装环境**：确保你本地安装了 [Node.js](https://nodejs.org/)。
2. **启动服务**：
   在项目根目录下运行：
   ```bash
   npx servor . index.html --browse
   ```
   或者使用 VS Code 的 **Live Server** 插件点击右下角的 "Go Live"。

## 如何连接到你的 AWS L4 后端？

按照你学习计划中的 **Week 15-16**，请执行以下步骤：

1. **后端准备**：
   在 AWS 实例上使用 FastAPI 封装你的 `agent.py`：
   ```python
   # 示例代码 (fastapi_server.py)
   from fastapi import FastAPI
   from agent import run_agent

   app = FastAPI()

   @app.post("/chat")
   async def chat(query: str):
       result = run_agent(query)
       return {"response": result}
   ```
2. **端口转发**：
   在 VS Code 的 **PORTS** 标签页，点击 **Add Port**，输入 `8000`。
3. **前端配置**：
   在 `services/bioService.ts` 中，取消 `USE_LOCAL_BACKEND` 的注释。

## 学习复盘点
- **前端架构**：React (UI) + Tailwind (样式) + Gemini (基础对话)。
- **后端对接**：通过端口转发实现混合云架构（本地展示，云端推理）。
