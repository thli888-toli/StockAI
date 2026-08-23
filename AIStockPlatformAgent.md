# AIStockPlatformAgent — System Architecture (Tech Review)

## 1. Purpose

本文档描述 StockAI 平台的技术结构、组件关系、数据流和依赖栈，供技术评审使用。

## 2. High-Level Architecture

```text
                         ┌─────────────────────────────────────────┐
                         │             LangGraph Orchestrator      │
                         │              (FastAPI :8020)             │
                         └───────────────┬─────────────────────────┘
                                         │ RemoteAgentClient (HTTP)
                 ┌───────────────────────┼───────────────────────────┐
                 ▼                       ▼                           ▼
           stock_data :8021         stock_news :8022            stock_quant :8024
                 │                       │                           │
                 └───────────┬───────────┴───────────────────────────┘
                             ▼
                     stock_analyst :8023
                             │
                             ▼
                   final report + summary
```

编排流：`stock_data -> stock_news -> stock_quant -> stock_analyst`。

## 3. Components

### 3.1 Framework

| 模块 | 说明 |
|---|---|
| `framework/orchestrator.py` | 运行创建/取消/列表，加载 graph manifest，执行 LangGraph，共享 SQLite checkpoint |
| `framework/agent_client.py` | 调用远程智能体，轮询任务状态，容忍瞬时 HTTP 错误 |
| `framework/agent_service.py` | 智能体 FastAPI 服务：`POST /tasks`、`GET /tasks/{id}` |
| `framework/graph_compiler.py` | 把 `GraphManifest` 编译为 LangGraph StateGraph |
| `framework/registry.py` / `registry_client.py` | 智能体注册与心跳 |
| `framework/checkpoint.py` | 长生命周期共享 `AsyncSqliteSaver`，WAL + busy_timeout |
| `framework/config.py` | 环境变量与 ModelConfig |
| `framework/llm.py` | LLM 文本调用封装 |

### 3.2 Agents

| Agent | 目录 | 职责 |
|---|---|---|
| `stock_data` | `plugins/stock_data` | AkShare 日线、增量缓存、公司信息、MACD/RSI/MA/特征矩阵 |
| `stock_news` | `plugins/stock_news` | 东方财富、巨潮、财联社/同花顺，交叉验证 |
| `stock_quant` | `plugins/stock_quant` | LSTM 日/周预测、月线技术趋势、持久化量化缓存 |
| `stock_analyst` | `plugins/stock_analyst` | LLM 中文报告 + summary，与量化信号交叉验证 |

共享逻辑：

- `plugins/stock_common.py`：符号校验、MACD、特征构建、重采样、标题相似度、缓存、阻塞调用封装。
- `plugins/stock_cache.py`：行情增量 SQLite 存储。

### 3.3 Portals

- 系统监控门户：`portal_backend/` + `portal-ui/`，端口 8030。
- 股票分析门户：`stockportal/` + `stockportal/ui/`，端口 8040。

股票门户前端：

- React 18 + TypeScript + Vite
- `react-markdown` + `remark-gfm` 渲染报告
- `lightweight-charts` 渲染 K 线/MACD/支撑阻力/箭头

### 3.4 Storage

| 文件 | 用途 |
|---|---|
| `state/stock_cache.db` | `stock_history_bars`、`stock_history_meta`、`quant_cache` |
| `state/orchestrator.db` | LangGraph checkpoints（WAL） |
| `state/portal.db` | 监控门户 agents/metrics/logs/runs |
| `state/stock_portal.db` | `users`、`sessions`、`watchlist` |

## 4. Data Flow

1. 用户登录获取 `Bearer token`，提交 6 位代码。
2. `stock_data` 检查历史缓存，缺失区间拉取 AkShare，计算特征矩阵并回写。
3. `stock_news` 抓取并交叉验证新闻。
4. `stock_quant` 计算特征哈希，命中缓存直接返回；否则训练 LSTM 日/周模型并计算月线技术趋势，写入缓存。
5. `stock_analyst` 组装市场数据、新闻、量化结果，调用 DeepSeek 生成 `report` 与 `summary`；失败则确定性回退。
6. 门户轮询 orchestrator run 状态，完成后展示报告；图表接口从 `outputs` 读取市场数据与量化结果渲染。

## 5. Quant Model Details

- 框架：PyTorch CPU，轻量 `LSTM -> Linear` 二分类。
- 日线周期：`5d`、`15d`；周线周期：`1w`；月线为确定性技术趋势。
- 特征：收益率、MACD、RSI、MA 比率、波动率、量比、换手率。
- 训练：walk-forward 扩展窗口；参数可按周期传入。
- 缓存：`(symbol, feature_hash)` -> 完整 quant payload。

## 6. Auth & Multi-Tenancy

- 模拟微信登录：`POST /api/login` 创建/复用用户，返回随机 token。
- 同一昵称复用最早用户；`default`/`默认用户` 映射到内置默认用户。
- `sessions.token` 关联 `users.id`；股票门户数据接口要求 `Authorization: Bearer <token>`。
- `watchlist` 主键 `(user_id, symbol)`。

## 7. Concurrency and Timeouts

- Orchestrator 使用共享 checkpoint 连接，串行化 checkpoint 写，避免 SQLite 锁。
- `RemoteAgentClient` 轮询容忍瞬时 HTTP 错误，连续失败阈值后失败。
- `RUN_TIMEOUT_SECONDS=600`；单智能体任务超时与 run 超时对齐。
- 阻塞 AkShare/LSTM 调用通过 `asyncio.to_thread` + `asyncio.wait_for`。

## 8. Tech Stack

- Backend：Python 3.12、FastAPI、Uvicorn、httpx、Pydantic
- Orchestration：LangGraph、langgraph-checkpoint-sqlite、aiosqlite
- LLM：langchain-openai、DeepSeek API（默认 `deepseek-v4-pro`）
- Data：AkShare、pandas、numpy
- ML：PyTorch（LSTM）、scikit-learn（metrics）
- Frontend：React 18、TypeScript、Vite、lightweight-charts、react-markdown
- Storage：SQLite

## 9. Configuration

`.env` 关键项：

```text
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-v4-pro
RUN_TIMEOUT_SECONDS=600
REGISTRY_URL=http://127.0.0.1:8001
ORCHESTRATOR_URL=http://127.0.0.1:8020
PORTAL_BACKEND_URL=http://127.0.0.1:8030
STOCKPORTAL_URL=http://127.0.0.1:8040
CHECKPOINT_DB=state/orchestrator.db
PORTAL_DB=state/portal.db
STOCK_CACHE_DB=state/stock_cache.db
STOCK_PORTAL_DB=state/stock_portal.db
```

## 10. Testing

- `tests/test_stock_agents.py`：量化/新闻/分析师单元测试。
- `tests/test_stock_agent_handlers.py`：agent handler 集成测试。
- `tests/test_stock_portal.py`：认证、多用户、去重、刷新拦截、图表。
- `tests/test_iteration1.py` / `test_iteration2.py`：框架与监控门户。
- `e2e_test_600988.py`：真实 AkShare 端到端流水线。
