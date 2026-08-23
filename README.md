# StockAI — A-share Stock Analysis Platform

基于 LangGraph 的 A 股多智能体分析平台。系统从 AkShare/东方财富拉取行情与新闻，由 LSTM 量化模型给出多周期方向概率，再由 LLM（默认 DeepSeek-V4-Pro）生成中文投资分析报告，并提供多用户股票池和 K 线图展示。

## 架构概览

```text
stock_data -> stock_news -> stock_quant -> stock_analyst
```

| 组件 | 端口 | 职责 |
|---|---:|---|
| Registry | 8001 | 智能体注册与心跳 |
| Orchestrator | 8020 | LangGraph 编排、运行管理、SQLite checkpoint |
| `stock_data` | 8021 | 行情抓取、MACD、特征矩阵、增量缓存 |
| `stock_news` | 8022 | 东方财富 + 巨潮 + 第二媒体源，交叉验证 |
| `stock_analyst` | 8023 | LLM 分析总结、与量化信号交叉验证 |
| `stock_quant` | 8024 | LSTM 日/周预测 + 月线技术趋势 |
| 系统监控门户 | 8030 | 注册中心/运行/日志/指标管理 |
| 股票分析门户 | 8040 | 多用户股票池、图表、分析报告 |

## 快速开始

```powershell
python -m pip install -r requirements.txt

cd stockportal\ui
npm install
npm run build
cd ..\..

cd portal-ui
npm install
npm run build
cd ..

python -m main run
```

- 股票分析门户：<http://127.0.0.1:8040>
- 系统监控门户：<http://127.0.0.1:8030>

首次打开股票门户时，输入昵称点击“微信登录（模拟）”创建用户；输入 6 位 A 股代码点击“分析”即可加入个人股票池。

## 运行命令

```powershell
python -m main registry
python -m main agent --manifest plugins/stock_data/agent.yaml
python -m main agent --manifest plugins/stock_news/agent.yaml
python -m main agent --manifest plugins/stock_quant/agent.yaml
python -m main agent --manifest plugins/stock_analyst/agent.yaml
python -m main orchestrator --manifest config/orchestration.yaml
python -m main portal
python -m main stockportal
python -m main run
```

## 关键能力

- 行情数据：AkShare 日线 OHLCV，首次拉取 3 年，之后增量补缺。
- 新闻：东方财富 + 巨潮资讯公告 + 财联社/同花顺快讯关键词匹配，双源交叉验证。
- 量化：LSTM 日线 `5d/15d`、周线 `1w`；月线 `1mo` 采用 MACD + 均线确定性趋势。
- 分析报告：DeepSeek-V4-Pro 输出纯中文 Markdown + 结构化结论 `summary`。
- 图表：日/周/月 K 线、MACD(6,13,5)、MA(20/66/154/250)、斐波那契支撑阻力、趋势箭头、LLM 观点。
- 多用户：模拟微信登录，每个用户独立股票池；同日重复分析会被拦截。
- 持久化：SQLite 保存历史行情、量化缓存、用户/会话、股票池和运行记录。

## LLM 配置

`.env` 中默认：

```text
DEEPSEEK_MODEL=deepseek-v4-pro
RUN_TIMEOUT_SECONDS=600
```

可通过 `DEEPSEEK_MODEL` 切换为 `deepseek-v4-flash` 或 OpenAI 兼容模型；未配置 LLM 时分析师回退到确定性中文报告。

## 测试

```powershell
python -m pytest tests -q --ignore=tests/test_iteration2.py
python e2e_test_600988.py
```

完整测试包含网络/门户聚合测试，建议本地服务未运行时执行。`e2e_test_600988.py` 使用真实 AkShare 数据验证整条流水线。

## 数据存储

- `state/stock_cache.db`：行情历史 + 量化预测缓存
- `state/orchestrator.db`：LangGraph checkpoint
- `state/portal.db`：监控门户运行/日志/指标
- `state/stock_portal.db`：用户、会话、股票池
