# AIStockAnalysisPlatform — Requirements

## 1. Overview

构建一个 A 股智能分析平台，基于现有 LangGraph 智能体编排框架，完成以下流程：

```text
stock_data -> stock_news -> stock_quant -> stock_analyst
```

用户通过独立股票门户提交 6 位 A 股代码，系统抓取行情与新闻、运行量化模型、由 LLM 生成中文分析报告，并在图表中展示技术指标、支撑阻力、趋势箭头和 LLM 观点。

## 2. Users and Roles

- 普通用户：登录后可维护个人股票池、查看 K 线图和中文分析报告。
- 默认用户：`default`，用于承接历史迁移数据。
- 技术运维：通过系统监控门户查看运行、日志和指标。

## 3. Functional Requirements

### 3.1 数据获取

- 输入必须是 6 位 A 股代码，否则拒绝。
- 使用 AkShare/东方财富获取 3 年日线 OHLCV（前复权）。
- 首次拉取全量，后续只补缺失区间。
- 本地计算周线 `W-FRI`、月线 `ME`。
- 计算 MACD(12/26/9)、RSI14、MA(20/66/154/250)、波动率、量比和收益率。
- 提供公司名称与行业；东方财富失败时依次回退巨潮资讯和雪球。

### 3.2 新闻

- 来源：
  - 东方财富个股/行业新闻。
  - 巨潮资讯官方公告。
  - 第二媒体源（财联社、同花顺快讯）用公司简称/行业关键词匹配。
- 东方财富与巨潮按标题相似度（字符二元组 Jaccard ≥ 0.5）和日期（±2 天）交叉验证。
- 至少一个来源成功即认为新闻阶段成功；两个都失败才报错。

### 3.3 量化预测

- 模型：LSTM（PyTorch，CPU）。
- 周期：
  - 日线：`5d`、`15d` 方向概率。
  - 周线：`1w` 方向概率（周线特征 + LSTM）。
  - 月线：`1mo` 确定性趋势（月线 MACD 柱状值 + 收盘价 vs MA20）。
- 输出 `horizons`，包含 `up_probability / direction / confidence`。
- 提供日线、周线 walk-forward AUC。
- 持久化缓存：按 `symbol + 特征哈希` 存预测结果，相同数据不重复训练。

### 3.4 分析报告

- LLM 默认 `deepseek-v4-pro`，可通过 `.env` 覆盖。
- LLM 输出中文 Markdown 报告 + 结构化结论：
  - `report`：数据概览、MACD、新闻催化、量化信号、LLM 与量化交叉验证、情景展望、风险、免责声明。
  - `summary`：`overall`（bullish/bearish/neutral）+ 一句中文总结。
- 无 LLM 时使用确定性回退报告。
- 量化 AUC 接近 0.5 时对高概率信号折价，避免强方向结论。

### 3.5 股票门户

- 模拟微信登录，多用户隔离，每个用户独立股票池。
- Analyze：股票已存在则提示“股票已经在股票池中”。
- Refresh：同一天已完成分析则提示“今日股票分析已经生成，无需重复刷新”。
- 每行支持查看结果、错误、图表、刷新、删除。
- 图表：
  - 日/周/月 K 线切换，K 线与 MACD 图同步缩放。
  - MA(20/66/154/250)。
  - MACD(6/13/5)。
  - 斐波那契回撤支撑/阻力，按当前周期分别计算。
  - 日线 `5d/15d` 趋势箭头。
  - 周线、月线趋势和 LLM 观点文字徽标。

## 4. Non-Functional Requirements

- 持久化：SQLite 保存历史行情、量化缓存、用户/会话、股票池和运行记录。
- 并发：orchestrator 共享 checkpoint 连接，避免 `database is locked`。
- 性能：量化结果缓存；运行超时默认 600 秒。
- 语言：门户与报告均为中文。
- 安全：股票门户所有数据接口需要 Bearer token。

## 5. Interfaces

### Orchestrator

- `POST /runs`、`GET /runs/{run_id}`、`GET /runs`、`POST /runs/{run_id}/cancel`

### Stock Portal

- `POST /api/login`、`GET /api/me`
- `GET/POST /api/watchlist`
- `POST /api/watchlist/{symbol}/refresh`
- `DELETE /api/watchlist/{symbol}`
- `GET /api/watchlist/{symbol}/chart?period=daily|weekly|monthly`

## 6. Defaults and Assumptions

- 3 年历史，MACD 技术信号默认 12/26/9，图表 MACD 为 6/13/5。
- 周线用 `W-FRI`，月线用 `ME`。
- 月线为确定性技术趋势，不训练模型；代理概率 0.7/0.3/0.5。
- 斐波那契回撤位：0%、23.6%、38.2%、50%、61.8%、78.6%、100%。
- 现有历史数据迁移到 `default` 用户。
