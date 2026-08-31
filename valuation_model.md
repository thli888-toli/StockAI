# 估值参数与本地估值模型说明

> **以后遇到估值问题，优先看是不是这几个参数或原有参数能解决；确实没有对应参数时再改代码。**
> 改代码后，把新行为参数化并同步更新本文档，保证下次可以直接通过参数微调。

本文档是估值模块的唯一参数事实源。运行时参数全部来自 `config/valuation/default.json`
（默认值），个股可用 `config/valuation/{symbol}.json` 覆盖其中任意键（flat 键、深合并、
未出现的键继承 default）。未知键保留不报错，方便未来加参数。

---

## 参数总表

### 基础与 DCF

| 参数 | 默认值 | 说明 |
|---|---|---|
| `config_version` | "1.0" | 配置版本号（元数据，不参与计算）。 |
| `discount_rate` | 0.10 | DCF / DDM 折现率。调低则估值抬升。 |
| `terminal_growth` | 0.02 | DCF 永续增速（须小于折现率）。 |
| `forecast_years` | 5 | DCF 显性期年数。 |
| `fallback_growth` | 0.05 | 缺少历史 CAGR 与一致预期增速时的默认增长假设。 |
| `max_growth` | 0.30 | 增长假设上限（DCF/DDM 共用）。 |
| `sensitivity_rates` | [0.08, 0.12] | DCF/DDM 敏感性分析的折现率集合。 |
| `growth_cagr_weight` | 0.6 | DCF 增长假设中"近3年营收 CAGR"的权重，一致预期权重 = 1 − 该值。 |

### DDM（股息折现）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `min_dividend_yield` | 0.01 | DDM 最低股息率门槛，低于则判定不适用。 |
| `ddm_growth_cap` | 0.10 | DDM 可持续增长率 g = ROE×(1−分红率) 的上限。 |

### 相对估值

| 参数 | 默认值 | 说明 |
|---|---|---|
| `target_percentile` | 0.50 | 龙头分支历史锚使用的自身历史分位：≤0.25 用 p25，≥0.75 用 p75，其余用 p50（默认）。 |
| `verdict_band` | 0.10 | 低估/合理/高估判定带宽：当前价相对中枢偏离在 ±band 内判"合理"。 |
| `method_weights` | {relative:1.0, dcf:0.8, ddm:0.5} | 三种方法综合时的权重（加权中位数）。 |
| `outlier_band` | [0.2, 5.0] | 方法间离群剔除带：某方法价格低于中位数×0.2 或高于×5 时剔除。 |
| `cyclical_keywords` | 证券/券商/银行/保险/有色/钢铁/煤炭/航运/地产/航空 | 行业名命中任一关键词判为周期行业，PE-TTM 不参与相对估值。 |
| `peg_factor` | 1.0 | PE 目标倍数按 PEG 校准系数：目标 = 增速% × factor。 |
| `min_peg_growth` | 0.15 | PEG 校准最低增速，低于则不校准。 |
| `peg_band` | [0.6, 1.5] | PEG 校准钳制带：校准结果限制在原始倍数的 [低, 高] 倍内。 |
| `target_band` | 0.15 | 缺少历史 p25/p75 时，用目标倍数 ±band 构造区间。 |
| `metric_outlier_factor` | 4.0 | 口径内异常放大剔除：隐含价超过口径中位数该倍数时剔除。 |
| `history_weight` | 0.8 | 相对估值中"自身历史分位"混入的权重（普通公司）。 |
| `history_cap` | 3.0 | 自身历史锚参与的上限倍数（防止历史泡沫倍数主导）。 |
| `peer_industry_band` | [0.6, 1.6] | 同行中位数相对行业整体中位数的合理偏离带，超带改用行业。 |
| `growth_leader_threshold` | 0.30 | 高成长龙头分支判定阈值（一致预期增速 ≥ 该值）。 |
| `peer_own_premium_factor` | 2.0 | 溢价龙头保护：当前 PB/PS > 同行中位数×该值时，锚定自身历史 p50；无历史则弃用该口径。调低更易触发（估值抬升）。 |
| `leader_history_weight` | 0.5 | 龙头分支（有 forward 数据时）自身历史锚的校准权重。 |
| `leader_history_cap_factor` | 1.5 | 龙头分支自身历史锚相对主锚倍数的上限系数。 |
| `leader_primary_min_weight` | 1.0 | 龙头分支（无 forward 数据）中枢取中位的主锚权重门槛：只统计权重 ≥ 该值的口径；调低可让参考锚（如本地模型）参与中枢。 |
| `ps_peer_divergence_factor` | 2.5 | 同行 PS 中位数超过自身当前 PS 该倍数时，弃用 PS 口径。 |
| `cyclical_boom_pe_ratio` | 0.5 | 周期景气高点检测：当前 PE ≤ 同行/历史中位数×该值时判盈利见顶，弃用 PE（主路径与龙头分支均生效）。 |

### 重组/资产注入检测

| 参数 | 默认值 | 说明 |
|---|---|---|
| `restructuring_pb_ratio` | 10.0 | 疑似重组判定 PB 门槛。 |
| `restructuring_ps_ratio` | 10.0 | 疑似重组判定 PS 门槛。 |
| `restructuring_max_eps` | 0.5 | 重组检测 EPS 上限（盈利高于该值不判重组）。 |
| `restructuring_min_bps` | 1.0 | 重组检测每股净资产下限（双低才可能判重组）。 |
| `restructuring_min_sps` | 1.0 | 重组检测每股营收下限（双低才可能判重组）。 |
| `peer_mismatch_keywords` | ["不匹配","应替换为"] | LLM 校验同行名单时，提示文本命中任一关键词则同行降级为参考。 |

### 手动同行名单

| 参数 | 默认值 | 说明 |
|---|---|---|
| `manual_peers` | 无 | 手动指定可比公司，优先级高于 `config/fundamental_peers.yaml`；写 `[]` 表示清空手动名单走自动同行。条目可为 6 位代码或 {"code","name"}。注意：名单仍会经过 LLM 校验细化。 |

### 人工估值覆盖

| 参数 | 默认值 | 说明 |
|---|---|---|
| `manual_fair_value` | 无 | 报表失真（资产注入/重组/私有化未并表）时的个股人工估值覆盖。对象结构 `{"low": 15, "mid": 18, "high": 21, "note": "说明"}`；mid 有效时跳过常规估值管线，报告显示"人工估值区间（待并表确认）"，低估/合理/高估仍按 mid 与现价自动判定。 |

### 本地 LightGBM 模型

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_enabled` | true | 本地模型总开关。 |
| `model_min_confidence` | 0.5 | 本地模型最低置信度（特征完整度），低于则不参与估值。 |
| `model_anchor_weight` | 0.5 | 本地模型目标倍数混入相对估值的权重（普通公司路径）。 |
| `model_models_dir` | "state/valuation_model" | 模型产物目录（训练数据、模型文件、meta.json）。 |

> 注意：`model_anchor_weight` 设为 0 只会让模型锚"零权重参与"，但仍会占住
> 截尾均值的排序位置影响剔除结果；要彻底移除模型锚（例如跨行业模型对
> 高成长机器人/AI 公司明显低估时），请用 `model_enabled: false`。

---

## 本地 LightGBM 估值倍数模型

没有成熟的开源"A股估值专用小模型"可直接下载，本项目用全市场 A 股财报数据自训：

- **训练命令**：`python -m main train-valuation-model`（可选 `--report-date YYYYMMDD`）。
  自动跳过未来/取不到数的报告期，行情源东财失败时自动回退腾讯接口。
- **数据**：最近一期业绩报表横截面（约 5000+ 支）为特征，当前行情 PE-TTM/PB/PS 为标签。
- **模型**：三个独立 LightGBM 回归器（PE/PB/PS），目标取 log，行业做分类变量。
- **产物**：`state/valuation_model/{pe,pb,ps}_model.joblib` + `meta.json` + `training.csv`。
- **推理**：`get_model_targets` 工具按个股财务指标推理目标倍数，毫秒级、无网络；
  模型未训练时优雅降级（返回"模型未训练"，不影响主链路）。
- **参与方式**：相对估值各口径按 `model_anchor_weight` 混入模型倍数，并作为
  同行/历史/行业全部缺失时的最后兜底；高成长龙头分支（无 forward 数据）中
  模型倍数默认只作参考，是否参与中枢由 `leader_primary_min_weight` 控制。

---

## 排查与微调流程

1. 先看报告里的"目标倍数层级/口径明细/notes"，定位是哪条口径在压低或抬高中枢。
2. 对照上表找对应参数，在 `config/valuation/{symbol}.json` 覆盖该股默认值并刷新验证。
3. 常用旋钮速查：
   - 同行倍数偏低 → `manual_peers` 换名单；
   - 本地模型锚偏低 → `model_anchor_weight` 调低或 `model_enabled: false`；
   - PE 被 PEG 压低 → `peg_factor` 调高或 `peg_band` 放宽；
   - 龙头被自身历史锚定 → `leader_primary_min_weight`、`leader_history_weight`、`leader_history_cap_factor`；
   - 溢价龙头被同行拉低 → `peer_own_premium_factor` 调低；
   - DCF 偏低 → `discount_rate` 调低、`growth_cagr_weight` 调向一致预期。
4. 以上都无法解决，才考虑改代码；改完把新行为参数化并更新本文档。
