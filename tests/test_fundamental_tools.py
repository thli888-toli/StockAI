from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.stock_fundamental import tools  # noqa: E402
from plugins.stock_fundamental.fundamental_cache import FundamentalCacheStore  # noqa: E402


def _cache(monkeypatch, tmp_path) -> FundamentalCacheStore:
    store = FundamentalCacheStore(tmp_path / "cache.db")
    monkeypatch.setattr(tools, "FUNDAMENTAL_CACHE", store)
    return store


class FakeAk:
    def __init__(self) -> None:
        self.balance_calls = 0

    def stock_balance_sheet_by_report_em(self, symbol):
        assert symbol == "SH600519"
        self.balance_calls += 1
        return pd.DataFrame(
            {
                "REPORT_DATE": ["2026-06-30", "2025-12-31"],
                "TOTAL_ASSETS": [100.0, 90.0],
                "TOTAL_LIABILITIES": [30.0, 28.0],
                "TOTAL_PARENT_EQUITY": [70.0, 62.0],
                "MONETARYFUNDS": [50.0, 45.0],
            }
        )

    def stock_profit_sheet_by_report_em(self, symbol):
        assert symbol == "SH600519"
        return pd.DataFrame(
            {
                "REPORT_DATE": ["2026-06-30", "2025-12-31"],
                "TOTAL_OPERATE_INCOME": [100.0, 180.0],
                "OPERATE_INCOME": [98.0, 175.0],
                "PARENT_NETPROFIT": [40.0, 70.0],
                "BASIC_EPS": [3.0, 5.5],
                "TOTAL_OPERATE_INCOME_YOY": [12.0, 8.0],
            }
        )

    def stock_cash_flow_sheet_by_report_em(self, symbol):
        assert symbol == "SH600519"
        return pd.DataFrame(
            {
                "REPORT_DATE": ["2026-06-30", "2025-12-31"],
                "NETCASH_OPERATE": [55.0, 90.0],
                "CONSTRUCT_LONG_ASSET": [5.0, 8.0],
                "ASSIGN_DIVIDEND_PORFIT": [30.0, 60.0],
            }
        )

    def stock_financial_analysis_indicator_em(self, symbol, indicator):
        assert symbol == "600519.SH"
        assert indicator == "按报告期"
        return pd.DataFrame(
            {
                "REPORT_DATE": ["2026-06-30", "2025-12-31"],
                "EPSJB": [3.0, 5.5],
                "BPS": [60.0, 55.0],
                "ROEJQ": [12.0, 25.0],
                "XSMLL": [90.0, 91.0],
                "XSJLL": [40.0, 42.0],
                "ZCFZL": [30.0, 31.0],
                "TOTALOPERATEREVETZ": [12.0, 8.0],
                "PARENTNETPROFITTZ": [15.0, 10.0],
                "DJD_TOI_YOY": [11.0, 9.0],
                "DJD_DPNP_QOQ": [5.0, -3.0],
            }
        )

    def stock_financial_analysis_indicator(self, symbol, start_year):
        assert symbol == "600519"
        return pd.DataFrame(
            {
                "日期": ["2026-06-30", "2025-12-31"],
                "摊薄每股收益(元)": [3.0, 5.5],
                "每股净资产_调整后(元)": [60.0, 55.0],
                "净资产收益率(%)": [12.0, 25.0],
                "销售毛利率(%)": [90.0, 91.0],
                "资产负债率(%)": [30.0, 31.0],
            }
        )

    def stock_financial_abstract(self, symbol):
        assert symbol == "600519"
        return pd.DataFrame(
            {
                "选项": ["报告期", "报告期"],
                "指标": ["净利润", "营业总收入"],
                "20260630": [40.0, 100.0],
                "20251231": [70.0, 180.0],
            }
        )

    def stock_zygc_em(self, symbol):
        assert symbol == "SH600519"
        return pd.DataFrame(
            {
                "报告日期": ["2026-06-30", "2026-06-30"],
                "分类类型": ["按产品分类", "按产品分类"],
                "主营构成": ["茅台酒", "系列酒"],
                "主营收入": [80.0, 20.0],
                "收入比例": [0.8, 0.2],
                "毛利率": [0.93, 0.72],
            }
        )

    def stock_value_em(self, symbol):
        assert symbol == "600519"
        return pd.DataFrame(
            {
                "数据日期": ["2026-08-24"],
                "当日收盘价": [1304.66],
                "总市值": [1.63e12],
                "流通市值": [1.63e12],
                "总股本": [1_250_081_601],
                "流通股本": [1_250_081_601],
                "PE(TTM)": [20.03],
                "PE(静)": [19.81],
                "市净率": [6.49],
                "PEG值": [-4.8],
                "市现率": [13.69],
                "市销率": [9.41],
            }
        )

    def stock_zh_valuation_baidu(self, symbol, indicator, period):
        assert symbol == "600519"
        assert period == "近三年"
        if indicator == "市盈率(TTM)":
            values = [25.0 + index * 0.1 for index in range(100)]
        elif indicator == "市净率":
            values = [8.0 + index * 0.05 for index in range(100)]
        else:
            values = [10.0 + index * 0.1 for index in range(100)]
        dates = pd.bdate_range("2024-01-01", periods=100)
        return pd.DataFrame({"date": dates, "value": values})

    def stock_zh_valuation_comparison_em(self, symbol):
        raise RuntimeError("eastmoney comparison API unavailable")

    def stock_industry_pe_ratio_cninfo(self, symbol, date):
        assert symbol == "证监会行业分类"
        return pd.DataFrame(
            {
                "行业名称": ["酒、饮料和精制茶制造业", "农、林、牧、渔业"],
                "公司数量": [50.0, 44.0],
                "静态市盈率-中位数": [22.0, 73.51],
                "静态市盈率-算术平均": [30.0, 101.28],
                "静态市盈率-加权平均": [25.0, 19.16],
            }
        )

    def stock_research_report_em(self, symbol):
        assert symbol == "600519"
        return pd.DataFrame(
            {
                "日期": ["2026-08-21"],
                "东财评级": ["买入"],
                "机构": ["西南证券"],
                "行业": ["白酒Ⅱ"],
                "2026-盈利预测-收益": [69.83],
                "2026-盈利预测-市盈率": [18.59],
                "2027-盈利预测-收益": [75.82],
            }
        )

    def stock_yjyg_em(self, date):
        assert date == "20260630"
        return pd.DataFrame(
            {
                "股票代码": ["600519", "000001"],
                "预测指标": ["归属于上市公司股东的净利润", "归属于上市公司股东的净利润"],
                "业绩变动幅度": [15.0, 20.0],
                "预告类型": ["预增", "预增"],
                "公告日期": ["2026-08-19", "2026-08-19"],
            }
        )


@pytest.fixture
def fake_ak(monkeypatch, tmp_path):
    _cache(monkeypatch, tmp_path)
    fake = FakeAk()
    monkeypatch.setattr(tools, "_call_ak", lambda function: function(fake))
    return fake


@pytest.mark.asyncio
async def test_get_company_profile_uses_cninfo_and_market_data(fake_ak):
    frame = pd.DataFrame(
        {
            "公司名称": ["贵州茅台酒股份有限公司"],
            "A股简称": ["贵州茅台"],
            "所属行业": ["酒、饮料和精制茶制造业"],
            "上市日期": ["2001-08-27"],
        }
    )
    fake_ak.stock_profile_cninfo = lambda symbol: frame
    fake_ak.stock_individual_info_em = lambda symbol: pd.DataFrame(
        {"item": ["股票简称"], "value": ["贵州茅台"]}
    )
    profile = await tools.run_tool(
        "get_company_profile",
        "600519",
        {"company_name": "", "industry": ""},
        [],
    )
    assert profile["company_name"] == "贵州茅台酒股份有限公司"
    assert profile["industry"] == "酒、饮料和精制茶制造业"


@pytest.mark.asyncio
async def test_get_financial_statements_normalizes_columns(fake_ak):
    statements = await tools.run_tool("get_financial_statements", "600519", {}, [])
    assert statements["periods"] == ["2026-06-30", "2025-12-31"]
    balance = statements["balance_sheet"][0]
    assert balance["total_assets"] == 100.0
    assert balance["parent_equity"] == 70.0
    income = statements["income_statement"][0]
    assert income["operate_income"] == 98.0
    assert income["total_operate_income_yoy"] == 12.0
    cash = statements["cash_flow"][0]
    assert cash["netcash_operate"] == 55.0
    assert cash["construct_long_asset"] == 5.0


@pytest.mark.asyncio
async def test_get_financial_indicators_merges_sources(fake_ak):
    indicators = await tools.run_tool("get_financial_indicators", "600519", {}, [])
    assert indicators["em"][0]["eps"] == 3.0
    assert indicators["em"][0]["roe"] == 12.0
    assert indicators["sina"][0]["roe"] == 12.0
    assert indicators["abstract"][0]["net_profit"] == 40.0
    assert indicators["latest"]["report_date"] == "2026-06-30"
    assert indicators["latest"]["bps"] == 60.0


@pytest.mark.asyncio
async def test_get_financial_abstract_includes_business_composition(fake_ak):
    abstract = await tools.run_tool("get_financial_abstract", "600519", {}, [])
    assert abstract["periods"][0]["net_profit"] == 40.0
    assert abstract["business_composition"][0]["item"] == "茅台酒"


@pytest.mark.asyncio
async def test_get_valuation_snapshot_parses_value_em(fake_ak):
    snapshot = await tools.run_tool("get_valuation_snapshot", "600519", {}, [])
    assert snapshot["pe_ttm"] == 20.03
    assert snapshot["pb"] == 6.49
    assert snapshot["total_shares"] == 1_250_081_601
    assert snapshot["data_date"] == "2026-08-24"


@pytest.mark.asyncio
async def test_get_historical_valuation_percentile_computes_stats(fake_ak):
    historical = await tools.run_tool("get_historical_valuation_percentile", "600519", {}, [])
    pe = historical["metrics"]["pe_ttm"]
    assert pe["samples"] == 100
    assert 0 <= pe["percentile"] <= 100
    assert pe["p50"] is not None
    pb = historical["metrics"]["pb"]
    assert pb["latest"] is not None


@pytest.mark.asyncio
async def test_get_industry_valuation_comparison_falls_back_to_cninfo(fake_ak):
    industry = await tools.run_tool(
        "get_industry_valuation_comparison",
        "600519",
        {"industry": "酒、饮料和精制茶制造业"},
        [],
    )
    assert industry["source"] == "cninfo"
    assert industry["pe"]["median"] == 22.0
    assert industry["matched_industry"] == "酒、饮料和精制茶制造业"


@pytest.mark.asyncio
async def test_get_industry_valuation_comparison_uses_eastmoney_when_available(fake_ak):
    fake_ak.stock_zh_valuation_comparison_em = lambda symbol: pd.DataFrame(
        {
            "代码": ["600519", "600000"],
            "简称": ["贵州茅台", "浦发银行"],
            "市盈率-TTM": [20.03, 8.0],
            "市净率-MRQ": [6.49, 0.6],
            "市销率-TTM": [9.41, 1.0],
            "行业": ["白酒Ⅱ", "银行"],
        }
    )
    comparison = await tools.run_tool(
        "get_industry_valuation_comparison",
        "600519",
        {"industry": "白酒"},
        [],
    )
    assert comparison["source"] == "eastmoney"
    assert comparison["pe"]["stock"] == 20.03
    assert comparison["pe"]["median"] == pytest.approx(14.015)
    assert comparison["pb"]["median"] == pytest.approx(3.545)


@pytest.mark.asyncio
async def test_get_earnings_forecast_returns_research_and_guidance(fake_ak):
    forecast = await tools.run_tool("get_earnings_forecast", "600519", {}, [])
    assert forecast["research_reports"][0]["eps_2026"] == 69.83
    assert forecast["earnings_guidance"][0]["change_pct"] == 15.0


@pytest.mark.asyncio
async def test_estimate_fair_value_tool_uses_metrics(monkeypatch, tmp_path):
    _cache(monkeypatch, tmp_path)
    result = await tools.run_tool(
        "estimate_fair_value",
        "600519",
        {},
        [],
        metrics={
            "eps_ttm": 2.0,
            "bps": 10.0,
            "sps_ttm": 5.0,
            "fcf": 2e9,
            "dps": 1.0,
            "roe": 0.15,
            "payout_ratio": 0.5,
            "revenue_growth_cagr": 0.1,
            "forecast_growth": 0.08,
            "current_price": 20.0,
            "total_shares": 1e9,
            "valuation": {"pe_ttm": 15.0, "pb": 2.0},
            "historical": {"pe_ttm": {"p50": 18.0}, "pb": {"p50": 2.5}},
            "industry": {},
        },
    )
    assert "fair_value_range" in result


@pytest.mark.asyncio
async def test_run_tool_unknown_name_raises(monkeypatch, tmp_path):
    _cache(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        await tools.run_tool("no_such_tool", "600519", {}, [])


def test_cache_store_roundtrip_and_ttl(tmp_path):
    store = FundamentalCacheStore(tmp_path / "cache.db")
    assert store.get("600519", "t1", "2026-08-24", 3600) is None
    store.put("600519", "t1", "2026-08-24", '{"a":1}')
    assert store.get("600519", "t1", "2026-08-24", 3600) == '{"a":1}'
    assert store.get("600519", "t1", "2026-08-24", -1) is None
