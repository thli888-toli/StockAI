from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stockportal import refresh_reports  # noqa: E402
from stockportal.refresh_reports import (  # noqa: E402
    create_runs,
    distinct_watchlist_symbols,
    poll_runs,
    refresh_symbol_reports,
    run_cli,
)
from stockportal.store import WatchlistStore  # noqa: E402


def _outputs(symbol: str) -> dict:
    return {
        "market_data": json.dumps(
            {
                "symbol": symbol,
                "company_name": f"公司{symbol}",
                "industry": "测试行业",
            },
            ensure_ascii=False,
        ),
        "news": "{}",
        "quant": "{}",
        "fundamental": json.dumps(
            {
                "summary": {
                    "valuation_verdict": "合理",
                    "fair_value_range": {"low": 10.0, "mid": 12.0, "high": 15.0},
                }
            },
            ensure_ascii=False,
        ),
        "report": f"# 报告 {symbol}",
    }


def _seed_store(db_path: Path) -> WatchlistStore:
    store = WatchlistStore(db_path)
    store.upsert("user-a", "600519", run_id="old-a", status="completed", outputs={"report": "# old"})
    store.upsert("user-b", "600519", run_id="old-b", status="running", outputs={})
    store.upsert("user-a", "300024", run_id="old-c", status="completed", outputs={"report": "# old"})
    return store


def _fake_orchestrator(monkeypatch, outputs_by_symbol: dict[str, dict] | None = None):
    runs: dict[str, dict] = {}
    calls: list[str] = []

    def post_json(url, payload, timeout):
        calls.append(f"post:{payload['query']}")
        run_id = f"run-{len(runs) + 1}"
        run = {
            "run_id": run_id,
            "query": payload["query"],
            "status": "queued",
            "outputs": {},
        }
        runs[run_id] = run
        return run

    def get_json(url, timeout):
        run_id = url.rstrip("/").rsplit("/", 1)[-1]
        run = runs[run_id]
        if outputs_by_symbol:
            outputs = outputs_by_symbol.get(run["query"], _outputs(run["query"]))
        else:
            outputs = _outputs(run["query"])
        run["status"] = "completed"
        run["outputs"] = outputs
        return run

    monkeypatch.setattr(refresh_reports, "_post_json", post_json)
    monkeypatch.setattr(refresh_reports, "_get_json", get_json)
    return runs, calls


def _fake_submit_orchestrator(monkeypatch):
    posts: list[str] = []

    def post_json(url, payload, timeout):
        posts.append(payload["query"])
        return {
            "run_id": f"run-{payload['query']}",
            "query": payload["query"],
            "status": "queued",
        }

    def get_json(url, timeout):
        raise AssertionError("refresh_symbol_reports should not poll runs")

    monkeypatch.setattr(refresh_reports, "_post_json", post_json)
    monkeypatch.setattr(refresh_reports, "_get_json", get_json)
    return posts


def test_refresh_submits_runs_without_polling(monkeypatch, tmp_path):
    db_path = tmp_path / "portal.db"
    store = _seed_store(db_path)
    posts = _fake_submit_orchestrator(monkeypatch)

    results = refresh_symbol_reports(
        ["600519", "300024"], orchestrator_url="http://orch", db_path=db_path
    )
    by_symbol = {item["symbol"]: item for item in results}
    assert by_symbol["600519"]["status"] == "submitted"
    assert by_symbol["600519"]["run_id"] == "run-600519"
    assert by_symbol["300024"]["status"] == "submitted"
    assert posts == ["600519", "300024"]

    # Submission points watchlist rows at the new run, but does not poll for outputs.
    for user_id in ("user-a", "user-b"):
        item = store.get(user_id, "600519")
        assert item["run_id"] == "run-600519"
        assert item["status"] == "queued"
        assert item["outputs"] == {}


def test_refresh_submits_again_regardless_of_existing_state(monkeypatch, tmp_path):
    db_path = tmp_path / "portal.db"
    store = _seed_store(db_path)
    posts = _fake_submit_orchestrator(monkeypatch)

    results = refresh_symbol_reports(
        ["600519"], orchestrator_url="http://orch", db_path=db_path
    )
    assert results[0]["status"] == "submitted"
    assert posts == ["600519"]
    assert store.get("user-a", "600519")["run_id"] == "run-600519"


def test_refresh_reports_submission_failure(monkeypatch, tmp_path):
    db_path = tmp_path / "portal.db"
    _seed_store(db_path)

    def post_json(url, payload, timeout):
        if payload["query"] == "600519":
            return {}
        return {
            "run_id": "run-300024",
            "query": payload["query"],
            "status": "queued",
        }

    monkeypatch.setattr(refresh_reports, "_post_json", post_json)

    results = refresh_symbol_reports(
        ["600519", "300024"], orchestrator_url="http://orch", db_path=db_path
    )
    assert results[0]["status"] == "failed"
    assert "run_id" in results[0]["error"]
    assert results[1]["status"] == "submitted"
    assert results[1]["run_id"] == "run-300024"


def test_refresh_invalid_symbol(monkeypatch, tmp_path):
    db_path = tmp_path / "portal.db"
    calls: list[str] = []

    def post_json(url, payload, timeout):
        calls.append(payload["query"])
        return {
            "run_id": f"run-{payload['query']}",
            "query": payload["query"],
            "status": "queued",
        }

    monkeypatch.setattr(refresh_reports, "_post_json", post_json)
    results = refresh_symbol_reports(
        ["ABC", "600519"], orchestrator_url="http://orch", db_path=db_path
    )
    assert results[0]["status"] == "failed"
    assert "6-digit" in results[0]["error"]
    assert results[1]["status"] == "submitted"
    assert calls == ["600519"]


def test_refresh_submits_without_watchlist_rows(monkeypatch, tmp_path):
    db_path = tmp_path / "portal.db"
    posts = _fake_submit_orchestrator(monkeypatch)
    results = refresh_symbol_reports(
        ["688999"], orchestrator_url="http://orch", db_path=db_path
    )
    assert results[0]["status"] == "submitted"
    assert results[0]["run_id"] == "run-688999"
    assert results[0]["updated_rows"] == 0
    assert posts == ["688999"]


def test_create_runs_submits_all_symbols_upfront(monkeypatch):
    posts: list[str] = []

    def post_json(url, payload, timeout):
        posts.append(payload["query"])
        return {"run_id": f"run-{payload['query']}", "query": payload["query"], "status": "queued"}

    monkeypatch.setattr(refresh_reports, "_post_json", post_json)
    runs, failures = create_runs(["600519", "300024", "600110"], "http://orchestrator")
    assert failures == []
    assert posts == ["600519", "300024", "600110"]
    assert set(runs) == {"600519", "300024", "600110"}


def test_create_runs_reports_per_symbol_failures(monkeypatch):
    def post_json(url, payload, timeout):
        if payload["query"] == "600519":
            return {}
        return {"run_id": f"run-{payload['query']}", "query": payload["query"], "status": "queued"}

    monkeypatch.setattr(refresh_reports, "_post_json", post_json)
    runs, failures = create_runs(["600519", "300024"], "http://orchestrator")
    assert "600519" not in runs
    assert failures == [("600519", "orchestrator did not return a run_id for 600519")]
    assert "300024" in runs


def test_poll_runs_polls_until_terminal(monkeypatch):
    states = {
        "run-600519": iter(["queued", "running", "completed"]),
        "run-300024": iter(["queued", "completed"]),
    }

    def get_json(url, timeout):
        run_id = url.rstrip("/").rsplit("/", 1)[-1]
        status = next(states[run_id])
        return {"run_id": run_id, "status": status, "outputs": {}}

    monkeypatch.setattr(refresh_reports, "_get_json", get_json)
    runs = {
        "600519": {"run_id": "run-600519", "status": "queued"},
        "300024": {"run_id": "run-300024", "status": "queued"},
    }
    result = poll_runs(runs, "http://orchestrator", run_timeout=600.0, poll_interval=0.01)
    assert result["600519"]["status"] == "completed"
    assert result["300024"]["status"] == "completed"


def test_poll_runs_times_out_running_run(monkeypatch):
    def get_json(url, timeout):
        return {"run_id": "run-x", "status": "running", "outputs": {}}

    monkeypatch.setattr(refresh_reports, "_get_json", get_json)
    runs = {"600519": {"run_id": "run-x", "status": "queued"}}
    result = poll_runs(runs, "http://orchestrator", run_timeout=0.02, poll_interval=0.005)
    assert result["600519"]["status"] == "failed"
    assert "timed out" in result["600519"]["error"]


def test_poll_runs_marks_disappeared_run_failed(monkeypatch):
    def get_json(url, timeout):
        raise RuntimeError("404 run not found")

    monkeypatch.setattr(refresh_reports, "_get_json", get_json)
    runs = {"600519": {"run_id": "run-x", "status": "queued"}}
    result = poll_runs(runs, "http://orchestrator", run_timeout=600.0, poll_interval=0.01)
    assert result["600519"]["status"] == "failed"
    assert "run poll failed after 3 attempts" in result["600519"]["error"]


def test_poll_runs_survives_transient_poll_failures(monkeypatch):
    attempts = {"run-x": 0}

    def get_json(url, timeout):
        attempts["run-x"] += 1
        if attempts["run-x"] < 3:
            raise RuntimeError("orchestrator busy")
        return {"run_id": "run-x", "status": "completed", "outputs": {}}

    monkeypatch.setattr(refresh_reports, "_get_json", get_json)
    runs = {"600519": {"run_id": "run-x", "status": "queued"}}
    result = poll_runs(runs, "http://orchestrator", run_timeout=600.0, poll_interval=0.01)
    assert result["600519"]["status"] == "completed"


def test_run_cli_parses_symbols_and_exit_codes(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_refresh(symbols, orchestrator_url=None, db_path=None):
        captured["symbols"] = symbols
        return [
            {"symbol": "600519", "status": "submitted", "run_id": "r1"},
            {"symbol": "300024", "status": "failed", "error": "boom"},
        ]

    monkeypatch.setattr(refresh_reports, "refresh_symbol_reports", fake_refresh)
    code = run_cli("600519, 300024", False, "http://orch", tmp_path / "p.db")
    assert code == 1
    assert captured["symbols"] == ["600519", "300024"]

    monkeypatch.setattr(
        refresh_reports,
        "refresh_symbol_reports",
        lambda symbols, orchestrator_url=None, db_path=None: [
            {"symbol": "600519", "status": "submitted", "run_id": "r1"}
        ],
    )
    assert run_cli("600519", False, "http://orch", tmp_path / "p.db") == 0


def test_run_cli_accepts_chinese_comma_and_dunhao(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_refresh(symbols, orchestrator_url=None, db_path=None):
        captured["symbols"] = symbols
        return []

    monkeypatch.setattr(refresh_reports, "refresh_symbol_reports", fake_refresh)
    run_cli("600487，600110、300024  600519", False, "http://orch", tmp_path / "p.db")
    assert captured["symbols"] == ["600487", "600110", "300024", "600519"]


def test_run_cli_all_refreshes_distinct_symbols(monkeypatch, tmp_path):
    db_path = tmp_path / "portal.db"
    store = _seed_store(db_path)
    store.upsert("user-c", "600110", status="failed", outputs={})
    captured: dict = {}

    def fake_refresh(symbols, orchestrator_url=None, db_path=None):
        captured["symbols"] = symbols
        return [
            {
                "symbol": symbol,
                "status": "submitted",
                "run_id": f"r-{symbol}",
            }
            for symbol in symbols
        ]

    monkeypatch.setattr(refresh_reports, "refresh_symbol_reports", fake_refresh)
    code = run_cli(None, True, "http://orch", db_path)
    assert code == 0
    assert captured["symbols"] == ["300024", "600110", "600519"]


def test_run_cli_all_with_empty_db_does_nothing(monkeypatch, tmp_path):
    db_path = tmp_path / "empty.db"
    called = []

    def fake_refresh(symbols, orchestrator_url):
        called.append(symbols)
        return []

    monkeypatch.setattr(refresh_reports, "refresh_symbol_reports", fake_refresh)
    code = run_cli(None, True, "http://orch", db_path)
    assert code == 0
    assert called == []


def test_run_cli_requires_symbols_or_all(monkeypatch, tmp_path):
    monkeypatch.setattr(
        refresh_reports,
        "refresh_symbol_reports",
        lambda *args, **kwargs: [],
    )
    code = run_cli(None, False, "http://orch", tmp_path / "p.db")
    assert code == 2


def test_distinct_watchlist_symbols_dedupes_across_users(tmp_path):
    db_path = tmp_path / "portal.db"
    store = _seed_store(db_path)
    store.upsert("user-b", "600519", outputs={})
    store.upsert("user-c", "600110", outputs={})
    assert distinct_watchlist_symbols(db_path) == ["300024", "600110", "600519"]


def test_main_parser_has_refresh_reports_command():
    from main import build_parser

    args = build_parser().parse_args(["refresh-reports", "--symbols", "600519,300024"])
    assert args.command == "refresh-reports"
    assert args.symbols == "600519,300024"
    assert args.all is False

    args_all = build_parser().parse_args(["refresh-reports", "--all"])
    assert args_all.all is True
    assert args_all.symbols is None

    with pytest.raises(SystemExit):
        refresh_reports.main(["refresh-reports"])
