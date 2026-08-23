from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

import httpx
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stockportal.app as stock_portal_app  # noqa: E402
from stockportal.store import WatchlistStore  # noqa: E402
from stockportal.auth import AuthStore  # noqa: E402


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError(f"HTTP {self.status_code}")


def _run_payload(run_id: str, status: str, query: str, outputs=None, error=None) -> dict:
    return {
        "run_id": run_id,
        "graph_config": "orchestration.yaml",
        "status": status,
        "query": query,
        "outputs": outputs or {},
        "error": error,
        "events": [],
    }


def _make_app(tmp_path, orchestrator_url="http://fake-orchestrator"):
    return stock_portal_app.create_stock_portal_app(
        orchestrator_url=orchestrator_url,
        db_path=str(tmp_path / "watchlist.db"),
    )


def _auth_headers(client, nickname: str = "test_user") -> dict:
    response = client.post("/api/login", json={"nickname": nickname})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_stock_portal_create_run_proxies_orchestrator(monkeypatch, tmp_path):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse(200, _run_payload("r1", "running", json["query"]))

    monkeypatch.setattr(stock_portal_app.httpx, "post", fake_post)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"query": "600519"})
    assert response.status_code == 200
    assert response.json()["run_id"] == "r1"
    assert captured["json"]["query"] == "600519"


def test_stock_portal_get_run_proxies_orchestrator(monkeypatch, tmp_path):
    def fake_get(url, timeout=None):
        return FakeResponse(
            200,
            _run_payload("r-complete", "completed", "600519", outputs={"report": "# report"}),
        )

    monkeypatch.setattr(stock_portal_app.httpx, "get", fake_get)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/runs/r-complete")
    assert response.status_code == 200
    assert response.json()["run_id"] == "r-complete"


def test_watchlist_add_starts_run(monkeypatch, tmp_path):
    def fake_post(url, json=None, timeout=None):
        return FakeResponse(200, _run_payload("r1", "running", json["query"]))

    monkeypatch.setattr(stock_portal_app.httpx, "post", fake_post)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        headers = _auth_headers(client)
        response = client.post("/api/watchlist", json={"query": "600519"}, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "600519"
    assert body["status"] == "running"
    assert body["run_id"] == "r1"


def test_watchlist_add_marks_failed_when_orchestrator_down(monkeypatch, tmp_path):
    def fake_post(url, json=None, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(stock_portal_app.httpx, "post", fake_post)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        headers = _auth_headers(client)
        response = client.post("/api/watchlist", json={"query": "600519"}, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "600519"
    assert body["status"] == "failed"
    assert body["run_id"] is None
    assert "orchestrator run creation failed" in body["error"]


def test_watchlist_rejects_bad_symbol(monkeypatch, tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        headers = _auth_headers(client)
        response = client.post("/api/watchlist", json={"query": "AAPL"}, headers=headers)
    assert response.status_code == 422


def test_watchlist_requires_auth(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/watchlist").status_code == 401


def test_login_reuses_user_by_nickname(tmp_path):
    auth = AuthStore(tmp_path / "auth.db")
    first = auth.login(nickname="thli88")
    second = auth.login(nickname="thli88")
    assert second["user_id"] == first["user_id"]


def test_watchlist_is_isolated_per_user(monkeypatch, tmp_path):
    def fake_post(url, json=None, timeout=None):
        return FakeResponse(200, _run_payload("r1", "running", json["query"]))

    monkeypatch.setattr(stock_portal_app.httpx, "post", fake_post)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        headers_a = _auth_headers(client, "user_a")
        headers_b = _auth_headers(client, "user_b")
        client.post("/api/watchlist", json={"query": "600519"}, headers=headers_a)
        assert client.get("/api/watchlist", headers=headers_b).json() == []
        assert len(client.get("/api/watchlist", headers=headers_a).json()) == 1


def test_watchlist_duplicate_add_returns_already_exists(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_post(url, json=None, timeout=None):
        calls["count"] += 1
        return FakeResponse(200, _run_payload(f"r{calls['count']}", "running", json["query"]))

    monkeypatch.setattr(stock_portal_app.httpx, "post", fake_post)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        headers = _auth_headers(client)
        client.post("/api/watchlist", json={"query": "600519"}, headers=headers)
        second = client.post("/api/watchlist", json={"query": "600519"}, headers=headers)

    assert second.status_code == 200
    assert second.json() == {"already_exists": True, "message": "股票已经在股票池中"}
    assert calls["count"] == 1


def test_watchlist_refresh_existing_and_404_missing(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_post(url, json=None, timeout=None):
        calls["count"] += 1
        return FakeResponse(200, _run_payload(f"r{calls['count']}", "running", json["query"]))

    monkeypatch.setattr(stock_portal_app.httpx, "post", fake_post)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        headers = _auth_headers(client)
        client.post("/api/watchlist", json={"query": "600519"}, headers=headers)
        refreshed = client.post("/api/watchlist/600519/refresh", headers=headers)
        missing = client.post("/api/watchlist/000001/refresh", headers=headers)

    assert refreshed.status_code == 200
    assert refreshed.json()["run_id"] == "r2"
    assert missing.status_code == 404


def test_watchlist_refresh_blocks_when_completed_today(tmp_path):
    db_path = tmp_path / "watchlist.db"
    app = stock_portal_app.create_stock_portal_app(
        orchestrator_url="http://fake-orchestrator",
        db_path=str(db_path),
    )
    with TestClient(app) as client:
        login_response = client.post("/api/login", json={"nickname": "user"})
        user_id = login_response.json()["user_id"]
        headers = {"Authorization": f"Bearer {login_response.json()['token']}"}
        WatchlistStore(db_path).upsert(
            user_id,
            "600519",
            run_id="r1",
            status="completed",
            outputs={"report": '{"report":"# r","summary":{"overall":"neutral","text":"中性"}}'},
        )
        response = client.post("/api/watchlist/600519/refresh", headers=headers)
    assert response.status_code == 200
    assert response.json()["already_generated"] is True
    assert "无需重复刷新" in response.json()["message"]


def test_watchlist_remove_deletes_and_404_missing(monkeypatch, tmp_path):
    def fake_post(url, json=None, timeout=None):
        return FakeResponse(200, _run_payload("r1", "running", json["query"]))

    monkeypatch.setattr(stock_portal_app.httpx, "post", fake_post)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        headers = _auth_headers(client)
        client.post("/api/watchlist", json={"query": "600519"}, headers=headers)
        removed = client.delete("/api/watchlist/600519", headers=headers)
        missing = client.delete("/api/watchlist/600519", headers=headers)

    assert removed.status_code == 200
    assert removed.json() == {"deleted": True}
    assert missing.status_code == 404


def test_watchlist_list_syncs_status_and_metadata(monkeypatch, tmp_path):
    def fake_post(url, json=None, timeout=None):
        return FakeResponse(200, _run_payload("r1", "running", json["query"]))

    def fake_get(url, timeout=None):
        outputs = {
            "market_data": '{"symbol":"600519","company_name":"贵州茅台","industry":"白酒"}',
            "report": "# report",
        }
        return FakeResponse(
            200,
            [_run_payload("r1", "completed", "600519", outputs=outputs)],
        )

    monkeypatch.setattr(stock_portal_app.httpx, "post", fake_post)
    monkeypatch.setattr(stock_portal_app.httpx, "get", fake_get)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        headers = _auth_headers(client)
        client.post("/api/watchlist", json={"query": "600519"}, headers=headers)
        response = client.get("/api/watchlist", headers=headers)

    assert response.status_code == 200
    item = response.json()[0]
    assert item["status"] == "completed"
    assert item["company_name"] == "贵州茅台"
    assert item["industry"] == "白酒"
    assert item["outputs"]["report"] == "# report"


def test_watchlist_list_marks_missing_running_run_failed(monkeypatch, tmp_path):
    def fake_post(url, json=None, timeout=None):
        return FakeResponse(200, _run_payload("r1", "running", json["query"]))

    def fake_get(url, timeout=None):
        return FakeResponse(200, [])

    monkeypatch.setattr(stock_portal_app.httpx, "post", fake_post)
    monkeypatch.setattr(stock_portal_app.httpx, "get", fake_get)
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        headers = _auth_headers(client)
        client.post("/api/watchlist", json={"query": "600519"}, headers=headers)
        response = client.get("/api/watchlist", headers=headers)

    assert response.status_code == 200
    item = response.json()[0]
    assert item["status"] == "failed"
    assert "run no longer available" in item["error"]


def test_watchlist_store_roundtrip_and_delete(tmp_path):
    store = WatchlistStore(tmp_path / "watchlist.db")
    store.upsert(
        "default",
        "600519",
        run_id="r1",
        status="completed",
        outputs={"report": "# report"},
        company_name="贵州茅台",
        industry="白酒",
    )
    item = store.get("default", "600519")
    assert item is not None
    assert item["status"] == "completed"
    assert item["outputs"]["report"] == "# report"
    assert item["company_name"] == "贵州茅台"
    assert store.delete("default", "600519") is True
    assert store.get("default", "600519") is None
    assert store.delete("default", "600519") is False


def test_watchlist_migrates_legacy_rows_to_default_user(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE watchlist (
            symbol TEXT PRIMARY KEY,
            company_name TEXT NOT NULL DEFAULT '',
            industry TEXT NOT NULL DEFAULT '',
            run_id TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            error TEXT,
            outputs TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO watchlist(
            symbol, company_name, industry, run_id, status, error, outputs, created_at, updated_at
        )
        VALUES('600519', '贵州茅台', '白酒', NULL, 'completed', NULL, '{}', '2026-01-01', '2026-01-01')
        """
    )
    conn.commit()
    conn.close()

    store = WatchlistStore(db_path)
    items = store.all_items("default")
    assert len(items) == 1
    assert items[0]["symbol"] == "600519"
    assert items[0]["user_id"] == "default"


def test_chart_payload_builds_candles_ma_and_macd():
    dates = pd.bdate_range("2024-01-01", periods=300)
    close = pd.Series(range(100, 400), dtype="float64")
    market_data = {
        "daily_features": [
            {
                "date": day.strftime("%Y-%m-%d"),
                "open": float(price - 1),
                "high": float(price + 1),
                "low": float(price - 2),
                "close": float(price),
                "volume": 1000.0,
            }
            for day, price in zip(dates, close)
        ]
    }

    quant = {
        "horizons": {
            "5d": {"direction": "down"},
            "15d": {"direction": "up"},
            "1w": {"direction": "down"},
            "1mo": {"direction": "flat"},
        }
    }
    daily = stock_portal_app._build_chart_payload("600988", market_data, "daily", quant)
    assert daily is not None
    assert len(daily["candles"]) == 300
    assert set(daily["ma"]) == {"ma20", "ma66", "ma154", "ma250"}
    assert daily["macd"]["histogram"]
    assert daily["signals"] == {"5d": "down", "15d": "up", "1w": "down", "1mo": "flat"}
    assert set(daily["levels"]) == {"daily", "weekly", "monthly"}
    for level_set in daily["levels"].values():
        assert level_set["support"] < level_set["resistance"]

    weekly = stock_portal_app._build_chart_payload("600988", market_data, "weekly")
    assert weekly is not None
    assert len(weekly["candles"]) < 300


def test_summary_from_report_parses_json_and_falls_back_to_none():
    summary = stock_portal_app._summary_from_report(
        '{"report":"# r","summary":{"overall":"bullish","text":"偏多"}}'
    )
    assert summary == {"overall": "bullish", "text": "偏多"}
    assert stock_portal_app._summary_from_report("# plain markdown") is None
    assert stock_portal_app._summary_from_report(None) is None


def test_fibonacci_levels_bracket_last_close():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "open": [10.0] * 10,
            "high": [10, 12, 10, 9, 14, 10, 9, 13, 10, 11],
            "low": [10, 9, 9, 8, 9, 8, 7, 9, 9, 10],
            "close": [10, 11, 10, 9, 12, 10, 8, 12, 10, 10.5],
            "volume": [1.0] * 10,
        }
    )
    support, resistance = stock_portal_app._fibonacci_levels(frame)
    assert support < 10.5 < resistance
