"""Backend CLI tool to refresh stock reports without the portal same-day dedupe.

The portal blocks re-running an analysis for the same symbol on the same day.
This tool intentionally bypasses that check: it submits one fresh orchestrator
run per requested symbol and returns immediately. It does not poll for final
results; the orchestrator queue completes the work and the portal syncs
completed runs into watchlist rows.

Usage (from repo root):

    python -m main refresh-reports --symbols 600519,300024
    python -m main refresh-reports --all

`--symbols` refreshes an explicit list; `--all` refreshes every distinct
symbol currently registered in the watchlist DB across all users, writing each
result back to every watchlist row that contains the symbol.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from framework.config import ORCHESTRATOR_URL, STOCK_PORTAL_DB
from stockportal.store import WatchlistStore


SYMBOL_RE = re.compile(r"^\d{6}$")
TERMINAL_STATUSES = ("completed", "failed")
SYMBOL_SEPARATOR_RE = re.compile(r"[,，、\s]+")
MAX_POLL_FAILURES = 3


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def create_runs(
    symbols: list[str],
    orchestrator_url: str,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]]]:
    """Create one orchestrator run per symbol (parallel submission)."""
    base = orchestrator_url.rstrip("/")
    runs_by_symbol: dict[str, dict[str, Any]] = {}
    failures: list[tuple[str, str]] = []
    for symbol in symbols:
        try:
            run = _post_json(f"{base}/runs", {"query": symbol}, timeout=30.0)
            if not run.get("run_id"):
                raise RuntimeError(f"orchestrator did not return a run_id for {symbol}")
            runs_by_symbol[symbol] = run
        except Exception as exc:  # noqa: BLE001
            failures.append((symbol, str(exc)))
    return runs_by_symbol, failures


def poll_runs(
    runs_by_symbol: dict[str, dict[str, Any]],
    orchestrator_url: str,
    run_timeout: float = 1800.0,
    poll_interval: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Poll every run concurrently until each reaches a terminal state.

    Runs are submitted in parallel and the orchestrator queue manages
    concurrency, so the per-run timeout starts when the run first becomes
    ``running`` and queue wait is not counted. A run that disappears from the
    orchestrator (e.g. after a restart) surfaces as a poll failure.
    """
    base = orchestrator_url.rstrip("/")
    running_deadline: dict[str, float] = {}
    poll_failures: dict[str, int] = {}
    while True:
        pending = [
            symbol
            for symbol, run in runs_by_symbol.items()
            if str(run.get("status")) not in TERMINAL_STATUSES
        ]
        if not pending:
            return runs_by_symbol
        for symbol in pending:
            run = runs_by_symbol[symbol]
            status = str(run.get("status") or "queued")
            now = time.monotonic()
            if status == "running" and symbol not in running_deadline:
                running_deadline[symbol] = now + run_timeout
            if symbol in running_deadline and now > running_deadline[symbol]:
                runs_by_symbol[symbol] = {
                    **run,
                    "status": "failed",
                    "error": f"run polling timed out after {run_timeout:g}s",
                }
                continue
            try:
                runs_by_symbol[symbol] = _get_json(
                    f"{base}/runs/{run['run_id']}",
                    timeout=20.0,
                )
                poll_failures[symbol] = 0
            except Exception as exc:  # noqa: BLE001
                poll_failures[symbol] = poll_failures.get(symbol, 0) + 1
                if poll_failures[symbol] >= MAX_POLL_FAILURES:
                    runs_by_symbol[symbol] = {
                        **run,
                        "status": "failed",
                        "error": (
                            f"run poll failed after {MAX_POLL_FAILURES} attempts: {exc}"
                        ),
                    }
                    poll_failures[symbol] = 0
        if any(
            str(run.get("status")) not in TERMINAL_STATUSES
            for run in runs_by_symbol.values()
        ):
            time.sleep(poll_interval)


def distinct_watchlist_symbols(db_path: str | Path) -> list[str]:
    """Return every distinct symbol in the watchlist DB across all users."""
    store = WatchlistStore(db_path)
    with store.lock, store.conn:
        rows = store.conn.execute(
            "SELECT DISTINCT symbol FROM watchlist ORDER BY symbol"
        ).fetchall()
    return [str(row["symbol"]) for row in rows]


def refresh_symbol_reports(
    symbols: list[str],
    orchestrator_url: str = ORCHESTRATOR_URL,
    db_path: str | Path = STOCK_PORTAL_DB,
) -> list[dict[str, Any]]:
    """Submit one orchestrator run per symbol and return immediately.

    The CLI never polls for terminal status. It points every watchlist row for
    each submitted symbol at the new run id and marks it queued; the portal
    then syncs completed runs into those rows later.
    """
    store = WatchlistStore(db_path)
    results_by_symbol: dict[str, dict[str, Any]] = {}
    valid_symbols: list[str] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol or "").strip()
        if not SYMBOL_RE.fullmatch(symbol):
            results_by_symbol[symbol] = {
                "symbol": symbol,
                "status": "failed",
                "error": "symbol must be a 6-digit A-share code",
                "run_id": None,
                "updated_rows": 0,
            }
        else:
            valid_symbols.append(symbol)

    runs_by_symbol, create_failures = create_runs(valid_symbols, orchestrator_url)
    failure_by_symbol = dict(create_failures)
    for symbol in valid_symbols:
        run = runs_by_symbol.get(symbol)
        if run:
            updated_rows = 0
            for row in store.all_by_symbol(symbol):
                store.upsert(
                    row["user_id"],
                    symbol,
                    run_id=run.get("run_id"),
                    status=str(run.get("status") or "queued"),
                    error=run.get("error"),
                    outputs=run.get("outputs") or {},
                    company_name=row.get("company_name") or "",
                    industry=row.get("industry") or "",
                )
                updated_rows += 1
            results_by_symbol[symbol] = {
                "symbol": symbol,
                "status": "submitted",
                "run_id": run.get("run_id"),
                "error": None,
                "updated_rows": updated_rows,
            }
        else:
            results_by_symbol[symbol] = {
                "symbol": symbol,
                "status": "failed",
                "run_id": None,
                "error": failure_by_symbol.get(symbol, "submission failed"),
                "updated_rows": 0,
            }

    return [results_by_symbol[str(raw_symbol or "").strip()] for raw_symbol in symbols]


def run_cli(
    symbols: str | None,
    refresh_all: bool,
    orchestrator_url: str,
    db_path: str | Path,
) -> int:
    if refresh_all:
        symbol_list = distinct_watchlist_symbols(db_path)
        if not symbol_list:
            print("no watchlist symbols found, nothing to refresh")
            return 0
    else:
        symbol_list = [
            item for item in SYMBOL_SEPARATOR_RE.split(symbols or "") if item
        ]
        if not symbol_list:
            print("no symbols provided; use --symbols or --all", file=sys.stderr)
            return 2
    results = refresh_symbol_reports(
        symbol_list,
        orchestrator_url=orchestrator_url,
        db_path=db_path,
    )
    failed = 0
    for item in results:
        if item["status"] == "submitted":
            print(f"SUBMITTED {item['symbol']} run={item.get('run_id')}")
        else:
            failed += 1
            print(
                f"FAILED {item['symbol']}: {item.get('error') or item.get('status')}",
                file=sys.stderr,
            )
    print(f"submitted {len(results) - failed}/{len(results)} reports")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh stock reports in the backend, bypassing the portal same-day dedupe."
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated 6-digit A-share codes, e.g. 600519,300024",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Refresh every distinct symbol registered in the watchlist DB across all users",
    )
    parser.add_argument("--orchestrator", default=ORCHESTRATOR_URL)
    parser.add_argument("--db", default=str(STOCK_PORTAL_DB))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.all and not args.symbols:
        parser.error("either --symbols or --all is required")
    return run_cli(
        args.symbols,
        args.all,
        args.orchestrator,
        args.db,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
