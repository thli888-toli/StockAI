"""CLI entry point for the Day 5 local stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 5 agent orchestration framework")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("registry", help="Run the live registry")

    agent_parser = sub.add_parser("agent", help="Run one agent service")
    agent_parser.add_argument("--manifest", required=True)

    orch_parser = sub.add_parser("orchestrator", help="Run the orchestrator")
    orch_parser.add_argument("--manifest", default="config/orchestration.yaml")
    orch_parser.add_argument("--port", type=int, default=8020)
    orch_parser.add_argument(
        "--checkpoint-db",
        default="state/orchestrator.db",
        help="SQLite checkpoint DB path (per-market orchestrators should use separate files)",
    )
    orch_parser.add_argument(
        "--queue-db",
        default="state/orchestrator_queue.db",
        help="SQLite run-queue DB path (per-market orchestrators should use separate files)",
    )

    refresh_parser = sub.add_parser(
        "refresh-reports",
        help="Refresh backend stock reports for a list of symbols (bypasses portal same-day dedupe)",
    )
    refresh_parser.add_argument("--symbols", default=None)
    refresh_parser.add_argument(
        "--all",
        action="store_true",
        help="Refresh every distinct symbol registered in the watchlist DB across all users",
    )
    refresh_parser.add_argument("--orchestrator", default=None)
    refresh_parser.add_argument("--db", default=None)
    refresh_parser.add_argument(
        "--market",
        choices=("a", "us"),
        default="a",
        help="Market to refresh: 'a' for A-share (default), 'us' for US stocks",
    )

    train_parser = sub.add_parser(
        "train-valuation-model",
        help="Train local LightGBM PE/PB/PS target-multiple models from the latest A-share cross-section",
    )
    train_parser.add_argument(
        "--report-date",
        default=None,
        help="Explicit report period YYYYMMDD (default: latest available)",
    )

    sub.add_parser("portal", help="Run the portal backend and serve the built UI")

    sub.add_parser("stockportal", help="Run the dedicated stock analysis portal")

    sub.add_parser("run", help="Run registry, agents, and orchestrator locally")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    from framework.runner import run_agent, run_orchestrator, run_portal_backend, run_registry, run_stock_portal, start_all

    if args.command == "registry":
        run_registry()
    elif args.command == "agent":
        run_agent(args.manifest)
    elif args.command == "orchestrator":
        run_orchestrator(
            args.manifest,
            port=args.port,
            checkpoint_db=args.checkpoint_db,
            queue_db=args.queue_db,
        )
    elif args.command == "refresh-reports":
        from framework.config import ORCHESTRATOR_URL, STOCK_PORTAL_DB, US_ORCHESTRATOR_URL
        from stockportal.refresh_reports import run_cli

        if not args.all and not args.symbols:
            refresh_parser.error("either --symbols or --all is required")
        return run_cli(
            args.symbols,
            args.all,
            (
                US_ORCHESTRATOR_URL
                if args.market == "us"
                else args.orchestrator or ORCHESTRATOR_URL
            ),
            args.db or STOCK_PORTAL_DB,
            args.market,
        )
    elif args.command == "train-valuation-model":
        from plugins.stock_fundamental.valuation_model import train

        summary = train(report_date=args.report_date)
        import json

        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "portal":
        run_portal_backend()
    elif args.command == "stockportal":
        run_stock_portal()
    elif args.command == "run":
        start_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
