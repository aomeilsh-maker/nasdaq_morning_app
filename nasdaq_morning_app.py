#!/usr/bin/env python3
"""NASDAQ Morning app entrypoint (orchestrator only)."""

from __future__ import annotations

from nasdaq_data_layer import get_x_fetch_status
from nasdaq_report_renderer import render_report
from nasdaq_strategy_layer import (
    build_long_term_views,
    build_picks,
    calc_last5_winrate_and_update_history,
)


def main() -> None:
    picks, regime = build_picks(limit=10)
    winrate = calc_last5_winrate_and_update_history(picks)
    long_views = build_long_term_views()
    x_status = get_x_fetch_status("NVDA")
    render_report(picks, regime, winrate, long_views, x_status)


if __name__ == "__main__":
    main()
