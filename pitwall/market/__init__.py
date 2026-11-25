"""pitwall.market: the engine's odds against market prices (Phase 14).

Paper trading only: nothing here places bets, logs in to a bookmaker or exchange, or
handles money. Real-money online betting is illegal in India; this is educational.

    pitwall-f1 market template 2026 Abu      # a CSV to type public prices into
    pitwall-f1 market check 2026 Abu --moment quali
    pitwall-f1 paper trade 2026 Abu --moment quali
    pitwall-f1 paper settle                  # after the race
    pitwall-f1 paper report                  # -> reports/market.md
"""
from .compare import (
    BenterFit, MarketScore, benter_walk_forward, fit_benter, market_snapshot, race_table,
    score_vs_market,
)
from .odds import (
    american_from_decimal, decimal_from_american, decimal_from_fractional, devig, edge,
    expected_value, implied_probability, overround, shin_z, to_decimal,
)
from .paper import (
    INITIAL_BANKROLL, PaperStats, TicketExistsError, bankroll, load_settlements,
    load_tickets, make_ticket, paper_stats, risk_of_ruin, save_settlement, save_ticket, settle,
)
from .report import graded_market_rows, write_market_report
from .sources import (
    ManualCsvSource, MarketDataError, PolymarketSource, Roster, load_prices, write_template,
)
from .staking import StakePolicy, growth_rate, kelly_multi, kelly_single, stakes

__all__ = [
    "BenterFit", "MarketScore", "benter_walk_forward", "fit_benter", "market_snapshot",
    "race_table", "score_vs_market", "american_from_decimal", "decimal_from_american",
    "decimal_from_fractional", "devig", "edge", "expected_value", "implied_probability",
    "overround", "shin_z", "to_decimal", "INITIAL_BANKROLL", "PaperStats",
    "TicketExistsError", "bankroll", "load_settlements", "load_tickets", "make_ticket",
    "paper_stats", "risk_of_ruin", "save_settlement", "save_ticket", "settle",
    "graded_market_rows", "write_market_report", "ManualCsvSource", "MarketDataError",
    "PolymarketSource", "Roster", "load_prices", "write_template", "StakePolicy",
    "growth_rate", "kelly_multi", "kelly_single", "stakes",
]
