"""Pre-match odds client — pulled once per gameweek, shortly before the deadline (BUILD_PLAN
4b.1).

Built on The Odds API (the-odds-api.com) — a single, well-documented, free-tier-available
provider covering match-result/totals odds broadly and player-prop markets (anytime goalscorer)
where the plan tier supports them. One snapshot a week keeps this comfortably within any
free-tier request budget (BUILD_PLAN 4b.1). Kept a thin, swappable wrapper — everything
downstream (``divergence.py``) depends only on this module's plain parsed dataclasses, never on
the provider's raw JSON shape, so switching providers later doesn't ripple outward.

Deliberately outside ``engine/`` and never called by any backtest (BUILD_PLAN's Phase 4b
framing) — this client is pulled live, at decision time, once per gameweek, never during
backtesting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_TIMEOUT = 15.0
DEFAULT_SPORT_KEY = "soccer_epl"
DEFAULT_REGION = "uk"

# Match-result / totals markets, pulled via the bulk `/sports/{sport}/odds` endpoint.
MATCH_MARKETS = ("h2h", "totals")
# Anytime-goalscorer odds are a per-event market, pulled via `/sports/{sport}/events/{id}/odds` --
# not every plan tier includes player props, so callers should treat an empty result as "not
# available on this plan," not an error (see `get_anytime_scorer_odds`).
ANYTIME_SCORER_MARKET = "player_goal_scorer_anytime"


class OddsClientError(RuntimeError):
    """Raised when the odds provider returns something the client can't use."""


def _api_key_from_env() -> str:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise OddsClientError(
            "ODDS_API_KEY is not set -- copy .env.example to .env and fill in a real key "
            "(BUILD_PLAN 4b.1); never hardcode it in source"
        )
    return api_key


@dataclass
class OddsClient:
    """Thin wrapper over The Odds API. A plain ``httpx.Client`` is injected so tests can swap in
    an ``httpx.MockTransport`` instead of hitting the network — same convention as
    ``engine.data.fpl_client.FPLClient``."""

    api_key: str = field(default_factory=_api_key_from_env)
    base_url: str = BASE_URL
    sport_key: str = DEFAULT_SPORT_KEY
    region: str = DEFAULT_REGION
    client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=DEFAULT_TIMEOUT))

    def __enter__(self) -> OddsClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        try:
            response = self.client.get(
                f"{self.base_url}{path}", params={**params, "apiKey": self.api_key}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OddsClientError(f"odds API request failed for {path}: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise OddsClientError(f"odds API returned non-JSON body for {path}") from exc

    def get_match_odds(self) -> list[dict[str, Any]]:
        """One gameweek's worth of raw match-result (h2h) and total-goals (totals) odds, one
        entry per fixture — pulled once per gameweek shortly before the deadline (BUILD_PLAN
        4b.1). Feed the result into :func:`parse_match_result_odds`."""
        return self._get(
            f"/sports/{self.sport_key}/odds",
            {"regions": self.region, "markets": ",".join(MATCH_MARKETS), "oddsFormat": "decimal"},
        )

    def get_anytime_scorer_odds(self, event_id: str) -> dict[str, Any]:
        """Raw anytime-goalscorer odds for one fixture. Feed the result into
        :func:`parse_anytime_scorer_odds`. Player-prop markets are a paid-tier feature on most
        odds providers, so an empty ``bookmakers`` list here means "not available on this plan,"
        not an error."""
        return self._get(
            f"/sports/{self.sport_key}/events/{event_id}/odds",
            {"regions": self.region, "markets": ANYTIME_SCORER_MARKET, "oddsFormat": "decimal"},
        )


@dataclass(frozen=True)
class MatchResultOdds:
    """One fixture's consensus decimal odds for the three match-result outcomes — the mean
    decimal price across every bookmaker the provider returned, a simple, transparent way to
    collapse multiple quotes into one number (BUILD_PLAN 4b.2 removes the margin from this later,
    in ``divergence.py``; this module only parses and averages, never adjusts)."""

    fixture_id: str
    home_team: str
    away_team: str
    home_odds: float
    draw_odds: float
    away_odds: float


@dataclass(frozen=True)
class AnytimeScorerOdds:
    """One player's consensus decimal odds to score anytime in one fixture."""

    fixture_id: str
    player_name: str
    odds: float


def _mean_outcome_price(
    bookmakers: list[dict[str, Any]], market_key: str, outcome_name: str
) -> float | None:
    prices = [
        outcome["price"]
        for bookmaker in bookmakers
        for market in bookmaker.get("markets", [])
        if market.get("key") == market_key
        for outcome in market.get("outcomes", [])
        if outcome.get("name") == outcome_name
    ]
    return sum(prices) / len(prices) if prices else None


def parse_match_result_odds(raw_events: list[dict[str, Any]]) -> list[MatchResultOdds]:
    """Parse :meth:`OddsClient.get_match_odds`'s raw response into one :class:`MatchResultOdds`
    per fixture that actually carries an ``h2h`` market. A fixture with no bookmaker offering
    that market yet (too far out, or the provider hasn't populated it) is silently skipped rather
    than raising — a live overlay pulling odds for an entire gameweek at once should degrade
    gracefully per-fixture, not fail the whole pull over one not-yet-priced match."""
    results = []
    for event in raw_events:
        bookmakers = event.get("bookmakers", [])
        home_odds = _mean_outcome_price(bookmakers, "h2h", event["home_team"])
        away_odds = _mean_outcome_price(bookmakers, "h2h", event["away_team"])
        draw_odds = _mean_outcome_price(bookmakers, "h2h", "Draw")
        if home_odds is None or away_odds is None or draw_odds is None:
            continue
        results.append(
            MatchResultOdds(
                fixture_id=event["id"],
                home_team=event["home_team"],
                away_team=event["away_team"],
                home_odds=home_odds,
                draw_odds=draw_odds,
                away_odds=away_odds,
            )
        )
    return results


def parse_anytime_scorer_odds(raw_event: dict[str, Any]) -> list[AnytimeScorerOdds]:
    """Parse :meth:`OddsClient.get_anytime_scorer_odds`'s raw response into one
    :class:`AnytimeScorerOdds` per named outcome, averaged across bookmakers the same way as
    :func:`parse_match_result_odds`. Returns an empty list (not an error) when the plan tier
    doesn't carry this market at all."""
    bookmakers = raw_event.get("bookmakers", [])
    player_names = {
        outcome["name"]
        for bookmaker in bookmakers
        for market in bookmaker.get("markets", [])
        if market.get("key") == ANYTIME_SCORER_MARKET
        for outcome in market.get("outcomes", [])
    }
    results = []
    for player_name in sorted(player_names):
        price = _mean_outcome_price(bookmakers, ANYTIME_SCORER_MARKET, player_name)
        if price is not None:
            results.append(
                AnytimeScorerOdds(fixture_id=raw_event["id"], player_name=player_name, odds=price)
            )
    return results
