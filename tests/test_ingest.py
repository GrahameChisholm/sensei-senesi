"""End-to-end test for engine.data.ingest — the on-demand "produce a clean snapshot for the
current gameweek" operation the Phase 1 Definition of Done asks for. All HTTP is mocked.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy.orm import Session

from engine.data import storage
from engine.data.fpl_client import FPLClient
from engine.data.ingest import capture_current_gameweek
from engine.data.snapshots import load_snapshot_tables
from engine.data.understat_client import UnderstatClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# Full-size fixtures (not the lean ones test_fpl_client/test_understat_client use) because
# validation.py's row-count floors are tuned to real data volumes — a 30-row sample would always
# fail validation and never reach "ok", which would defeat the point of this end-to-end test.
BOOTSTRAP = json.loads((FIXTURES_DIR / "fpl_bootstrap_static_full.json").read_text())
FPL_FIXTURES = json.loads((FIXTURES_DIR / "fpl_fixtures.json").read_text())
LEAGUE_DATA = json.loads((FIXTURES_DIR / "understat_league_data_full.json").read_text())
PLAYER_DATA = json.loads((FIXTURES_DIR / "understat_player_data.json").read_text())

SEASON = "2025-26"
GW = 1
CAPTURED_AT = datetime(2026, 8, 1, 9, 0, tzinfo=UTC)


def _element_summary_for(element_id: int) -> dict:
    # A1: one real-shaped history row per player -- enough for validate_fpl_element_summaries'
    # row-count floor (400) across the fixture's 450 elements, and to exercise the
    # live_adapter's column mapping in later tests.
    return {
        "fixtures": [],
        "history": [
            {
                "element": element_id,
                "round": GW,
                "minutes": 90,
                "total_points": 6,
                "opponent_team": 2,
                "was_home": True,
                "kickoff_time": "2025-08-16T14:00:00Z",
                "team_h_score": 2,
                "team_a_score": 1,
            }
        ],
        "history_past": [],
    }


def _fpl_client() -> FPLClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if "bootstrap-static" in request.url.path:
            return httpx.Response(200, json=BOOTSTRAP)
        if "element-summary" in request.url.path:
            element_id = int(request.url.path.rstrip("/").rsplit("/", 1)[-1])
            return httpx.Response(200, json=_element_summary_for(element_id))
        if "fixtures" in request.url.path:
            return httpx.Response(200, json=FPL_FIXTURES)
        raise AssertionError(f"unexpected FPL request: {request.url}")

    return FPLClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _understat_client() -> UnderstatClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if "getLeagueData" in request.url.path:
            return httpx.Response(200, json=LEAGUE_DATA)
        if "getPlayerData" in request.url.path:
            return httpx.Response(200, json=PLAYER_DATA)
        raise AssertionError(f"unexpected Understat request: {request.url}")

    return UnderstatClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_capture_current_gameweek_produces_ok_snapshot_and_records_freshness(tmp_path):
    db_path = str(tmp_path / "fpl.sqlite")
    manifest = capture_current_gameweek(
        fpl_client=_fpl_client(),
        understat_client=_understat_client(),
        season=SEASON,
        understat_season_start_year=2023,
        gameweek=GW,
        base_dir=tmp_path / "snapshots",
        captured_at=CAPTURED_AT,
        db_path=db_path,
    )

    assert manifest.sources["fpl"].status == "ok"
    assert manifest.sources["understat"].status == "ok"
    # A1: the two live per-gameweek history sources engineer_features needs but the snapshot
    # never carried before -- see engine.data.ingest.capture_current_gameweek's new closures.
    assert manifest.sources["fpl_element_summaries"].status == "ok"
    assert manifest.sources["understat_player_histories"].status == "ok"
    assert manifest.all_ok_or_fallback

    fpl_tables = load_snapshot_tables(tmp_path / "snapshots", SEASON, GW, CAPTURED_AT, "fpl")
    assert len(fpl_tables["elements"]) == 450
    assert len(fpl_tables["fixtures"]) == 10

    understat_tables = load_snapshot_tables(
        tmp_path / "snapshots", SEASON, GW, CAPTURED_AT, "understat"
    )
    assert len(understat_tables["players"]) == 350

    element_summaries = load_snapshot_tables(
        tmp_path / "snapshots", SEASON, GW, CAPTURED_AT, "fpl_element_summaries"
    )
    assert len(element_summaries["histories"]) == 450  # one history row per element, per fixture
    assert set(element_summaries["histories"]["element"]) == {
        e["id"] for e in BOOTSTRAP["elements"]
    }

    understat_histories = load_snapshot_tables(
        tmp_path / "snapshots", SEASON, GW, CAPTURED_AT, "understat_player_histories"
    )
    # Every matched player contributes the fixture's 10 matches -- some real, positive number of
    # players must have matched (the whole point of this source) without asserting an exact count
    # that would make this test brittle to crosswalk-heuristic changes.
    assert len(understat_histories["histories"]) > 0
    assert len(understat_histories["histories"]) % 10 == 0
    assert "fpl_id" in understat_histories["histories"].columns

    engine = storage.init_db(db_path)
    with Session(engine) as session:
        fpl_freshness = session.get(storage.DataFreshness, "fpl")
        understat_freshness = session.get(storage.DataFreshness, "understat")
        element_summaries_freshness = session.get(storage.DataFreshness, "fpl_element_summaries")
        assert fpl_freshness.last_attempt_ok is True
        # SQLite has no native tz-aware storage — datetimes round-trip as naive UTC.
        assert fpl_freshness.last_successful_pull_at == CAPTURED_AT.replace(tzinfo=None)
        assert understat_freshness.last_attempt_ok is True
        assert element_summaries_freshness.last_attempt_ok is True


def test_capture_current_gameweek_records_failed_attempt_when_source_has_no_fallback(tmp_path):
    def broken_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    broken_fpl = FPLClient(client=httpx.Client(transport=httpx.MockTransport(broken_handler)))
    db_path = str(tmp_path / "fpl.sqlite")

    manifest = capture_current_gameweek(
        fpl_client=broken_fpl,
        understat_client=_understat_client(),
        season=SEASON,
        understat_season_start_year=2023,
        gameweek=GW,
        base_dir=tmp_path / "snapshots",
        captured_at=CAPTURED_AT,
        db_path=db_path,
        retries=1,
    )

    assert manifest.sources["fpl"].status == "missing"
    assert manifest.sources["understat"].status == "ok"
    # A1: both new sources depend on the FPL client too (element ids, and the crosswalk's
    # fpl_id_by_name) -- a fully-broken FPL client with no prior snapshot to fall back to must
    # degrade the same way, not crash capture_current_gameweek outright.
    assert manifest.sources["fpl_element_summaries"].status == "missing"
    assert manifest.sources["understat_player_histories"].status == "missing"

    engine = storage.init_db(db_path)
    with Session(engine) as session:
        fpl_freshness = session.get(storage.DataFreshness, "fpl")
        assert fpl_freshness.last_attempt_ok is False
