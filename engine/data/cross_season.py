"""Cross-season player/team history — the real, unresolved gap ``engine/data/live_adapter.py``'s
own docstring names: at a season's true opening gameweek, *no player* has any this-season
minutes/starts history at all yet, so ``training_history`` for GW1 is empty regardless of how
good the live team-rate pooling is, and ``backtest.run_season.fit_fn`` crashes fitting sklearn
models on zero samples.

This module closes that gap by re-keying **last season's** real, already-cached vaastav
``merged_gw`` rows onto **this season's** player/team ids (via FPL's own season-stable ``code``
field) and shifting their gameweek numbers negative, so they sort strictly before this season's
target rows without needing any change to ``backtest.run_season.engineer_features`` itself — every
per-player feature there is grouped by ``player_id`` and sorted by real ``kickoff_time`` (not
gameweek number), and every team-rate join keys off ``(team, gameweek)`` pairs derived from the
same frame, so a negative offset is internally consistent regardless of the numbering convention
(verified by reading ``backtest/run_season.py`` directly, not assumed).

**Two hazards this creates, both handled explicitly, never left implicit:**

1. **Opponent-id collisions across seasons.** ``engineer_features`` maps ``opponent_team`` through
   the *live* ``teams`` table. A prior-season row's ``opponent_team`` is a *prior-season* id — team
   ids get reassigned across promotion/relegation, so mapping it through the live table would
   silently mislabel a fixture. :func:`prior_season_merged_gw` remaps every ``opponent_team``
   through ``team_id_map``, falling back to a synthetic id (:func:`synthetic_team_rows`) for a
   relegated club absent from this season's live teams altogether.
2. **A player who left the league entirely** (retired, dropped out of the top four divisions) has
   no ``code`` match in the live elements table — dropped from the training frame, not defaulted,
   matching this repo's existing "never default a missing rate to zero" discipline
   (ENGINE_IMPROVEMENTS_2.md C.1/C.2).

**A known, narrower gap not solved here.** A club's own name is stable across seasons for any club
that survives (Understat/FPL never rename a surviving club), so :func:`prior_season_merged_gw`
deliberately does *not* rewrite a row's ``team`` field — ``engineer_features`` looks up team-level
rates by that same name. But a player who played for a club **relegated** at the end of last season
(even if the player themselves transferred to a surviving club) still has their own prior-season
rows' ``team`` set to that now-relegated club's name — which typically has no entry in this
season's pooled ``team_histories`` (it isn't tracked once relegated), so those specific rows'
``team_xg_per_90``/``team_xga_per_90`` still resolve to ``NaN`` and get dropped. This narrows —
rather than eliminates — the training signal for players who transferred in from a relegated club;
fixing it would mean also pooling relegated clubs' own historical team rates, not attempted here.

**A second, separate cold-start gap this module also closes.** Re-keying ``merged_gw`` alone is
not sufficient: ``engineer_features``'s ``npxg_per_90``/``xa_per_90`` columns come from
``player_histories`` (Understat per-match rows), a wholly separate input from ``merged_gw``, built
by ``engine.data.live_adapter.build_player_histories_from_live_snapshot`` from the live snapshot's
own ``understat_player_histories`` source alone — with **no prior-season pooling at all**, unlike
the existing team-rate fix. At a real season's opening gameweek this is empty for every player (the
same pre-season Understat gap the team-rate fix already works around), so without also fixing
this, every row's ``npxg_per_90``/``xa_per_90`` — including the prior-season *training* rows the
first fix supplies — would still be ``NaN``, and ``engineer_features``' dropna would still empty
the training frame. :func:`remap_player_histories`/:func:`merge_player_histories` close this by
re-keying and stitching in the output of ``backtest.run_season.fetch_understat_player_histories``
(which — per that function's own docstring, ENGINE_IMPROVEMENTS_2.md C.3 — already spans several
prior seasons per player once given last season's own crosswalk) onto this season's element ids.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from engine.data.live_adapter import MERGED_GW_COLUMNS

__all__ = [
    "PRIOR_SEASON_GAMEWEEKS",
    "fetch_vaastav_players_raw",
    "player_code_map",
    "team_id_map",
    "synthetic_team_rows",
    "prior_season_merged_gw",
    "remap_player_histories",
    "merge_player_histories",
]

# Gameweek offset applied to every prior-season row so it sorts strictly before this season's
# target rows: last season's GW38 becomes 0, GW1 becomes -37, etc. Any value >= the prior season's
# own gameweek count works (real ordering comes from `kickoff_time`, not this number — see this
# module's own docstring); 38 is simply the real length of a normal season, matching
# `simulator/chip_calendar.py`'s own `FIRST_HALF_LAST_GAMEWEEK`-adjacent convention of using real
# FPL calendar facts rather than an arbitrary round number.
PRIOR_SEASON_GAMEWEEKS = 38

_VAASTAV_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

_PLAYERS_RAW_COLUMNS = [
    "id",
    "code",
    "web_name",
    "first_name",
    "second_name",
    "team",
    "element_type",
]


def _season_label(season_start_year: int) -> str:
    return f"{season_start_year}-{str(season_start_year + 1)[-2:]}"


def fetch_vaastav_players_raw(
    season_start_year: int, cache_dir: Path, client: httpx.Client, refresh: bool = False
) -> pd.DataFrame:
    """One row per element vaastav's archive recorded for ``season_start_year``, carrying FPL's
    own season-stable ``code`` field alongside that season's element ``id`` — the join key
    :func:`player_code_map` needs. Cached as parquet under ``cache_dir/vaastav/{label}/``,
    matching ``backtest.run_season.fetch_vaastav_teams``'s own caching convention exactly."""
    label = _season_label(season_start_year)
    cache_path = cache_dir / "vaastav" / label / "players_raw.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)[_PLAYERS_RAW_COLUMNS]
    response = client.get(f"{_VAASTAV_RAW_BASE}/{label}/players_raw.csv")
    response.raise_for_status()
    import io

    df = pd.read_csv(io.StringIO(response.text))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df[_PLAYERS_RAW_COLUMNS]


def player_code_map(prior_players_raw: pd.DataFrame, live_elements: pd.DataFrame) -> dict[int, int]:
    """Prior-season element id -> this-season element id, joined on FPL's season-stable ``code``
    field. Raises on a duplicate ``code`` within either season's own table — a wrong merge there
    would silently attribute one player's entire history to another, the same failure mode
    ``engine/data/crosswalk.py`` refuses to risk for its own (Understat) cross-referencing."""
    for label, frame in (
        ("prior_players_raw", prior_players_raw),
        ("live_elements", live_elements),
    ):
        duplicated = frame["code"][frame["code"].duplicated()]
        if not duplicated.empty:
            raise ValueError(
                f"duplicate code(s) in {label}: {sorted(duplicated.unique().tolist())}"
            )

    prior_id_by_code = dict(zip(prior_players_raw["code"], prior_players_raw["id"], strict=True))
    live_id_by_code = dict(zip(live_elements["code"], live_elements["id"], strict=True))
    return {
        prior_id: live_id_by_code[code]
        for code, prior_id in prior_id_by_code.items()
        if code in live_id_by_code
    }


def team_id_map(prior_teams: pd.DataFrame, live_teams: pd.DataFrame) -> dict[int, int]:
    """Prior-season team id -> this-season team id, joined on club ``code`` (stable across
    promotion/relegation, unlike the plain numeric ``id`` vaastav/bootstrap otherwise assign fresh
    each season). A relegated club — absent from ``live_teams`` — is simply absent from the
    result; :func:`synthetic_team_rows` is what gives it a resolvable identity in the current
    season's frame instead."""
    live_id_by_code = dict(zip(live_teams["code"], live_teams["id"], strict=True))
    return {
        int(row.id): live_id_by_code[row.code]
        for row in prior_teams.itertuples()
        if row.code in live_id_by_code
    }


def synthetic_team_rows(prior_teams: pd.DataFrame, live_teams: pd.DataFrame) -> pd.DataFrame:
    """One ``teams``-shaped row per club present in ``prior_teams`` but absent from ``live_teams``
    (a relegated club) — allocated a fresh id above every live id, so
    ``engineer_features``'s ``opponent_team -> name`` mapping resolves a remapped prior-season
    fixture's opponent to a real name instead of colliding with (and mislabelling as) whichever
    live club happens to have reused that same numeric id this season.

    Only the columns ``engineer_features``/its own team-name lookup actually reads are populated
    (``id``, ``name``, ``short_name``, ``code``) — every other column real ``teams`` frames carry
    (strength ratings, etc.) is irrelevant for a club that isn't actually playing this season and
    is left absent."""
    live_codes = set(live_teams["code"])
    relegated = prior_teams[~prior_teams["code"].isin(live_codes)]
    if relegated.empty:
        return pd.DataFrame(columns=["id", "name", "short_name", "code"])
    next_id = int(live_teams["id"].max()) + 1
    rows = []
    for offset, row in enumerate(relegated.itertuples()):
        rows.append(
            {
                "id": next_id + offset,
                "name": row.name,
                "short_name": row.short_name,
                "code": row.code,
            }
        )
    return pd.DataFrame(rows)


def prior_season_merged_gw(
    prior_merged_gw: pd.DataFrame,
    code_map: dict[int, int],
    prior_team_id_to_current: dict[int, int],
    relegated_team_ids: dict[int, int],
    gameweek_offset: int = PRIOR_SEASON_GAMEWEEKS,
) -> pd.DataFrame:
    """Re-key one prior season's real ``merged_gw`` rows (vaastav's ``element``/``round`` naming,
    same shape ``fetch_vaastav_merged_gw`` returns) onto this season's player ids and a negative
    gameweek range, in exactly :data:`~engine.data.live_adapter.MERGED_GW_COLUMNS`' column order —
    the contract ``backtest.run_season.engineer_features`` (via
    ``engine.data.live_adapter.build_merged_gw``) expects.

    ``relegated_team_ids`` maps a prior-season team **id** (not name/code) present in
    :func:`synthetic_team_rows`'s input to the synthetic id it was given — used only to remap
    ``opponent_team`` for a fixture against a since-relegated club; every other ``opponent_team``
    goes through ``prior_team_id_to_current`` instead. ``position``/``team`` are kept at their
    **prior-season** values deliberately (see this module's own docstring) — they are historical
    facts every lagged per-player/per-team feature is computed from, not corrected to this
    season's team/position, which the live target row already carries in full.

    Players who left the league entirely (no entry in ``code_map``) are dropped, not defaulted —
    matching ENGINE_IMPROVEMENTS_2.md C.1/C.2's "never default a missing rate to zero" rule.

    Every real vaastav ``merged_gw`` export (verified against this repo's own cached 2024/25 and
    2025/26 parquets) carries **both** ``round`` and ``GW`` already, always identical — renaming
    ``round -> GW`` unconditionally would silently produce two same-named ``GW`` columns (pandas
    allows this, and every later ``df["GW"]`` access then returns a 2-column frame instead of a
    Series, corrupting the gameweek-offset arithmetic below and the final column selection alike).
    ``round`` is only ever renamed when ``GW`` isn't already present.
    """
    df = prior_merged_gw.rename(columns={"element": "player_id"}).copy()
    if "GW" not in df.columns and "round" in df.columns:
        df = df.rename(columns={"round": "GW"})
    elif "round" in df.columns:
        df = df.drop(columns=["round"])

    df["player_id"] = df["player_id"].map(code_map)
    df = df[df["player_id"].notna()].copy()
    df["player_id"] = df["player_id"].astype(int)

    df["GW"] = df["GW"].astype(int) - gameweek_offset

    combined_team_map = {**prior_team_id_to_current, **relegated_team_ids}
    df["opponent_team"] = df["opponent_team"].astype(int).map(combined_team_map)
    unresolved = df["opponent_team"].isna()
    if unresolved.any():
        unresolved_ids = prior_merged_gw.loc[df.index[unresolved], "opponent_team"].unique()
        raise ValueError(
            "prior_season_merged_gw: opponent_team id(s) with no current-season or synthetic "
            f"mapping: {sorted(unresolved_ids)}"
        )
    df["opponent_team"] = df["opponent_team"].astype(int)

    for column in ("value", "selected", "transfers_out", "transfers_balance"):
        if column not in df.columns:
            df[column] = 0

    # T-A: a prior-season row is always real played history, never a synthesized target row, so
    # this defaults to False -- matching MERGED_GW_COLUMNS' contract (see
    # engine.data.live_adapter) without needing the real vaastav export to carry the column
    # itself.
    if "is_synthesized_target" not in df.columns:
        df["is_synthesized_target"] = False

    # T-K: penalties_order is a live-only squad-role snapshot with no retained per-gameweek
    # history (see engine.data.live_adapter.MERGED_GW_COLUMNS' own comment). A prior-season row
    # never carries one, so this defaults to NaN for the same column-contract reason
    # is_synthesized_target defaults to False above.
    if "penalties_order" not in df.columns:
        df["penalties_order"] = np.nan

    missing = [c for c in MERGED_GW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"prior_merged_gw is missing expected column(s): {missing} — check the cached "
            "vaastav merged_gw parquet's shape against MERGED_GW_COLUMNS"
        )
    return df[MERGED_GW_COLUMNS].reset_index(drop=True)


def remap_player_histories(
    prior_player_histories: dict[int, pd.DataFrame], code_map: dict[int, int]
) -> dict[int, pd.DataFrame]:
    """Re-key a dict of Understat player histories from prior-season FPL element ids (the shape
    ``backtest.run_season.fetch_understat_player_histories`` returns, itself already spanning
    several seasons back per ENGINE_IMPROVEMENTS_2.md C.3) onto this season's element ids, via the
    same ``code_map`` :func:`player_code_map` builds. A player absent from ``code_map`` (left the
    league) is dropped, matching :func:`prior_season_merged_gw`'s own discipline."""
    result: dict[int, pd.DataFrame] = {}
    for prior_id, history in prior_player_histories.items():
        current_id = code_map.get(prior_id)
        if current_id is None:
            continue
        result[current_id] = history
    return result


def merge_player_histories(
    prior: dict[int, pd.DataFrame], current: dict[int, pd.DataFrame]
) -> dict[int, pd.DataFrame]:
    """Concatenate an (already current-season-keyed, via :func:`remap_player_histories`)
    prior-season history dict with this season's own live-tracked history dict, per player,
    re-sorted chronologically — exactly the single "spanning prior seasons into the current one"
    history ``engine/rates.py``'s own module docstring says every caller should pass. Either side
    may be missing a given ``player_id`` (no prior-season history, or no matches yet this season)
    without special-casing — a plain per-key union with an empty-frame default on whichever side
    lacks it."""
    result: dict[int, pd.DataFrame] = {}
    for player_id in set(prior) | set(current):
        frames = [
            df
            for df in (prior.get(player_id), current.get(player_id))
            if df is not None and not df.empty
        ]
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        result[player_id] = combined.sort_values("date").reset_index(drop=True)
    return result
