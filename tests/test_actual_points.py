"""Tests for features.actual_points -- Season Replay's real-results scoring, exercising autosubs,
the bench-boost no-autosub rule, captain fallback, and hit-cost deduction."""

from __future__ import annotations

import pytest

from engine.scoring import DEF, FWD, GK, MID
from features.actual_points import score_actual_gameweek
from features.team_state import MyTeamState, SquadPlayer

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]


def _player(player_id: int, position: str, price: int = 40) -> SquadPlayer:
    return SquadPlayer(
        player_id=player_id, position=position, purchase_price=price, current_price=price
    )


def _squad() -> tuple[SquadPlayer, ...]:
    players = [_player(GK1, GK), _player(GK2, GK)]
    players += [_player(pid, DEF) for pid in DEF_IDS]
    players += [_player(pid, MID) for pid in MID_IDS]
    players += [_player(pid, FWD) for pid in FWD_IDS]
    return tuple(players)


def _starting_xi() -> tuple[int, ...]:
    return (GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2])


def _bench_order() -> tuple[int, ...]:
    return (DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2)


def _team_state(captain_id: int = 21, vice_captain_id: int = 22) -> MyTeamState:
    return MyTeamState(
        squad=_squad(),
        starting_xi=_starting_xi(),
        bench_order=_bench_order(),
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        bank=0,
        free_transfers=1,
        chips_remaining=frozenset(),
    )


def _all_played_90(points_per_player: float = 2.0) -> tuple[dict, dict]:
    all_ids = [GK1, GK2, *DEF_IDS, *MID_IDS, *FWD_IDS]
    minutes = dict.fromkeys(all_ids, 90)
    points = dict.fromkeys(all_ids, points_per_player)
    return minutes, points


class TestNoChipScoring:
    def test_sums_effective_xi_plus_captain_bonus(self):
        minutes, points = _all_played_90(2.0)
        result = score_actual_gameweek(
            gameweek=5,
            team_state=_team_state(captain_id=21),
            chip=None,
            hit_cost=0,
            minutes_by_player=minutes,
            points_by_player=points,
        )
        # 11 starters at 2.0 each + captain (21) doubled once more (+2.0)
        assert result.points == pytest.approx(11 * 2.0 + 2.0)
        assert result.effective_captain_id == 21
        assert len(result.effective_xi) == 11

    def test_bench_does_not_count_without_bench_boost(self):
        minutes, points = _all_played_90(2.0)
        points[DEF_IDS[4]] = 999.0  # bench player -- must not leak into the total
        result = score_actual_gameweek(
            gameweek=5,
            team_state=_team_state(),
            chip=None,
            hit_cost=0,
            minutes_by_player=minutes,
            points_by_player=points,
        )
        assert result.points < 100


class TestAutosubs:
    def test_zero_minute_starter_is_replaced_from_bench(self):
        minutes, points = _all_played_90(2.0)
        blanked_starter = FWD_IDS[1]
        minutes[blanked_starter] = 0
        result = score_actual_gameweek(
            gameweek=5,
            team_state=_team_state(),
            chip=None,
            hit_cost=0,
            minutes_by_player=minutes,
            points_by_player=points,
        )
        assert blanked_starter not in result.effective_xi
        # apply_autosubs is position-blind among outfielders -- the first bench_order entry
        # (DEF_IDS[4], per this fixture's _bench_order()) comes on, not necessarily a FWD reserve.
        assert DEF_IDS[4] in result.effective_xi
        assert len(result.effective_xi) == 11


class TestBenchBoost:
    def test_all_fifteen_count_with_no_autosubs(self):
        minutes, points = _all_played_90(2.0)
        result = score_actual_gameweek(
            gameweek=5,
            team_state=_team_state(),
            chip="bench_boost",
            hit_cost=0,
            minutes_by_player=minutes,
            points_by_player=points,
        )
        assert len(result.effective_xi) == 15
        assert result.points == pytest.approx(15 * 2.0 + 2.0)  # + captain bonus

    def test_blanked_starter_is_not_autosubbed_and_scores_zero(self):
        minutes, points = _all_played_90(2.0)
        blanked_starter = FWD_IDS[1]
        minutes[blanked_starter] = 0
        points[blanked_starter] = 0.0
        result = score_actual_gameweek(
            gameweek=5,
            team_state=_team_state(),
            chip="bench_boost",
            hit_cost=0,
            minutes_by_player=minutes,
            points_by_player=points,
        )
        # All 15 still counted (no autosub under BB) -- 14 scorers at 2.0 + captain bonus + 0
        assert result.points == pytest.approx(14 * 2.0 + 2.0)


class TestTripleCaptain:
    def test_captain_gets_tripled_not_doubled(self):
        minutes, points = _all_played_90(2.0)
        result = score_actual_gameweek(
            gameweek=5,
            team_state=_team_state(captain_id=21),
            chip="triple_captain",
            hit_cost=0,
            minutes_by_player=minutes,
            points_by_player=points,
        )
        assert result.points == pytest.approx(11 * 2.0 + 2 * 2.0)


class TestCaptainFallback:
    def test_vice_gets_multiplier_when_captain_blanks(self):
        minutes, points = _all_played_90(2.0)
        minutes[21] = 0  # captain blanks
        result = score_actual_gameweek(
            gameweek=5,
            team_state=_team_state(captain_id=21, vice_captain_id=22),
            chip=None,
            hit_cost=0,
            minutes_by_player=minutes,
            points_by_player=points,
        )
        assert result.effective_captain_id == 22


class TestHitCost:
    def test_hit_cost_is_deducted(self):
        minutes, points = _all_played_90(2.0)
        result = score_actual_gameweek(
            gameweek=5,
            team_state=_team_state(),
            chip=None,
            hit_cost=4,
            minutes_by_player=minutes,
            points_by_player=points,
        )
        assert result.points == pytest.approx(11 * 2.0 + 2.0 - 4)
        assert result.hit_cost == 4

    def test_negative_hit_cost_raises(self):
        minutes, points = _all_played_90(2.0)
        with pytest.raises(ValueError, match="non-negative"):
            score_actual_gameweek(
                gameweek=5,
                team_state=_team_state(),
                chip=None,
                hit_cost=-4,
                minutes_by_player=minutes,
                points_by_player=points,
            )


class TestUnknownChip:
    def test_raises(self):
        minutes, points = _all_played_90(2.0)
        with pytest.raises(ValueError, match="unknown chip"):
            score_actual_gameweek(
                gameweek=5,
                team_state=_team_state(),
                chip="not_a_chip",
                hit_cost=0,
                minutes_by_player=minutes,
                points_by_player=points,
            )


class TestMissingResults:
    def test_player_with_no_result_is_treated_as_a_blank_and_autosubbed(self):
        # A missing entry defaults to 0 minutes (Mapping.get(id, 0)), which is exactly the signal
        # apply_autosubs already acts on -- so a genuinely missing result is treated the same as a
        # real 0-minute blank (subbed out), not silently scored as a real appearance.
        minutes, points = _all_played_90(2.0)
        del minutes[FWD_IDS[0]]
        del points[FWD_IDS[0]]
        result = score_actual_gameweek(
            gameweek=5,
            team_state=_team_state(),
            chip=None,
            hit_cost=0,
            minutes_by_player=minutes,
            points_by_player=points,
        )
        assert FWD_IDS[0] not in result.effective_xi
        assert result.points == pytest.approx(11 * 2.0 + 2.0)

    def test_bench_boost_scores_a_missing_result_as_zero_with_no_substitution(self):
        # Under Bench Boost there is no autosub concept at all -- a missing result just contributes
        # nothing, and the player stays exactly where they were picked.
        minutes, points = _all_played_90(2.0)
        del minutes[FWD_IDS[0]]
        del points[FWD_IDS[0]]
        result = score_actual_gameweek(
            gameweek=5,
            team_state=_team_state(),
            chip="bench_boost",
            hit_cost=0,
            minutes_by_player=minutes,
            points_by_player=points,
        )
        assert len(result.effective_xi) == 15
        assert result.points == pytest.approx(14 * 2.0 + 2.0)
