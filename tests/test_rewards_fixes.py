"""
SPEC-7 reward fix verification tests.
All 5 tests must pass after implementing fixes 1–6.
"""
import types
import pytest

from firewatch_env.rewards import grade, EpisodeResult


def _er(affected, recovered, ticks, wrong, slo, bcm, static=None, total=None):
    """Helper: build an EpisodeResult with all required fields set."""
    e = EpisodeResult()
    e.services_affected = affected
    e.services_affected_static = static if static is not None else affected
    e.total_services_in_episode = total if total is not None else affected
    e.services_recovered = recovered
    e.ticks_taken = ticks
    e.wrong_actions = wrong
    e.final_slo_budget = slo
    e.bad_customer_minutes = bcm
    e.total_actions = max(ticks, 1)
    return e


# ── Fix 1: Tick-0 exploit ─────────────────────────────────────────────────

def test_tick0_exploit_blocked():
    """declare_resolved before tick 2 must never return a high score."""
    er = EpisodeResult()
    er.services_affected = 0
    er.services_affected_static = 3
    er.total_services_in_episode = 3
    er.services_recovered = 0
    er.ticks_taken = 0
    er.wrong_actions = 0
    er.final_slo_budget = 100.0
    er.bad_customer_minutes = 0.0
    er.total_actions = 1
    s = grade(er, "easy")
    assert s <= 0.10, f"tick-0 exploit gives {s:.3f}, expected <= 0.10"


# ── Fix 2: No reward cliff ────────────────────────────────────────────────

def test_partial_recovery_early_exit_no_cliff():
    """Partial recovery + early declare_resolved must not wipe BCM and SLO to zero."""
    er = _er(affected=3, recovered=2, ticks=6, wrong=0,
             slo=70.0, bcm=15.0, static=3, total=3)
    s = grade(er, "easy")
    assert s > 0.20, f"partial+early gives {s:.3f}, expected > 0.20 (no cliff)"


# ── Fix 3: Blast radius equivalency ──────────────────────────────────────

def test_blast_radius_fast_agent_scores_higher():
    """Agent that stopped cascade at 1 service scores higher than one who let it reach 3."""
    # Agent A: stopped cascade at 1 service
    a = _er(affected=1, recovered=1, ticks=5, wrong=0,
            slo=90.0, bcm=5.0, static=3, total=3)
    # Agent B: let cascade reach all 3 services then fixed all
    b = _er(affected=3, recovered=3, ticks=5, wrong=0,
            slo=90.0, bcm=5.0, static=3, total=3)
    sa = grade(a, "easy")
    sb = grade(b, "easy")
    assert sa > sb, f"fast agent {sa:.3f} should beat slow agent {sb:.3f}"


# ── Fix 6: Weighted mean error rate ──────────────────────────────────────

def test_weighted_mean_error_rate_weights_by_dependents():
    """api-gateway with 3 dependents dominates over a leaf service."""
    from firewatch_env.rewards import _weighted_mean_error_rate  # added in Task 4
    def _svc(er):
        return types.SimpleNamespace(http_server_error_rate=er)

    # 5 services; svc-a/b/c each depend on api-gateway
    services = {
        "api-gateway": _svc(0.8),
        "cache":       _svc(0.1),
        "svc-a":       _svc(0.0),
        "svc-b":       _svc(0.0),
        "svc-c":       _svc(0.0),
    }
    dep_graph = {
        "svc-a":       ["api-gateway"],
        "svc-b":       ["api-gateway"],
        "svc-c":       ["api-gateway"],
        "api-gateway": [],
        "cache":       [],
    }
    weighted = _weighted_mean_error_rate(services, dep_graph)
    unweighted = (0.8 + 0.1 + 0.0 + 0.0 + 0.0) / 5  # 0.18
    assert weighted > unweighted, (
        f"weighted {weighted:.3f} should > unweighted {unweighted:.3f} "
        f"(api-gateway has 3 dependents, dominates)"
    )


# ── Variance check ────────────────────────────────────────────────────────

def test_variance_check():
    """Overall score distribution must satisfy these four invariants."""
    perfect = grade(_er(3, 3,  5, 0, 90.0,  5.0, 3, 3), "easy")
    zero    = grade(_er(3, 0,  1, 0, 98.0,  0.0, 3, 3), "easy")
    wrong   = grade(_er(3, 2, 20, 4, 40.0, 50.0, 3, 3), "easy")

    assert perfect > 0.80, f"perfect={perfect:.3f}, expected > 0.80"
    assert zero    < 0.10, f"zero={zero:.3f}, expected < 0.10"
    assert 0.10 <= wrong <= 0.60, f"wrong={wrong:.3f}, expected in [0.10, 0.60]"
    assert perfect - zero >= 0.50, f"gap={perfect - zero:.3f}, expected >= 0.50"


# ── Fix 4: MTTM requires 3 consecutive zero-BCM ticks ─────────────────────

def test_mttm_requires_3_consecutive_zero_bcm_ticks():
    """MTTM must not be granted until 3 consecutive ticks with bcm_delta == 0."""
    from firewatch_env.simulation import IncidentMetrics
    m = IncidentMetrics()
    m.update(bcm_delta=1.0, current_tick=1)   # BCM still moving
    m.update(bcm_delta=0.0, current_tick=2)   # streak=1
    m.update(bcm_delta=0.0, current_tick=3)   # streak=2
    assert m.mttm_achieved_tick is None, "must not grant MTTM after only 2 consecutive zeros"
    m.update(bcm_delta=0.0, current_tick=4)   # streak=3 → granted at tick 4-2=2
    assert m.mttm_achieved_tick == 2, f"expected mttm_achieved_tick=2, got {m.mttm_achieved_tick}"


def test_mttm_streak_resets_on_nonzero():
    """A non-zero BCM tick must reset the streak — MTTM only after 3 unbroken zeros."""
    from firewatch_env.simulation import IncidentMetrics
    m = IncidentMetrics()
    m.update(bcm_delta=0.0, current_tick=1)   # streak=1
    m.update(bcm_delta=0.0, current_tick=2)   # streak=2
    m.update(bcm_delta=1.0, current_tick=3)   # non-zero resets streak
    m.update(bcm_delta=0.0, current_tick=4)   # streak=1 again
    m.update(bcm_delta=0.0, current_tick=5)   # streak=2
    assert m.mttm_achieved_tick is None, "streak was reset; MTTM must not be granted yet"
    m.update(bcm_delta=0.0, current_tick=6)   # streak=3 → granted at tick 6-2=4
    assert m.mttm_achieved_tick == 4, f"expected mttm_achieved_tick=4, got {m.mttm_achieved_tick}"
