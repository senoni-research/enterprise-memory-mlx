from __future__ import annotations

import math

import pytest

from enterprise_memory_mlx.gates import (
    DEFAULT_PROMOTION_GATES,
    GateObservation,
    GateSpec,
    clopper_pearson_upper,
    evaluate_gate,
    minimum_units_for_zero_failures,
)


def binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def test_minimum_units_match_design_document() -> None:
    assert minimum_units_for_zero_failures(0.01, 0.95) == 299
    assert minimum_units_for_zero_failures(0.005, 0.95) == 598


def test_zero_failure_bound_closed_form() -> None:
    for units in (10, 50, 299, 1000):
        expected = 1.0 - 0.05 ** (1.0 / units)
        assert clopper_pearson_upper(0, units) == pytest.approx(expected, abs=1e-12)


def test_zero_failure_bound_straddles_one_percent_at_299() -> None:
    assert clopper_pearson_upper(0, 299) <= 0.01
    assert clopper_pearson_upper(0, 298) > 0.01


def test_upper_bound_satisfies_binomial_identity() -> None:
    # Clopper-Pearson upper bound p_u for k failures in n satisfies
    # BinomCDF(k; n, p_u) = 1 - confidence.
    for failures, units in ((1, 100), (2, 250), (5, 500), (10, 1000)):
        upper = clopper_pearson_upper(failures, units, confidence=0.95)
        assert binomial_cdf(failures, units, upper) == pytest.approx(0.05, abs=1e-6)


def test_bound_monotone_in_failures_and_saturates() -> None:
    previous = 0.0
    for failures in range(0, 11):
        upper = clopper_pearson_upper(failures, 100)
        assert upper > previous
        previous = upper
    assert clopper_pearson_upper(100, 100) == 1.0


def test_paraphrases_cluster_into_one_unit() -> None:
    spec = GateSpec(
        name="stale_current_answer",
        denominator_unit="supersession_scenario",
        target_upper_rate=0.01,
        cluster_key="scenario_id",
    )
    # 600 probe rows over 9 facts are 9 independent units, not 600.
    observations = [
        GateObservation(cluster_id=f"fact-{index % 9}", passed=True) for index in range(600)
    ]
    result = evaluate_gate(spec, observations)
    assert result.independent_units == 9
    assert result.status == "insufficient_data"
    assert result.minimum_required_units == 299


def test_gate_passes_with_enough_clean_units() -> None:
    spec = GateSpec(
        name="stale_current_answer",
        denominator_unit="supersession_scenario",
        target_upper_rate=0.01,
    )
    observations = [GateObservation(cluster_id=f"unit-{i}", passed=True) for i in range(299)]
    result = evaluate_gate(spec, observations)
    assert result.status == "pass"


def test_zero_tolerance_fails_on_any_failing_unit() -> None:
    spec = GateSpec(
        name="hallucinated_citation",
        denominator_unit="citation_opportunity",
        target_upper_rate=0.005,
        critical_failure_policy="zero_tolerance",
    )
    observations = [GateObservation(cluster_id=f"unit-{i}", passed=True) for i in range(1000)]
    observations.append(GateObservation(cluster_id="unit-bad", passed=False))
    result = evaluate_gate(spec, observations)
    assert result.status == "fail"
    assert result.failing_clusters == ("unit-bad",)


def test_bounded_gate_uses_exact_bound_when_failures_observed() -> None:
    spec = GateSpec(
        name="old_new_conflation",
        denominator_unit="supersession_scenario",
        target_upper_rate=0.01,
        critical_failure_policy="bounded",
    )
    # One failure in 1000 units: exact upper bound ~0.0047 < 0.01 -> pass.
    observations = [GateObservation(cluster_id=f"unit-{i}", passed=True) for i in range(999)]
    observations.append(GateObservation(cluster_id="unit-bad", passed=False))
    result = evaluate_gate(spec, observations)
    assert result.status == "pass"
    assert result.upper_bound is not None and result.upper_bound < 0.01

    # One failure in 200 units: upper bound ~0.023 > 0.01 -> fail.
    observations = [GateObservation(cluster_id=f"unit-{i}", passed=True) for i in range(199)]
    observations.append(GateObservation(cluster_id="unit-bad", passed=False))
    result = evaluate_gate(spec, observations)
    assert result.status == "fail"


def test_unit_fails_when_any_probe_in_cluster_fails() -> None:
    spec = GateSpec(
        name="new_current_value_strict",
        denominator_unit="supersession_scenario",
        target_upper_rate=0.05,
    )
    observations = [
        GateObservation(cluster_id="scenario-1", passed=True),
        GateObservation(cluster_id="scenario-1", passed=False),
        GateObservation(cluster_id="scenario-2", passed=True),
    ]
    result = evaluate_gate(spec, observations)
    assert result.independent_units == 2
    assert result.failing_units == 1


def test_default_promotion_gates_have_valid_specs() -> None:
    names = {gate.name for gate in DEFAULT_PROMOTION_GATES}
    assert names == {
        "new_current_value_strict",
        "stale_current_answer",
        "old_new_conflation",
        "hallucinated_citation",
    }
    for gate in DEFAULT_PROMOTION_GATES:
        assert gate.minimum_independent_units >= 59  # 5% claim needs >= 59 units
        payload = gate.to_dict()
        assert payload["minimum_independent_units"] == gate.minimum_independent_units
