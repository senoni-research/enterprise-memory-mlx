"""Release-gate schema with exact one-sided confidence bounds.

Every promotion gate is a claim of the form "the failure rate of <unit> is at
most <target_upper_rate>". A gate can only pass when the one-sided
Clopper-Pearson upper confidence bound on the observed failure rate is at or
below the target. Repeated probes that share a cluster key (for example many
paraphrases of one fact) are collapsed into a single independent unit, and the
unit fails if any of its probes fail.

Delta-style gates (for example "unrelated-fact regression <= 2 points") are
paired comparisons against a reference run and are outside this schema; they
belong to the evaluation harness.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

CriticalFailurePolicy = Literal["zero_tolerance", "bounded"]
GateStatus = Literal["pass", "fail", "insufficient_data"]

_ONE_SIDED_INTERVAL = "clopper_pearson_one_sided"


def minimum_units_for_zero_failures(target_upper_rate: float, confidence: float = 0.95) -> int:
    """Smallest number of independent units for which zero observed failures
    yield a one-sided upper bound at or below ``target_upper_rate``.

    Closed form: n >= log(1 - confidence) / log(1 - p).
    Examples at 95% confidence: p=1% -> 299 units, p=0.5% -> 598 units.
    """
    if not 0.0 < target_upper_rate < 1.0:
        raise ValueError("target_upper_rate must be in (0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return math.ceil(math.log(1.0 - confidence) / math.log(1.0 - target_upper_rate))


def clopper_pearson_upper(failures: int, units: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper confidence bound for a binomial failure rate.

    Equals the ``confidence`` quantile of Beta(failures + 1, units - failures).
    Used instead of the rule-of-three approximation whenever failures > 0.
    """
    if units <= 0:
        raise ValueError("units must be positive")
    if not 0 <= failures <= units:
        raise ValueError("failures must be between 0 and units")
    if failures == units:
        return 1.0
    if failures == 0:
        # Closed form: (1 - p)^n = 1 - confidence.
        return 1.0 - (1.0 - confidence) ** (1.0 / units)
    return _beta_ppf(confidence, failures + 1.0, float(units - failures))


@dataclass(frozen=True)
class GateSpec:
    """Declarative definition of one promotion gate."""

    name: str
    denominator_unit: str
    target_upper_rate: float
    confidence: float = 0.95
    interval: str = _ONE_SIDED_INTERVAL
    cluster_key: str = "fact_id"
    critical_failure_policy: CriticalFailurePolicy = "bounded"
    description: str = ""

    def __post_init__(self) -> None:
        if self.interval != _ONE_SIDED_INTERVAL:
            raise ValueError(f"Unsupported interval: {self.interval}")
        if not 0.0 < self.target_upper_rate < 1.0:
            raise ValueError("target_upper_rate must be in (0, 1)")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")

    @property
    def minimum_independent_units(self) -> int:
        return minimum_units_for_zero_failures(self.target_upper_rate, self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.name,
            "denominator_unit": self.denominator_unit,
            "target_upper_rate": self.target_upper_rate,
            "confidence": self.confidence,
            "interval": self.interval,
            "minimum_independent_units": self.minimum_independent_units,
            "cluster_key": self.cluster_key,
            "critical_failure_policy": self.critical_failure_policy,
            "description": self.description,
        }


@dataclass(frozen=True)
class GateObservation:
    """One scored probe. ``cluster_id`` identifies the independent unit."""

    cluster_id: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    independent_units: int
    failing_units: int
    observed_rate: float
    upper_bound: float | None
    minimum_required_units: int
    reason: str
    failing_clusters: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status,
            "independent_units": self.independent_units,
            "failing_units": self.failing_units,
            "observed_rate": round(self.observed_rate, 6),
            "upper_bound": None if self.upper_bound is None else round(self.upper_bound, 6),
            "minimum_required_units": self.minimum_required_units,
            "reason": self.reason,
            "failing_clusters": list(self.failing_clusters),
        }


def evaluate_gate(spec: GateSpec, observations: Iterable[GateObservation]) -> GateResult:
    """Cluster observations into independent units, then apply the exact bound.

    A unit fails when any observation in its cluster fails (conservative).
    Outcomes:
    - ``fail`` when the policy is zero_tolerance and any unit fails, or when
      the upper bound exceeds the target despite observed failures.
    - ``insufficient_data`` when no failure was observed but the denominator is
      too small for the bound to reach the target.
    - ``pass`` only when the upper bound is at or below the target.
    """
    clusters: dict[str, bool] = defaultdict(lambda: True)
    for observation in observations:
        clusters[observation.cluster_id] = clusters[observation.cluster_id] and observation.passed

    units = len(clusters)
    minimum_required = spec.minimum_independent_units
    if units == 0:
        return GateResult(
            gate=spec.name,
            status="insufficient_data",
            independent_units=0,
            failing_units=0,
            observed_rate=0.0,
            upper_bound=None,
            minimum_required_units=minimum_required,
            reason="No observations were provided",
        )

    failing = tuple(sorted(name for name, passed in clusters.items() if not passed))
    failures = len(failing)
    observed_rate = failures / units
    upper = clopper_pearson_upper(failures, units, spec.confidence)

    if spec.critical_failure_policy == "zero_tolerance" and failures > 0:
        return GateResult(
            gate=spec.name,
            status="fail",
            independent_units=units,
            failing_units=failures,
            observed_rate=observed_rate,
            upper_bound=upper,
            minimum_required_units=minimum_required,
            reason=f"{failures} failing unit(s) under a zero-tolerance policy",
            failing_clusters=failing,
        )

    if upper <= spec.target_upper_rate:
        return GateResult(
            gate=spec.name,
            status="pass",
            independent_units=units,
            failing_units=failures,
            observed_rate=observed_rate,
            upper_bound=upper,
            minimum_required_units=minimum_required,
            reason=(
                f"Upper bound {upper:.4%} <= target {spec.target_upper_rate:.4%} "
                f"over {units} independent {spec.denominator_unit} unit(s)"
            ),
            failing_clusters=failing,
        )

    if failures == 0:
        return GateResult(
            gate=spec.name,
            status="insufficient_data",
            independent_units=units,
            failing_units=0,
            observed_rate=0.0,
            upper_bound=upper,
            minimum_required_units=minimum_required,
            reason=(
                f"Zero failures, but {units} unit(s) < the {minimum_required} required "
                f"to support a <= {spec.target_upper_rate:.4%} claim at "
                f"{spec.confidence:.0%} confidence"
            ),
        )

    return GateResult(
        gate=spec.name,
        status="fail",
        independent_units=units,
        failing_units=failures,
        observed_rate=observed_rate,
        upper_bound=upper,
        minimum_required_units=minimum_required,
        reason=(
            f"Upper bound {upper:.4%} exceeds target {spec.target_upper_rate:.4%} "
            f"({failures}/{units} failing units)"
        ),
        failing_clusters=failing,
    )


# Promotion gates from the agreed design. Rates are failure rates, so the
# "new current-value strict accuracy >= 95%" requirement appears here as a
# <= 5% failure-rate bound. Pilot-scale corpora cannot satisfy the
# denominators; results below the minimum units are diagnostic only.
DEFAULT_PROMOTION_GATES: tuple[GateSpec, ...] = (
    GateSpec(
        name="new_current_value_strict",
        denominator_unit="supersession_scenario",
        target_upper_rate=0.05,
        cluster_key="scenario_id",
        critical_failure_policy="bounded",
        description="Current-value answers must be strictly correct for >= 95% of scenarios.",
    ),
    GateSpec(
        name="stale_current_answer",
        denominator_unit="supersession_scenario",
        target_upper_rate=0.01,
        cluster_key="scenario_id",
        critical_failure_policy="zero_tolerance",
        description="A superseded value presented as current is a critical failure.",
    ),
    GateSpec(
        name="old_new_conflation",
        denominator_unit="supersession_scenario",
        target_upper_rate=0.01,
        cluster_key="scenario_id",
        critical_failure_policy="bounded",
        description="Answers must not blend superseded and current values without dates.",
    ),
    GateSpec(
        name="hallucinated_citation",
        denominator_unit="citation_opportunity",
        target_upper_rate=0.005,
        cluster_key="question_family_id",
        critical_failure_policy="zero_tolerance",
        description="An invented record identifier is a critical failure.",
    ),
)


def _beta_ppf(quantile: float, a: float, b: float) -> float:
    """Inverse regularized incomplete beta via bisection (monotone CDF)."""
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _reg_inc_beta(a, b, mid) < quantile:
            low = mid
        else:
            high = mid
        if high - low < 1e-13:
            break
    return (low + high) / 2.0


def _reg_inc_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via the standard continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Modified Lentz evaluation of the incomplete-beta continued fraction."""
    max_iterations = 300
    epsilon = 3e-14
    tiny = 1e-300

    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h
