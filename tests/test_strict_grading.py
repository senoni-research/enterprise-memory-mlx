from __future__ import annotations

import pytest

from enterprise_memory_mlx.split_contract import CriticalSlot
from enterprise_memory_mlx.strict_grading import grade_critical_slots


def _slot(
    slot: str,
    expected: str | None = None,
    forbidden: tuple[str, ...] = (),
) -> CriticalSlot:
    return CriticalSlot(slot=slot, expected=expected, forbidden=forbidden)


def test_number_words_and_digits_are_equivalent() -> None:
    slots = (_slot("number", "twenty-five"),)
    for answer in ("The allowance is twenty-five days.", "The allowance is 25 days."):
        result = grade_critical_slots(answer, slots)
        assert result.status == "pass"
        assert result.slot_results[0].expected == "25"
        assert "25" in result.extracted_values.number

    five = grade_critical_slots("Up to five days may be carried over.", (_slot("number", "5"),))
    fifteen = grade_critical_slots(
        "A human must reply within fifteen minutes.",
        (_slot("number", "15"),),
    )
    sixty = grade_critical_slots(
        "Unused accounts suspend after sixty days.",
        (_slot("number", "sixty"),),
    )
    assert five.status == "pass"
    assert fifteen.status == "pass"
    assert sixty.status == "pass"
    assert five.extracted_values.number == ("5",)
    assert fifteen.extracted_values.number == ("15",)
    assert sixty.extracted_values.number == ("60",)


def test_number_is_not_confused_with_time_date_percent_or_currency() -> None:
    result = grade_critical_slots(
        "Pay 10% on 31 March by 10:00, not the £500 threshold.",
        (_slot("number", "ten"),),
    )
    assert result.status == "hard_fail"
    assert result.slot_results[0].present is False
    assert "10" not in result.extracted_values.number
    assert "31" not in result.extracted_values.number
    assert "500" not in result.extracted_values.number


def test_currency_preserves_identity_and_grouping() -> None:
    slots = (_slot("currency_amount", "£750", forbidden=("£500",)),)
    assert grade_critical_slots("The threshold is £750.", slots).status == "pass"
    grouped = grade_critical_slots(
        "The threshold is £2,000.",
        (_slot("currency_amount", "£2000"),),
    )
    assert grouped.status == "pass"

    for answer in (
        "The threshold is £500.",
        "The threshold is $750.",
        "The threshold is €750.",
    ):
        result = grade_critical_slots(answer, slots)
        assert result.status == "hard_fail"
        assert result.slot_results[0].present is False or result.slot_results[0].forbidden_present


def test_forbidden_currency_with_expected_is_a_hard_fail() -> None:
    result = grade_critical_slots(
        "The threshold is £750, not the old £500 rule.",
        (_slot("currency_amount", "£750", forbidden=("£500",)),),
    )
    assert result.status == "hard_fail"
    assert result.slot_results[0].present is True
    assert result.slot_results[0].forbidden_present is True
    assert result.slot_results[0].contradiction is True


def test_historical_currency_pair_passes_when_both_are_expected() -> None:
    result = grade_critical_slots(
        "Until 31 August 2026 the threshold was £500; from 1 September 2026 it is £750.",
        (
            _slot("currency_amount", "£750"),
            _slot("currency_amount", "£500"),
            _slot("date", "1 September 2026"),
        ),
    )
    assert result.status == "pass"
    assert result.failed_slot_indexes == ()


def test_comparators_preserve_direction() -> None:
    above = (_slot("comparator", "above"),)
    assert grade_critical_slots("Spend above £500 needs approval.", above).status == "pass"
    assert grade_critical_slots("Spend over £500 needs approval.", above).status == "pass"
    assert grade_critical_slots("Spend greater than £500 needs approval.", above).status == "pass"

    for answer in (
        "Spend of at least £500 needs approval.",
        "Spend below £500 needs approval.",
        "Complete the checks before booking.",
    ):
        assert grade_critical_slots(answer, above).status == "hard_fail"

    before = (_slot("comparator", "before"),)
    assert grade_critical_slots("Approval is required before booking.", before).status == "pass"
    assert grade_critical_slots("Approval is required after booking.", before).status == "hard_fail"


def test_must_versus_must_not() -> None:
    slots = (_slot("negation", "must not"),)
    allowed = grade_critical_slots(
        "An item must not be split to avoid the threshold.",
        slots,
    )
    forbidden = grade_critical_slots(
        "An item must be split to stay under the threshold.",
        slots,
    )
    assert allowed.status == "pass"
    assert forbidden.status == "hard_fail"


def test_times_normalize_12_and_24_hour_forms() -> None:
    morning = (_slot("time", "10:00"),)
    afternoon = (_slot("time", "15:00"),)
    assert grade_critical_slots("Be reachable from 10:00 UK time.", morning).status == "pass"
    assert grade_critical_slots("Be reachable from 10am.", morning).status == "pass"
    assert grade_critical_slots("The window ends at 15:00.", afternoon).status == "pass"
    assert grade_critical_slots("The window ends at 3pm.", afternoon).status == "pass"
    assert grade_critical_slots("The window ends at 10:00.", afternoon).status == "hard_fail"


def test_dates_preserve_precision() -> None:
    march = grade_critical_slots(
        "Carried leave must be used by 31 March.",
        (_slot("date", "31 March"),),
    )
    september = grade_critical_slots(
        "The new threshold applies from 2026-09-01.",
        (_slot("date", "1 September 2026"),),
    )
    month_only = grade_critical_slots(
        "That value applied until 31 August 2026.",
        (_slot("date", "August 2026"),),
    )
    assert march.status == "pass"
    assert september.status == "pass"
    assert month_only.status == "pass"
    month_without_day = grade_critical_slots(
        "The rule changed in September 2026.",
        (_slot("date", "2026-09-01"),),
    )
    assert month_without_day.status == "review_required"
    assert month_without_day.slot_results[0].present is False
    assert month_without_day.failed_slot_indexes == ()


def test_entity_uses_token_boundaries_and_hyphen_variants() -> None:
    hr = (_slot("entity", "HR"),)
    assert grade_critical_slots("HR can approve an exceptional extension.", hr).status == "pass"
    assert (
        grade_critical_slots("The threat model does not mention this team.", hr).status
        == "hard_fail"
    )

    device = (_slot("entity", "company-managed device"),)
    assert grade_critical_slots(
        "Use a company managed device over the approved secure connection.",
        device,
    ).status == "pass"
    assert grade_critical_slots(
        "Use a company-managed device over the approved secure connection.",
        device,
    ).status == "pass"

    record = (_slot("entity", "FIN-EXP-001"),)
    assert grade_critical_slots("See FIN-EXP-001 for the threshold.", record).status == "pass"
    assert grade_critical_slots("See EXP-001 for the threshold.", record).status == "hard_fail"


def test_unit_and_record_status_are_exact() -> None:
    unit = (_slot("unit", "calendar days"),)
    assert (
        grade_critical_slots("Submit the receipt within ten calendar days.", unit).status == "pass"
    )
    assert (
        grade_critical_slots("Submit the receipt within ten hours.", unit).status == "hard_fail"
    )

    status = (_slot("record_status", "active"),)
    assert grade_critical_slots("The governed record is active.", status).status == "pass"
    assert grade_critical_slots("The governed record is retired.", status).status == "hard_fail"
    mixed_status = grade_critical_slots(
        "The governed record is active and also retired.",
        status,
    )
    assert mixed_status.status == "review_required"
    assert mixed_status.slot_results[0].present is True
    assert mixed_status.slot_results[0].contradiction is False


def test_provenance_forbids_record_citations() -> None:
    slots = (_slot("provenance", forbidden=("[record:]",)),)
    assert grade_critical_slots(
        "There is no authoritative company record for that.",
        slots,
    ).status == "pass"
    leaked = grade_critical_slots("See [record: FIN-EXP-001] for details.", slots)
    assert leaked.status == "hard_fail"
    assert leaked.slot_results[0].forbidden_present is True


def test_multiple_slots_fail_closed_when_exactly_one_is_wrong() -> None:
    slots = (
        _slot("currency_amount", "£750"),
        _slot("currency_amount", "£500"),
        _slot("date", "1 September 2026"),
    )
    result = grade_critical_slots(
        "From 1 September 2026 the threshold is £750.",
        slots,
    )
    assert result.status == "hard_fail"
    assert result.failed_slot_indexes == (1,)
    assert result.slot_results[0].status == "pass"
    assert result.slot_results[1].status == "hard_fail"
    assert result.slot_results[2].status == "pass"


def test_appended_competing_number_or_negation_needs_review() -> None:
    number = grade_critical_slots(
        "Twenty-five days of annual leave in addition to public holidays, or thirty days.",
        (_slot("number", "twenty-five"),),
    )
    assert number.status == "review_required"
    assert number.slot_results[0].present is True
    assert number.slot_results[0].contradiction is False
    assert number.failed_slot_indexes == ()

    negation = grade_critical_slots(
        "An item must not be split into smaller transactions to avoid the £500 approval threshold. "
        "An item must be split.",
        (_slot("negation", "must not"),),
    )
    assert negation.status == "review_required"
    assert negation.slot_results[0].present is True
    assert negation.slot_results[0].contradiction is False
    assert negation.failed_slot_indexes == ()


def test_empty_output_unsupported_type_and_malformed_expectations_fail_closed() -> None:
    empty = grade_critical_slots("", (_slot("number", "ten"),))
    assert empty.status == "hard_fail"
    assert empty.slot_results[0].present is False

    unsupported = grade_critical_slots("anything", (_slot("lemma", "x"),))
    assert unsupported.status == "hard_fail"
    assert "unsupported slot type" in unsupported.reasons[0]

    malformed = grade_critical_slots("The answer is ten.", (_slot("number", "not-a-number"),))
    assert malformed.status == "hard_fail"
    assert "malformed expectation" in malformed.reasons[0]

    blank = grade_critical_slots("The answer is ten.", (CriticalSlot(slot="number"),))
    assert blank.status == "hard_fail"
    assert "malformed expectation" in blank.reasons[0]


def test_no_slots_is_not_a_pass() -> None:
    result = grade_critical_slots("Any fluent answer.", ())
    assert result.status == "no_slots"
    assert result.slot_results == ()
    assert result.failed_slot_indexes == ()
    assert result.reasons == ("no critical slots declared",)


def test_repeated_calls_are_deterministic() -> None:
    slots = (
        _slot("currency_amount", "£500"),
        _slot("comparator", "above"),
        _slot("entity", "budget owner"),
    )
    answer = (
        "Because the spend is above £500, the relevant budget owner must give "
        "written approval before the booking is made."
    )
    first = grade_critical_slots(answer, slots)
    second = grade_critical_slots(answer, slots)
    assert first == second
    assert first.status == "pass"


def test_sibling_number_slots_are_not_treated_as_contradictions() -> None:
    result = grade_critical_slots(
        "P1 within fifteen minutes, P2 within two business hours, P3 by the next business day.",
        (_slot("number", "fifteen"), _slot("number", "two")),
    )
    assert result.status == "pass"
    assert result.extracted_values.number == ("15", "2")


def test_unrelated_negation_does_not_satisfy_a_scoped_rule() -> None:
    result = grade_critical_slots(
        "No. Two people who wrote the change may approve it.",
        (_slot("negation", "did not author"),),
    )
    assert result.status == "hard_fail"
    assert result.slot_results[0].present is False


def test_omitted_time_zone_is_not_inferred() -> None:
    result = grade_critical_slots(
        "Be reachable from 10:00.",
        (_slot("time", "10:00 UTC"),),
    )
    assert result.status == "review_required"
    assert result.slot_results[0].present is False
    assert result.failed_slot_indexes == ()


def test_unrelated_legitimate_number_is_not_a_certain_contradiction() -> None:
    result = grade_critical_slots(
        "Full-time staff receive twenty-five days of annual leave. "
        "Receipts must be submitted within ten calendar days.",
        (_slot("number", "twenty-five"),),
    )
    assert result.status == "review_required"
    assert result.slot_results[0].present is True
    assert result.slot_results[0].contradiction is False
    assert result.extracted_values.number == ("25", "10")
    assert result.failed_slot_indexes == ()
    assert result.reasons == ("slot 0 number: scope cannot be determined",)


def test_wrong_action_negation_is_not_a_pass() -> None:
    result = grade_critical_slots(
        "Confidential information must not be handled on a personal laptop. "
        "An item may be split to stay under the threshold.",
        (_slot("negation", "must not"),),
    )
    assert result.status == "review_required"
    assert result.slot_results[0].present is True
    assert result.slot_results[0].contradiction is False
    assert result.failed_slot_indexes == ()
    assert result.reasons == ("slot 0 negation: scope cannot be determined",)


def test_omitted_date_precision_is_not_a_certain_failure() -> None:
    result = grade_critical_slots(
        "The new threshold applies from September 2026.",
        (_slot("date", "1 September 2026"),),
    )
    assert result.status == "review_required"
    assert result.slot_results[0].present is False
    assert result.slot_results[0].contradiction is False
    assert result.extracted_values.date == ("2026-09",)
    assert result.failed_slot_indexes == ()
    assert "omitted precision" in result.reasons[0]


def test_boolean_answer_is_rejected() -> None:
    with pytest.raises(TypeError, match="answer must be a string"):
        grade_critical_slots(True, (_slot("number", "ten"),))  # type: ignore[arg-type]
