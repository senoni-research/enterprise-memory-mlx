# Qualification to the exact-memory technical note

This is a separately versioned qualification. It does not rewrite
`docs/exact-memory-long-context-qlora.md` and does not change the trained
adapter from `run-20260910T210406Z-seed42`.

The 2026-09-11 evaluation-pending technical review found that:

- the inspected Q/K/V kernel comparison omitted explicit `argnums`, so it
  does not establish Q, K and V gradient equivalence;
- the full-model numerical check did not cross the production 256-token
  attention or projection chunks;
- declared gradient/update tolerances can accept a missing update of size
  `5e-5`;
- historical executed-code identity is incomplete;
- historical 96-GiB application-memory compliance is not established.

Those limitations remain open. This evaluation-only completion does not
rerun backward or optimizer tests and does not make a stronger exactness
claim from the existing reports.

The original `int(None)` after-review failure is recorded as a strong
matching hypothesis for a discarded `set_wired_limit` return value, not as
a proven historical call site.
