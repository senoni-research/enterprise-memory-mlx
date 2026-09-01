# Knowledge record contract

## Required fields

| Field | Purpose |
|---|---|
| `id` | Stable, unique record identifier |
| `domain` | Adapter grouping and routing label |
| `title` | Human-readable concept name |
| `statement` | Complete canonical knowledge to internalise |
| `source_uri` | Pointer to the authoritative source |

## Governance fields

| Field | Values |
|---|---|
| `status` | `active`, `draft`, `retired` |
| `sensitivity` | `public`, `internal_shared`, `restricted`, `secret` |
| `effective_from` / `effective_to` | ISO dates |

## Access examples

Each record should contain several questions written independently by a domain expert. Answers can be shorter than the canonical statement, but must not add a rule absent from the source. Keywords are evaluation aids, not training truth.

## Dynamic knowledge

Do not compile values such as current stock, current bank balances, current employee records, live requirement revisions or current approval status. Those belong in a tool or retrieved context with permission checks and evidence.
