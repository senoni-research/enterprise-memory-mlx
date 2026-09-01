# Knowledge input

`records.jsonl` is deliberately synthetic. Replace it with approved company knowledge records that have explicit provenance and a sharing classification.

Every record requires:

- `id`, `domain`, `title`, `statement`, and `source_uri`
- `status`: `active`, `draft`, or `retired`
- `sensitivity`: `public`, `internal_shared`, `restricted`, or `secret`
- several independently worded question/answer pairs for access alignment and held-out evaluation

The compiler excludes `restricted`, `secret`, `draft`, and `retired` records by default. This is a governance control, not a claim that trained weights can enforce source-level permissions. Put truly private source material under `knowledge/private/`; that directory is ignored by Git.
