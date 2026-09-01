# Security and governance

## Default rule

Only train knowledge that every person or service allowed to possess the adapter is also allowed to know.

A model adapter is not a document store with row-level permissions. Deleting the source document does not prove that the knowledge has disappeared from the adapter. Training data must therefore be approved before compilation, not filtered only at inference time.

## Required controls for a real company experiment

1. Maintain canonical source identifiers, owners, effective dates and classifications.
2. Exclude secrets, personal data, credentials, live transactions and need-to-know material.
3. Hash the source snapshot and training configuration.
4. Store adapters in an access-controlled registry and never commit them to Git.
5. Run extraction, memorisation and out-of-scope tests before distribution.
6. Rebuild rather than mutating an untracked adapter when the source changes.
7. Retain the external source for exact attribution, audit and deletion workflows.

## Repository safeguards

- `restricted`, `secret`, `draft` and `retired` records are excluded by default.
- `knowledge/private/`, model weights and adapters are Git-ignored.
- The compiler records exclusions and a snapshot hash.
- Including restricted records requires two explicit CLI flags.
- Unknown/live questions are trained toward source-system consultation rather than invention.
