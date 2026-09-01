# Research notes and implementation boundaries

This repository is informed by three recent preprints and the current MLX-LM implementation.

## Inject, Align, Recover

**Qian Kou, Xiaofeng Shi, Xiaosong Qiu and Hua Zhou, “Inject, Align, Recover: Staged Post-Training for Retrieval-Free Document Knowledge Internalization,” arXiv:2608.20281, 20 August 2026.**

The paper separates structured document injection, QA access alignment and general-capability recovery. This repository implements the separation and the injection objective family. Its Recover stage is deliberately labelled **Recover-lite**: it resumes the LoRA adapter with low-rate general replay. It does not claim to reproduce the paper's checkpoint merge, TIES, DARE or SLERP results.

## Continual facts in weights

**Charles O'Neill, “Can a Language Model Learn Facts Continually in Its Weights?”, arXiv:2607.11020, 13 July 2026.**

The study finds that broad, diverse study representations create more usable knowledge than bare-statement writes, but sequential later writes can make earlier facts behaviourally inaccessible. The compiler therefore creates several access paths for each record, while `eval/retention.jsonl` keeps every record in the regression suite. This is a measurement discipline, not a solution to catastrophic interference.

## Parametric knowledge graph adapters

**Martino M. L. Pulici, Cuong Xuan Chu, Evgeny Kharlamov and Volker Tresp, “A Storage-Retrieval Gap in Parametric Knowledge Graph Memory,” arXiv:2608.25489, 26 August 2026.**

The paper compiles entity subgraphs into per-entity LoRA adapters and shows a gap between storing knowledge and selecting the right adapter. This repository starts at domain granularity to keep a laptop experiment tractable. Its lexical router is a transparent baseline, not a state-of-the-art learned selector.

## MLX-LM

The implementation targets **MLX-LM v0.31.3**. MLX-LM supports LoRA, QLoRA, DoRA and full fine-tuning; local JSONL chat/completion/text formats; prompt masking; gradient accumulation; checkpointing; adapter resume; and adapter fusion.

Official sources:

- https://github.com/ml-explore/mlx-lm
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- https://github.com/ml-explore/mlx

## Claims this repository does not make

- It does not prove that all company knowledge can or should be stored in weights.
- It does not make trained knowledge revocable or permission-aware.
- It does not implement reliable online continual learning.
- It does not guarantee that a low training loss means reliable recall.
- It does not solve adapter routing or composition.
- It does not replace retrieval, tools, databases, provenance or human sign-off.
