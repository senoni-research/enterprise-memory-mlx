from __future__ import annotations

from pathlib import Path

from .compiler import SYSTEM_PROMPT


def interactive_chat(
    model_name: str,
    adapter_path: Path | None,
    max_tokens: int = 320,
    *,
    allow_scientifically_invalid: bool = False,
) -> None:
    from .legacy_guard import guard_legacy_component

    guard_legacy_component(
        "inference.interactive_chat",
        allow_scientifically_invalid=allow_scientifically_invalid,
    )
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError('MLX-LM is required. Install with: pip install -e ".[mac]"') from exc

    model, tokenizer = load(
        model_name,
        adapter_path=str(adapter_path) if adapter_path else None,
    )
    sampler = make_sampler(temp=0.0)
    print("Enterprise memory chat. Type /quit to exit.")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question in {"/quit", "/exit"}:
            return
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        answer = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        print(f"model> {str(answer).strip()}")
