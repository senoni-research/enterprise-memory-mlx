.PHONY: setup doctor compile dry-run train ablation evaluate route chat benchmark review package test lint clean

LEGACY_DISABLED = Legacy pipeline disabled: use the revised evidence workflow. There is no override.

setup:
	./scripts/bootstrap_macos.sh

doctor:
	.venv/bin/emmlx doctor

compile:
	@echo "$(LEGACY_DISABLED)" >&2
	@exit 2

dry-run:
	@echo "$(LEGACY_DISABLED)" >&2
	@exit 2

train:
	@echo "$(LEGACY_DISABLED)" >&2
	@exit 2

ablation:
	@echo "$(LEGACY_DISABLED)" >&2
	@exit 2

evaluate:
	@echo "$(LEGACY_DISABLED)" >&2
	@exit 2

route:
	@echo "$(LEGACY_DISABLED)" >&2
	@exit 2

chat:
	@echo "$(LEGACY_DISABLED)" >&2
	@exit 2

benchmark:
	.venv/bin/emmlx benchmark --dry-run

review:
	@test -n "$(REVIEWER)" || { echo 'Usage: make review REVIEWER="Your Name"' >&2; exit 2; }
	.venv/bin/emmlx review --reviewer "$(REVIEWER)"

package:
	.venv/bin/python scripts/package_source.py

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

clean:
	rm -rf artifacts/datasets artifacts/configs artifacts/adapters artifacts/manifests artifacts/registry artifacts/eval
	touch artifacts/.gitkeep
