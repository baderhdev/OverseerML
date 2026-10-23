UV ?= uv

.PHONY: reproduce

# Rebuilds every result in RESULTS.md from scratch; see RESULTS.md for what each step does.
reproduce:
	$(UV) sync --locked
	$(UV) run overseer-train
	$(UV) run overseer-drift
	$(UV) run overseer-retrain
	docker build -t overseerml .
	$(UV) run python -m overseerml.loadtest
	$(UV) run python -m overseerml.results
