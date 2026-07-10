.PHONY: test architecture inventory next-architecture-plan compile-results audit-results build-data build-real-data run-domain run-domain-dry list-lmstudio-models gauntlet-qwen-coder model-matrix-lmstudio model-matrix-smoke model-matrix-canon-big analyze-model-matrix baselines baseline-smoke smoke-qwen smoke-qwen36 smoke-qwen-coder clean-caches

PY ?= .venv/bin/python
DOMAIN ?= ecology
FAMILY ?= qwen36
MODEL ?= qwen/qwen3-coder-next
MODEL_MATRIX_DOMAINS ?= ecology
MODEL_MATRIX_EPOCHS ?= 1
MODEL_MATRIX_BASE_URL ?= http://localhost:1234/v1
MODEL_MATRIX_RUN_ID ?= monitor_$(shell date +%Y%m%d_%H%M%S)
MODEL_MATRIX_PER_MODEL_TIMEOUT ?= 14400
MODEL_MATRIX_CONTEXT_LENGTH ?= 32768
MODEL_MATRIX_MAX_TOKENS ?= 16384

test:
	$(PY) -m py_compile core/*.py *.py scripts/*.py
	$(PY) test_generated_code.py
	$(PY) test_sbi_engine.py
	$(PY) test_sandbox_eval.py
	$(PY) test_real_warfarin_pkpd.py
	$(PY) test_real_battery_nasa.py
	$(PY) test_run_domain_config.py
	$(PY) test_domain_quarantine.py
	$(PY) test_model_matrix_status.py
	$(PY) test_analyze_model_matrix.py
	$(PY) test_run_model_matrix_preflight.py
	$(PY) test_baselines.py
	$(PY) test_stress.py

architecture:
	@sed -n '1,220p' ARCHITECTURE.md

inventory:
	$(PY) scripts/repo_inventory.py

next-architecture-plan:
	@sed -n '1,260p' artifacts/evaluations/next_architecture_iteration_plan.md
	@printf '\nCurrent state:\n'
	@$(PY) -m json.tool artifacts/evaluations/next_architecture_iteration_state.json

compile-results:
	$(PY) compile_paper_results.py

audit-results:
	$(PY) scripts/audit_results.py

build-data:
	./bootstrap_data.sh

build-real-data:
	$(PY) build_real_data.py

run-domain:
	$(PY) scripts/run_domain.py $(DOMAIN) --family $(FAMILY)

run-domain-dry:
	$(PY) scripts/run_domain.py $(DOMAIN) --family $(FAMILY) --dry-run

list-lmstudio-models:
	$(PY) scripts/lmstudio_models.py

gauntlet-qwen-coder:
	$(PY) scripts/run_gauntlet.py --family qwen3_coder_next --epochs 5 --held-out

model-matrix-lmstudio:
	$(PY) scripts/run_model_matrix.py --epochs 5 --held-out

model-matrix-smoke:
	$(PY) scripts/run_model_matrix.py --base-url $(MODEL_MATRIX_BASE_URL) --domains $(MODEL_MATRIX_DOMAINS) --epochs $(MODEL_MATRIX_EPOCHS) --held-out --run-id $(MODEL_MATRIX_RUN_ID) --context-length $(MODEL_MATRIX_CONTEXT_LENGTH) --max-tokens $(MODEL_MATRIX_MAX_TOKENS)

model-matrix-canon-big:
	$(PY) scripts/run_model_matrix.py --base-url $(MODEL_MATRIX_BASE_URL) --epochs 20 --held-out --run-id $(MODEL_MATRIX_RUN_ID) --per-model-timeout $(MODEL_MATRIX_PER_MODEL_TIMEOUT) --context-length $(MODEL_MATRIX_CONTEXT_LENGTH) --max-tokens $(MODEL_MATRIX_MAX_TOKENS) --resume

analyze-model-matrix:
	$(PY) scripts/analyze_model_matrix.py

baselines:
	$(PY) run_baselines.py

baseline-smoke:
	$(PY) run_baselines.py --domains ecology --pysr-iterations 1 --output artifacts/baselines/baseline_smoke.json

smoke-qwen:
	$(PY) scripts/smoke_qwen_lmstudio.py --model $(MODEL)

smoke-qwen36:
	$(PY) scripts/smoke_qwen_lmstudio.py --model qwen3.6-27b-nvfp4

smoke-qwen-coder:
	$(PY) scripts/smoke_qwen_lmstudio.py --model qwen/qwen3-coder-next

clean-caches:
	find . -type d -name '__pycache__' -not -path './.venv/*' -prune -exec rm -rf {} +
	find . -name '.DS_Store' -not -path './.venv/*' -delete
