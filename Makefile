.PHONY: test architecture inventory next-architecture-plan audit-results run-domain run-domain-dry list-lmstudio-models gauntlet-qwen-coder model-matrix-lmstudio model-matrix-smoke model-matrix-canon-big analyze-model-matrix smoke-qwen smoke-qwen36 smoke-qwen-coder clean-caches

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
	$(PY) -m compileall -q core scripts tests
	$(PY) -m tests.test_generated_code
	$(PY) -m tests.test_sbi_engine
	$(PY) -m tests.test_sandbox_eval
	$(PY) -m tests.test_real_warfarin_pkpd
	$(PY) -m tests.test_real_battery_nasa
	$(PY) -m tests.test_real_dream4
	$(PY) -m tests.test_real_ph_reactor
	$(PY) -m tests.test_real_parallel_wiener_hammerstein
	$(PY) -m tests.test_real_vitaldb
	$(PY) -m tests.test_real_cmapss
	$(PY) -m tests.test_real_uci_smartphone_har
	$(PY) -m tests.test_real_uci_gas_turbine
	$(PY) -m tests.test_real_uci_occupancy_detection
	$(PY) -m tests.test_real_uci_mhealth
	$(PY) -m tests.test_real_uci_isolet
	$(PY) -m tests.test_real_uci_pendigits
	$(PY) -m tests.test_uci_isolet_adapter_mechanics
	$(PY) -m tests.test_uci_isolet_synthetic_controls
	$(PY) -m tests.test_uci_isolet_real_baselines
	$(PY) -m tests.test_uci_pendigits_adapter_mechanics
	$(PY) -m tests.test_uci_pendigits_synthetic_controls
	$(PY) -m tests.test_uci_pendigits_real_baselines
	$(PY) -m tests.test_uci_optdigits_adapter_mechanics
	$(PY) -m tests.test_uci_optdigits_synthetic_controls
	$(PY) -m tests.test_uci_optdigits_real_baselines
	$(PY) -m tests.test_uci_daily_sports_source_gate
	$(PY) -m tests.test_uci_daily_sports_adapter_mechanics
	$(PY) -m tests.test_uci_daily_sports_synthetic_controls
	$(PY) -m tests.test_uci_daily_sports_real_baselines
	$(PY) -m tests.test_uci_spoken_arabic_digit_source_gate
	$(PY) -m tests.test_uci_spoken_arabic_digit_adapter_mechanics
	$(PY) -m tests.test_uci_spoken_arabic_digit_synthetic_controls
	$(PY) -m tests.test_uci_spoken_arabic_digit_real_baselines
	$(PY) -m tests.test_uci_statlog_landsat_source_gate
	$(PY) -m tests.test_uci_statlog_landsat_adapter_mechanics
	$(PY) -m tests.test_uci_statlog_landsat_synthetic_controls
	$(PY) -m tests.test_uci_statlog_landsat_real_baselines
	$(PY) -m tests.test_uci_image_segmentation_adapter_mechanics
	$(PY) -m tests.test_uci_image_segmentation_synthetic_controls
	$(PY) -m tests.test_uci_image_segmentation_real_baselines
	$(PY) -m tests.test_fashion_mnist_adapter_mechanics
	$(PY) -m tests.test_cifar10_adapter_mechanics
	$(PY) -m tests.test_cifar10_synthetic_controls
	$(PY) -m tests.test_cifar10_real_baselines
	$(PY) -m tests.test_svhn_source_gate
	$(PY) -m tests.test_svhn_adapter_mechanics
	$(PY) -m tests.test_svhn_synthetic_controls
	$(PY) -m tests.test_svhn_real_baselines
	$(PY) -m tests.test_cifar100_source_gate
	$(PY) -m tests.test_cifar100_adapter_mechanics
	$(PY) -m tests.test_cifar100_synthetic_controls
	$(PY) -m tests.test_cifar100_real_baselines
	$(PY) -m tests.test_uci_year_prediction_msd_source_gate
	$(PY) -m tests.test_uci_year_prediction_msd_adapter_mechanics
	$(PY) -m tests.test_uci_year_prediction_msd_synthetic_controls
	$(PY) -m tests.test_uci_year_prediction_msd_real_baselines
	$(PY) -m tests.test_uci_higgs_source_gate
	$(PY) -m tests.test_uci_higgs_adapter_mechanics
	$(PY) -m tests.test_uci_higgs_synthetic_controls
	$(PY) -m tests.test_uci_higgs_real_baselines
	$(PY) -m tests.test_smallnorb_source_gate
	$(PY) -m tests.test_smallnorb_adapter_mechanics
	$(PY) -m tests.test_smallnorb_synthetic_controls
	$(PY) -m tests.test_emnist_source_gate
	$(PY) -m tests.test_emnist_adapter_mechanics
	$(PY) -m tests.test_emnist_synthetic_controls
	$(PY) -m tests.test_stl10_source_gate
	$(PY) -m tests.test_stl10_adapter_mechanics
	$(PY) -m tests.test_stl10_synthetic_controls
	$(PY) -m tests.test_stl10_real_baselines
	$(PY) -m tests.test_speech_commands_source_gate
	$(PY) -m tests.test_speech_commands_adapter_mechanics
	$(PY) -m tests.test_speech_commands_synthetic_controls
	$(PY) -m tests.test_speech_commands_real_baselines
	$(PY) -m tests.test_uci_adult_source_gate
	$(PY) -m tests.test_uci_adult_adapter_mechanics
	$(PY) -m tests.test_uci_adult_synthetic_controls
	$(PY) -m tests.test_kmnist_source_gate
	$(PY) -m tests.test_kmnist_adapter_mechanics
	$(PY) -m tests.test_kmnist_synthetic_controls
	$(PY) -m tests.test_kmnist49_adapter_mechanics
	$(PY) -m tests.test_uci_ujiindoorloc_source_gate
	$(PY) -m tests.test_uci_ujiindoorloc_adapter_mechanics
	$(PY) -m tests.test_uci_ujiindoorloc_synthetic_controls
	$(PY) -m tests.test_uci_ujiindoorloc_real_baselines
	$(PY) -m tests.test_uci_wisdm_source_gate
	$(PY) -m tests.test_uci_wisdm_adapter_mechanics
	$(PY) -m tests.test_uci_wisdm_synthetic_controls
	$(PY) -m tests.test_uci_wisdm_real_baselines
	$(PY) -m tests.test_uci_emg_gestures_source_gate
	$(PY) -m tests.test_uci_mhealth_adapter_mechanics
	$(PY) -m tests.test_uci_mhealth_synthetic_controls
	$(PY) -m tests.test_uci_gas_turbine_adapter_mechanics
	$(PY) -m tests.test_uci_gas_turbine_synthetic_controls
	$(PY) -m tests.test_uci_gas_turbine_real_baselines
	$(PY) -m tests.test_uci_smartphone_har_adapter_mechanics
	$(PY) -m tests.test_uci_smartphone_har_synthetic_controls
	$(PY) -m tests.test_real_camels_us
	$(PY) -m tests.test_uci_flow_modulated_adapter_mechanics
	$(PY) -m tests.test_uci_electricity_load_adapter_mechanics
	$(PY) -m tests.test_uci_electricity_load_synthetic_controls
	$(PY) -m tests.test_uci_flow_modulated_synthetic_controls
	$(PY) -m tests.test_uci_flow_modulated_real_baselines
	$(PY) -m tests.test_camels_adapter_mechanics
	$(PY) -m tests.test_camels_synthetic_controls
	$(PY) -m tests.test_camels_real_train_baselines
	$(PY) -m tests.test_cmapss_synthetic_controls
	$(PY) -m tests.test_cmapss_real_source_train_baselines
	$(PY) -m tests.test_parallel_wiener_hammerstein_synthetic_controls
	$(PY) -m tests.test_parallel_wiener_hammerstein_non_llm_inference
	$(PY) -m tests.test_ph_reactor_synthetic_controls
	$(PY) -m tests.test_input_driven_eval
	$(PY) -m tests.test_ph_reactor_non_llm_inference
	$(PY) -m tests.test_run_domain_config
	$(PY) -m tests.test_domain_quarantine
	$(PY) -m tests.test_model_matrix_status
	$(PY) -m tests.test_analyze_model_matrix
	$(PY) -m tests.test_run_model_matrix_preflight
	$(PY) -m tests.test_baselines
	$(PY) -m tests.test_stress

architecture:
	@sed -n '1,220p' ARCHITECTURE.md

inventory:
	$(PY) scripts/repo_inventory.py

next-architecture-plan:
	@sed -n '1,260p' artifacts/evaluations/next_architecture_iteration_plan.md
	@printf '\nCurrent state:\n'
	@$(PY) -m json.tool artifacts/evaluations/next_architecture_iteration_state.json

audit-results:
	$(PY) scripts/audit_results.py

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

smoke-qwen:
	$(PY) scripts/smoke_qwen_lmstudio.py --model $(MODEL)

smoke-qwen36:
	$(PY) scripts/smoke_qwen_lmstudio.py --model qwen3.6-27b-nvfp4

smoke-qwen-coder:
	$(PY) scripts/smoke_qwen_lmstudio.py --model qwen/qwen3-coder-next

clean-caches:
	find . -type d -name '__pycache__' -not -path './.venv/*' -prune -exec rm -rf {} +
	find . -name '.DS_Store' -not -path './.venv/*' -delete
