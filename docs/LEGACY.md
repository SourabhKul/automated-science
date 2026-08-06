# Historical Compatibility Surface

The project began as a collection of domain-specific orchestrators and
loaders. The current entrypoint is the config-driven runner in
`scripts/run_domain.py`. Older root-level `qwen36_*`, `gemma4_*`, and
specialized orchestration files are preserved in the local cleanup backup,
rather than published beside the active implementation.

New work should add a domain configuration, a focused adapter, or a dedicated
script under `core/` and `scripts/` rather than copying an old orchestrator.
Compatibility loaders that remain useful for local replay live under
`scripts/legacy/`; they are not the preferred extension surface. The local
snapshot remains available for historical replay. A later naming and directory
migration can provide forwarding shims after the public project name is chosen.
