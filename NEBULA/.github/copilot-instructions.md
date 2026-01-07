# NEBULA — Copilot / Agent Instructions ✅

Summary
- Top-level purpose: NEBULA is a simulation pipeline that builds downstream star/scene products from propagated satellite/TLE inputs.  Key upstream stages (BASE → LOS → ILLUM → FLUX → LOS_FLUX → ... → WINDOW_SOURCES) are declared centrally in `Utility/PIPELINE/NEBULA_PIPELINE_MANIFEST.py` and ordered in `Utility/PIPELINE/NEBULA_PIPELINE_STAGES.py`.

Quick facts agents must know
- Entrypoint vs file layout: The manifest (`NEBULA_PIPELINE_MANIFEST`) maps canonical stage IDs to a module.function callable and to the exact pickle files (owned outputs) under `NEBULA_OUTPUT/` that the manager will delete when invalidating a stage.
- Recompute policy: recompute is *deletion-driven* — the manager deletes *only* owned pickle files (never directories) and then executes stages forward. See `Utility/PIPELINE/NEBULA_PIPELINE_MANAGER.py` for the plan-building and deletion semantics.
- Stage IDs: use canonical IDs from `NEBULA_PIPELINE_STAGES.Stage`. Parsers accept aliases (e.g., `LOSFLUX`, `LOS-FLUX`) and are case-insensitive.
- Configuration & conventions:
  - `Configuration/NEBULA_PATH_CONFIG.py` contains `NEBULA_OUTPUT_DIR`, `ensure_output_directory()` and `configure_logging()` — prefer these helpers when writing scripts.
  - `Configuration/NEBULA_SENSOR_CONFIG.py` exports `ACTIVE_SENSOR` as the canonical sensor config used across modules.
  - Modules are intentionally "fail-fast" on configuration mismatches (they raise `RuntimeError` on invalid inputs).
- Function signatures for stages: The pipeline manager passes `logger` and `force_recompute` by default but filters kwargs to match the callable's signature. To be safe, implement stage entrypoints that accept `logger` and `force_recompute=False` (or `**kwargs`) for forward-compatibility.

Developer workflows (how to run things)
- Interactive Mode (Mode A, recommended in dev):
  from a Python REPL or Spyder session, import and call the manager:

  ```py
  from Utility.PIPELINE.NEBULA_PIPELINE_MANAGER import run_pipeline
  run_pipeline(recompute_from_stage="FLUX", stop_after_stage="PIXELS", dry_run=True)
  ```

- CLI / Script examples:
  - Edit `pipeline_run.py` knobs (`RECOMPUTE_FROM_STAGE`, `STOP_AFTER_STAGE`, `DRY_RUN`) and run:
    ```bash
    python pipeline_run.py
    ```
  - Or use the manager CLI (works when running from repo root so packages import):
    ```bash
    python -m Utility.PIPELINE.NEBULA_PIPELINE_MANAGER --recompute-from FLUX --stop-after PIXELS --dry-run
    python -m Utility.PIPELINE.NEBULA_PIPELINE_MANAGER --recompute-from FLUX --stop-after PIXELS --execute
    ```

Testing & quick checks
- Smoke tests are simple scripts (not a formal pytest suite). Example:
  ```bash
  python test_zodiacal_light_config.py
  ```
- Add small, focused smoke tests for new config modules using the repo’s fail-fast style.

External dependencies & environment notes
- Common runtime libs used across the repo: `numpy`, `astropy` (+ IERS data), `astroquery` (Gaia TAP), `skyfield` (DE440s), `tqdm`, `matplotlib`, `pyvista` (viewer). Check imports in modules under `Utility/` to find specific uses.
- Ephemeris: the default DE440s BSP file location is `Input/NEBULA_EPHEMERIS/de440s.bsp` — many illumination/pointing functions accept an explicit path.
- Zodiacal-light backend: the WSL/m4opt bridge is used in `Utility/ZODIACAL_LIGHT/` and mentions WSL or an external binary; follow module comments for platform-specific preflight steps (IERS preflight, WSL guidance).

Project-specific patterns to follow
- Pickle-centric pipeline: intermediate artifacts are pickles in `NEBULA_OUTPUT/*_SatPickles` or specific subdirs (e.g., `STARS/<catalog>/`). When invalidating, the manager only removes the pickles listed in the manifest.
- Manifest-first edits: To add/modify upstream stages, update three places:
  1) `Utility/PIPELINE/NEBULA_PIPELINE_STAGES.py` (add or validate Stage and ordering)
  2) `Utility/PIPELINE/NEBULA_PIPELINE_MANIFEST.py` (add `StageSpec` with `entrypoint` and `owned_outputs_rel`)
  3) Implement the entrypoint callable under `Utility/` and make it importable (and test locally with `dry_run=True`).
- Logging: use the provided `configure_logging()`/module-level `get_logger()` patterns so logs are consistent and written to `NEBULA_OUTPUT/logs/`.

If something is unclear or you want me to expand any section (examples, dependency install steps, or a checklist for adding a stage), tell me which area and I’ll iterate. 🔧