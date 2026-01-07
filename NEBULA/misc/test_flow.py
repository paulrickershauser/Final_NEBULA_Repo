# -*- coding: utf-8 -*-
"""
NEBULA pipeline manager smoke test (Spyder-friendly)
- Tests imports for Utility.PIPELINE.{STAGES,MANIFEST,MANAGER}
- Tests plan resolution for recompute_from / stop_after (dry-run only)
- Optionally resolves manifest entrypoints to ensure module+function names are correct

Python 3.10+ recommended.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


# ---------------------------------------------------------------------
# 0) Point Python at your NEBULA repo root (so "Configuration" and "Utility" import)
# ---------------------------------------------------------------------
NEBULA_ROOT = Path(r"C:\Users\prick\Desktop\Research\NEBULA").resolve()

if not NEBULA_ROOT.exists():
    raise FileNotFoundError(f"NEBULA_ROOT not found: {NEBULA_ROOT}")

# sys.path controls module search; it is legal to modify it for a test harness. 
if str(NEBULA_ROOT) not in sys.path:
    sys.path.insert(0, str(NEBULA_ROOT))

print("\n=== ENV CHECK ===")
print("Python:", sys.version)
print("NEBULA_ROOT:", NEBULA_ROOT)
print("sys.path[0]:", sys.path[0])

# Basic sanity checks for package layout. Regular packages commonly use __init__.py. 
expected = [
    NEBULA_ROOT / "Configuration",
    NEBULA_ROOT / "Utility",
    NEBULA_ROOT / "Utility" / "PIPELINE",
    NEBULA_ROOT / "Utility" / "PIPELINE" / "__init__.py",
]
for p in expected:
    print(f"Exists? {p} -> {p.exists()}")


# ---------------------------------------------------------------------
# 1) Import-only smoke test (no side effects expected)
# ---------------------------------------------------------------------
print("\n=== IMPORT TESTS ===")
try:
    from Utility.PIPELINE import NEBULA_PIPELINE_STAGES as stages
    print("Imported:", stages.__file__)
except Exception:
    print("FAILED importing Utility.PIPELINE.NEBULA_PIPELINE_STAGES")
    traceback.print_exc()
    raise

try:
    from Utility.PIPELINE import NEBULA_PIPELINE_MANIFEST as manifest
    print("Imported:", manifest.__file__)
except Exception:
    print("FAILED importing Utility.PIPELINE.NEBULA_PIPELINE_MANIFEST")
    traceback.print_exc()
    raise

try:
    from Utility.PIPELINE import NEBULA_PIPELINE_MANAGER as manager
    print("Imported:", manager.__file__)
except Exception:
    print("FAILED importing Utility.PIPELINE.NEBULA_PIPELINE_MANAGER")
    traceback.print_exc()
    raise


# ---------------------------------------------------------------------
# 2) Show stage ordering + quick parser checks
# ---------------------------------------------------------------------
print("\n=== STAGE ORDER ===")
print([s.value for s in stages.STAGE_ORDER])

print("\n=== PARSE CHECKS (should show Stage or None) ===")
for s in ["LOS_FLUX", "LOSFLUX", "los-flux", "bad_stage", None]:
    parsed = stages.try_parse_stage(s)
    print(f"{s!r:>10} -> {parsed!r}")


# ---------------------------------------------------------------------
# 3) Entrypoint resolution test (imports pickler modules; does NOT execute them)
#    Set to False if you want to avoid importing heavy modules during testing.
# ---------------------------------------------------------------------
DO_RESOLVE_ENTRYPOINTS = True

if DO_RESOLVE_ENTRYPOINTS:
    print("\n=== ENTRYPOINT RESOLUTION (IMPORT ONLY) ===")
    for spec in manifest.iter_stage_specs_in_order():
        try:
            fn = manifest.load_entrypoint(spec.entrypoint)  # just import + getattr
            print(f"{spec.stage.value:>9} -> {fn.__module__}.{fn.__name__}")
        except Exception:
            print(f"FAILED resolving entrypoint for stage {spec.stage.value}")
            traceback.print_exc()
            raise


# ---------------------------------------------------------------------
# 4) Dry-run plan tests (verifies recompute_from / stop_after semantics)
# ---------------------------------------------------------------------
print("\n=== DRY-RUN PLAN TESTS ===")

TEST_CASES = [
    # (recompute_from, stop_after)
    (None, None),                # full run, no invalidation
    ("PIXELS", "PIXELS"),         # run PIXELS only; invalidate PIXELS only
    ("FLUX", "PIXELS"),           # run through PIXELS; invalidate FLUX..PIXELS
    ("LOSFLUX", "LOS_FLUX"),      # alias handling; should behave like LOS_FLUX
    ("BAD_STAGE", "BAD_STAGE"),   # invalid stop -> full run; invalid recompute -> invalidate none
]

for recompute_from, stop_after in TEST_CASES:
    print("\n--- CASE ---")
    print("recompute_from_stage =", recompute_from)
    print("stop_after_stage     =", stop_after)

    # Build plan and print it (does not delete or execute).
    plan = manager.build_plan(
        recompute_from_stage=recompute_from,
        stop_after_stage=stop_after,
        output_dir=None,
        invalidate_beyond_stop_after=False,  # safe default
    )
    manager.print_plan(plan)

    # Run pipeline in dry-run mode (prints the plan again; still no side effects).
    _ = manager.run_pipeline(
        recompute_from_stage=recompute_from,
        stop_after_stage=stop_after,
        dry_run=True,
    )


# ---------------------------------------------------------------------
# 5) (Optional) Controlled execution test — OFF by default
#    Only flip this once dry-run output looks correct.
# ---------------------------------------------------------------------
EXECUTE_PIXELS_ONLY = False

if EXECUTE_PIXELS_ONLY:
    print("\n=== EXECUTION TEST: RECOMPUTE PIXELS ONLY ===")
    _ = manager.run_pipeline(
        recompute_from_stage="PIXELS",
        stop_after_stage="PIXELS",
        dry_run=False,  # this will delete owned PIXELS pickles, then rebuild PIXELS
    )
    print("Executed PIXELS recompute successfully.")

print("\nAll tests completed.")
