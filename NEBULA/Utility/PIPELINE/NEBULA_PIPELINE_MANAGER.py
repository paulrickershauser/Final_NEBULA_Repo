"""
NEBULA_PIPELINE_MANAGER
======================

Execution engine for the NEBULA pipeline (full chain through ZODIACAL_LIGHT).

This module implements the policy you confirmed:

1) Strict chain:
   BASE → LOS → ILLUM → FLUX → LOS_FLUX → POINTING → ICRS → PIXELS → 
   TARGET_PHOTONS → TRACKING_MODE → SKY → GAIA → STAR_PROJECTION → STAR_PHOTONS
   → WINDOW_SOURCES-> ZODIACAL_LIGHT


2) Recompute is deletion-driven:
   - "Recompute from stage X" means: delete ONLY owned pickle outputs for X and
     downstream stages (never delete directories), then run forward.
   - Because the owned pickle(s) no longer exist, picklers will naturally take
     the "compute" path even if their internal defaults are force_recompute=False.

3) STOP_AFTER_STAGE semantics:
   - If STOP_AFTER_STAGE is a valid stage, run only through it (inclusive).
   - If STOP_AFTER_STAGE is invalid or None, run the full pipeline.

4) DRY_RUN:
   - Prints a plan (run list, invalidate list, delete list with exists status,
     and stage entrypoints). Does not delete or execute.

Usage (Mode A; Spyder-friendly)
-------------------------------
from Utility.PIPELINE.NEBULA_PIPELINE_MANAGER import run_pipeline

run_pipeline(
    recompute_from_stage="FLUX",
    stop_after_stage="PIXELS",
    dry_run=True,
)

Then once the plan looks correct:

run_pipeline(
    recompute_from_stage="FLUX",
    stop_after_stage="PIXELS",
    dry_run=False,
)

Optional wrapper
----------------
This file includes an argparse-based __main__ wrapper so you can run it via:
    runfile(".../NEBULA_PIPELINE_MANAGER.py", args="--dry-run --recompute-from FLUX")
but Mode A is the intended primary workflow.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR

from Utility.PIPELINE.NEBULA_PIPELINE_STAGES import (
    Stage,
    StageLike,
    STAGE_ORDER,
    is_valid_stage,
    resolve_invalidate_list,
    resolve_run_list,
    try_parse_stage,
)
from Utility.PIPELINE.NEBULA_PIPELINE_MANIFEST import (
    StageSpec,
    get_stage_spec,
    load_stage_callable,
    resolve_owned_output_paths_for_stages,
)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineRunPlan:
    """Fully resolved plan for a pipeline run (used for dry-run and execution)."""

    output_dir: Path
    recompute_from_stage: Optional[str]
    stop_after_stage: Optional[str]

    run_stages: Tuple[Stage, ...]
    invalidate_stages: Tuple[Stage, ...]
    delete_paths: Tuple[Path, ...]

    # Convenience strings for reporting
    stage_entrypoints: Tuple[str, ...]


@dataclass
class PipelineRunResult:
    """Results/telemetry from a pipeline run."""

    plan: PipelineRunPlan
    deleted_paths: List[Path]
    executed_stages: List[Stage]
    stage_return_values: Dict[Stage, Any]


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
def _get_default_logger() -> logging.Logger:
    """
    Create (or reuse) a default logger for the pipeline manager.

    We intentionally keep this small and deterministic to avoid duplicate
    handlers in iterative Spyder sessions.
    """
    logger = logging.getLogger("NEBULA_PIPELINE")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate prints via root logger handlers (common in Spyder/Jupyter).
    logger.propagate = False

    # Add a single stream handler if none exist.
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# -----------------------------------------------------------------------------
# Safety helpers
# -----------------------------------------------------------------------------
def _resolve_base_dir(output_dir: Optional[Path]) -> Path:
    """Return the base output directory (absolute)."""
    base = output_dir if output_dir is not None else NEBULA_OUTPUT_DIR
    return Path(base).resolve()


def _is_within_dir(path: Path, base_dir: Path) -> bool:
    """
    True if `path` is within `base_dir` (lexically), guarding against accidental
    deletion outside NEBULA_OUTPUT_DIR.
    """
    try:
        # PurePath.is_relative_to exists in Python 3.9+.
        return path.resolve().is_relative_to(base_dir.resolve())  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback for older Pythons (not expected in your environment).
        p = path.resolve()
        b = base_dir.resolve()
        return b == p or b in p.parents


def _safe_unlink(path: Path, base_dir: Path, logger: logging.Logger) -> bool:
    """
    Delete a file if it exists and is safely within base_dir.

    Returns
    -------
    bool
        True if a file was deleted, False otherwise.
    """
    p = path.resolve()

    if not _is_within_dir(p, base_dir):
        raise RuntimeError(
            f"Refusing to delete path outside output_dir.\n"
            f"  output_dir: {base_dir}\n"
            f"  candidate : {p}"
        )

    if not p.exists():
        return False

    if p.is_dir():
        # We never delete directories in this manager (by design).
        raise RuntimeError(f"Refusing to delete directory (files-only policy): {p}")

    logger.info(f"Deleting owned output: {p}")
    p.unlink()
    return True


def _format_entrypoint(spec: StageSpec) -> str:
    """Human-readable entrypoint string for dry-run reporting."""
    ep = spec.entrypoint
    return f"{spec.stage.value}: {ep.module}.{ep.function}(...)  # {spec.description}"

def _stage_outputs_exist(stage: Stage, *, output_dir: Path) -> bool:
    """
    True if all owned output pickle files for `stage` exist on disk.
    """
    paths = resolve_owned_output_paths_for_stages([stage], output_dir=output_dir)
    return all(p.exists() for p in paths)


def _first_missing_stage(
    stages_in_order: Sequence[Stage],
    *,
    output_dir: Path,
) -> Optional[Stage]:
    """
    Return the first stage (in the provided order) whose owned outputs are missing.
    """
    for st in stages_in_order:
        if not _stage_outputs_exist(st, output_dir=output_dir):
            return st
    return None

# -----------------------------------------------------------------------------
# Plan resolution
# -----------------------------------------------------------------------------
def build_plan(
    *,
    recompute_from_stage: Optional[StageLike] = None,
    stop_after_stage: Optional[StageLike] = None,
    output_dir: Optional[Path] = None,
    invalidate_beyond_stop_after: bool = False,
) -> PipelineRunPlan:
    """
    Resolve a concrete PipelineRunPlan from user knobs.

    Parameters
    ----------
    recompute_from_stage : StageLike | None
        If valid, invalidate (delete owned outputs) from this stage onward.
        If invalid/None, invalidate nothing.
    stop_after_stage : StageLike | None
        If valid, only run through this stage (inclusive).
        If invalid/None, run full chain.
    output_dir : Path | None
        If provided, override NEBULA_OUTPUT_DIR.
    invalidate_beyond_stop_after : bool
        Safety switch:
        - False (default): only delete outputs for stages that will be run in
          this invocation (prevents “delete but not rebuild” when stop_after is
          earlier than recompute_from).
        - True: delete from recompute_from onward regardless of stop_after.

    Returns
    -------
    PipelineRunPlan
    """
    base_dir = _resolve_base_dir(output_dir)

    # ------------------------------------------------------------------
    # 1) Required chain is still governed by STOP_AFTER semantics:
    #    BASE → ... → stop_after (inclusive), or full chain if stop_after invalid/None
    # ------------------------------------------------------------------
    required_chain = resolve_run_list(stop_after_stage)

    # ------------------------------------------------------------------
    # 2) New policy: run starts at RECOMPUTE_FROM if valid; otherwise BASE.
    #    Then we verify prerequisites prior to that start stage.
    # ------------------------------------------------------------------
    if is_valid_stage(recompute_from_stage):
        requested_start = try_parse_stage(recompute_from_stage)  # not None due to is_valid_stage
        assert requested_start is not None
    else:
        requested_start = required_chain[0]  # typically BASE

    # If stop_after cuts the chain before requested_start, this is an invalid request.
    if requested_start not in required_chain:
        raise ValueError(
            "Invalid stage combination: recompute_from_stage is downstream of stop_after_stage.\n"
            f"  recompute_from_stage: {recompute_from_stage}\n"
            f"  stop_after_stage    : {stop_after_stage}\n"
            f"  required_chain      : {[s.value for s in required_chain]}"
        )

    requested_idx = required_chain.index(requested_start)
    prereq_stages = required_chain[:requested_idx]  # stages strictly prior to requested_start

    # Auto-fix: if any prereq stage outputs are missing, expand run start earlier.
    first_missing = _first_missing_stage(prereq_stages, output_dir=base_dir)
    if first_missing is not None:
        actual_start = first_missing
    else:
        actual_start = requested_start

    actual_idx = required_chain.index(actual_start)
    run_list = required_chain[actual_idx:]  # run only from actual_start to stop_after

    # 3) Invalidate list (deletion-driven)
    #
    # If prerequisites were missing and we expanded the run start earlier, we must
    # also expand invalidation to that earlier stage, otherwise downstream stages
    # may "reuse existing" outputs that depend on now-rebuilt upstream products.
    effective_invalidate_from = recompute_from_stage
    if first_missing is not None:
        # Expand invalidation to the effective start stage we must rebuild from.
        effective_invalidate_from = actual_start

    
    invalidate_list_full = resolve_invalidate_list(effective_invalidate_from)

    if invalidate_beyond_stop_after:
        invalidate_list = invalidate_list_full
    else:
        run_set = set(run_list)
        invalidate_list = tuple(s for s in invalidate_list_full if s in run_set)

    # Resolve concrete delete paths (owned outputs only).
    delete_paths = tuple(
        resolve_owned_output_paths_for_stages(invalidate_list, output_dir=base_dir)
    )

    # Build entrypoint strings for the stages we will execute (run_list only).
    entrypoints: List[str] = []
    for st in run_list:
        spec = get_stage_spec(st)
        entrypoints.append(_format_entrypoint(spec))


    # Preserve original strings for reporting (canonicalized if valid).
    rec_s = (
        try_parse_stage(recompute_from_stage).value
        if is_valid_stage(recompute_from_stage)
        else (str(recompute_from_stage) if recompute_from_stage is not None else None)
    )
    stop_s = (
        try_parse_stage(stop_after_stage).value
        if is_valid_stage(stop_after_stage)
        else (str(stop_after_stage) if stop_after_stage is not None else None)
    )

    return PipelineRunPlan(
        output_dir=base_dir,
        recompute_from_stage=rec_s,
        stop_after_stage=stop_s,
        run_stages=tuple(run_list),
        invalidate_stages=tuple(invalidate_list),
        delete_paths=delete_paths,
        stage_entrypoints=tuple(entrypoints),
    )


def print_plan(plan: PipelineRunPlan, *, logger: Optional[logging.Logger] = None) -> None:
    """
    Print a dry-run plan (or general plan) in a human-readable format.
    """
    log = logger or _get_default_logger()

    # Header
    log.info("NEBULA pipeline plan")
    log.info(f"  output_dir           : {plan.output_dir}")
    log.info(f"  recompute_from_stage : {plan.recompute_from_stage}")
    log.info(f"  stop_after_stage     : {plan.stop_after_stage}")

    # Stage lists
    run_ids = [s.value for s in plan.run_stages]
    inv_ids = [s.value for s in plan.invalidate_stages]
    log.info(f"  RUN stages           : {run_ids}")
    log.info(f"  INVALIDATE stages    : {inv_ids}")

    # One-line prereq sentence (clean policy)
    rec = try_parse_stage(plan.recompute_from_stage) if plan.recompute_from_stage else None
    if rec is None:
        log.info("  PREREQS              : recompute_from_stage is None/invalid → no prereq gate; proceeding.")
    else:
        # Stages prior to recompute_from in canonical order
        rec_idx = STAGE_ORDER.index(rec)
        prior = STAGE_ORDER[:rec_idx]

        missing_prior = []
        for st in prior:
            if not _stage_outputs_exist(st, output_dir=plan.output_dir):
                missing_prior.append(st.value)

        if not missing_prior:
            log.info(f"  PREREQS              : All pickles prior to {rec.value} exist; proceeding.")
        else:
            expanded_to = plan.run_stages[0].value if plan.run_stages else "(none)"
            log.info(
                f"  PREREQS              : Missing prior-stage outputs {missing_prior} (prior to {rec.value}); "
                f"expanding run start to {expanded_to}."
            )

    # Delete list with exists status
    log.info("  DELETE owned outputs :")
    if not plan.delete_paths:
        log.info("    (none)")
    else:
        for p in plan.delete_paths:
            exists = p.exists()
            log.info(f"    - {p}  (exists={exists})")

    # Entrypoints
    log.info("  Entrypoints to call  :")
    for s in plan.stage_entrypoints:
        log.info(f"    - {s}")


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
# def _filtered_kwargs_for_callable(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Filter kwargs to only those accepted by the callable's signature.
#     This makes the manager robust to slight signature differences across picklers.
#     """
#     sig = inspect.signature(fn)
#     accepted = set(sig.parameters.keys())
#     return {k: v for k, v in kwargs.items() if k in accepted}
def _filtered_kwargs_for_callable(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs to only those accepted by the callable's signature.
    This makes the manager robust to slight signature differences across picklers.
    """
    sig = inspect.signature(fn)

    # If the callable accepts **kwargs, pass through everything.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _execute_stage(
    stage: Stage,
    *,
    logger: logging.Logger,
    force_recompute: bool,
) -> Any:
    """
    Execute a single stage by:
      - loading its callable lazily,
      - filtering standard kwargs (logger, force_recompute) to match its signature,
      - calling it.
    """
    spec = get_stage_spec(stage)
    fn = load_stage_callable(stage)

    # Standard kwargs we *prefer* to pass; filtered to match signature.
    kwargs: Dict[str, Any] = {"logger": logger, "force_recompute": force_recompute}

    # Add any fixed extra kwargs declared in the manifest.
    # (We keep this here so the manifest can evolve without touching the manager.)
    if spec.extra_kwargs:
        kwargs.update(dict(spec.extra_kwargs))

    filtered = _filtered_kwargs_for_callable(fn, kwargs)

    logger.info(f"Executing stage {stage.value}: {spec.entrypoint.module}.{spec.entrypoint.function}")
    return fn(**filtered)


def run_pipeline(
    *,
    recompute_from_stage: Optional[StageLike] = None,
    stop_after_stage: Optional[StageLike] = None,
    dry_run: bool = True,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    invalidate_beyond_stop_after: bool = False,
    # We keep this parameter for debugging parity with picklers, but the manager’s
    # intended default is deletion-driven recompute. Usually leave this False.
    force_recompute_on_entrypoints: bool = False,
) -> PipelineRunResult:
    """
    Run the NEBULA upstream pipeline through ZODIACAL_LIGHT.

    Parameters
    ----------
    recompute_from_stage : StageLike | None
        If valid, invalidate from this stage onward (delete owned outputs).
    stop_after_stage : StageLike | None
        If valid, stop after this stage (inclusive). If invalid/None, run full chain.
    dry_run : bool
        If True, prints the plan and returns without deleting or executing.
        Default is True for safety during initial integration.
    output_dir : Path | None
        Override NEBULA_OUTPUT_DIR (rare; mostly for tests).
    logger : logging.Logger | None
        If None, a default console logger is created.
    invalidate_beyond_stop_after : bool
        If True, deletes downstream outputs even if stop_after is earlier.
        Default False (safe).
    force_recompute_on_entrypoints : bool
        If True, passes force_recompute=True to stage entrypoints.
        Usually unnecessary because deletion controls recompute.

    Returns
    -------
    PipelineRunResult
        Contains the resolved plan, deleted paths, executed stages, and stage return values.
    """
    log = logger or _get_default_logger()

    # Build and print the plan.
    plan = build_plan(
        recompute_from_stage=recompute_from_stage,
        stop_after_stage=stop_after_stage,
        output_dir=output_dir,
        invalidate_beyond_stop_after=invalidate_beyond_stop_after,
    )
    print_plan(plan, logger=log)

    result = PipelineRunResult(
        plan=plan,
        deleted_paths=[],
        executed_stages=[],
        stage_return_values={},
    )

    if dry_run:
        log.info("DRY_RUN=True -> no deletions, no stage execution.")
        return result

    # 1) Delete owned outputs for invalidated stages (files only).
    base_dir = plan.output_dir
    for p in plan.delete_paths:
        deleted = _safe_unlink(p, base_dir=base_dir, logger=log)
        if deleted:
            result.deleted_paths.append(p)

    # 2) Execute stages in order (strict chain / stop_after respected).
    for st in plan.run_stages:
        retval = _execute_stage(
            st,
            logger=log,
            force_recompute=bool(force_recompute_on_entrypoints),
        )
        result.executed_stages.append(st)
        result.stage_return_values[st] = retval

    log.info("Pipeline run complete.")
    return result


# -----------------------------------------------------------------------------
# Optional CLI wrapper (kept small; Mode A is primary)
# -----------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    stages = [s.value for s in STAGE_ORDER]

    p = argparse.ArgumentParser(
        description="NEBULA upstream pipeline manager (through ZODIACAL_LIGHT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--recompute-from",
        dest="recompute_from",
        default=None,
        help=f"Stage ID to invalidate from (owned outputs deleted). Choices: {stages}",
    )
    p.add_argument(
        "--stop-after",
        dest="stop_after",
        default=None,
        help=f"Stage ID to stop after (inclusive). Choices: {stages}",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print plan only (no deletions, no execution).",
    )
    p.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Actually delete + run stages (overrides default dry-run behavior).",
    )
    p.add_argument(
        "--invalidate-beyond-stop-after",
        dest="invalidate_beyond_stop_after",
        action="store_true",
        help="Delete invalidated outputs even if stop_after is earlier (advanced).",
    )
    p.add_argument(
        "--force-recompute-on-entrypoints",
        dest="force_recompute_on_entrypoints",
        action="store_true",
        help="Also pass force_recompute=True to stage entrypoints (debug).",
    )
    return p


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Default behavior: dry-run unless --execute is provided.
    dry_run = True
    if args.execute:
        dry_run = False
    if args.dry_run:
        dry_run = True

    run_pipeline(
        recompute_from_stage=args.recompute_from,
        stop_after_stage=args.stop_after,
        dry_run=dry_run,
        invalidate_beyond_stop_after=bool(args.invalidate_beyond_stop_after),
        force_recompute_on_entrypoints=bool(args.force_recompute_on_entrypoints),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
