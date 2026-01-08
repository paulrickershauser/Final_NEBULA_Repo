"""NEBULA_EBS_SDA_SIM_SMOKETEST

Step 0 (integration smoke test):
- Treat Rachel Oliver's EBS_SDA_SIM code as an imported library (no edits).
- Validate we can call circuitry.ebCircuitSim().generate_events(...) with:
  * a minimal observer adapter (num_x_pix, num_y_pix)
  * a minimal time-array adapter (t_array.utc shaped 6 x N)
  * a minimal HDF5 frames file with datasets named "ph_flux_time_itr_########"
  * a minimal attribution dictionary pickle (keys are time offsets in seconds, as astropy Quantity)

This script intentionally does NOT do:
- Radiometry conversions (e.g., ph/m^2/s -> ph/s/pixel)
- Optics / PSF / distortions
- NEBULA pipeline-stage wiring

Those will be Steps 1-4 in the bridge plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import os
import sys
import tempfile

import h5py
import numpy as np
from astropy import units as u


# -----------------------------------------------------------------------------
# Library import helpers (no edits to Rachel's code)
# -----------------------------------------------------------------------------

def _add_to_syspath(path: Path) -> None:
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def import_ebs_sda_sim(ebs_repo_root: Optional[Path] = None) -> Tuple[object, object]:
    """Import Rachel Oliver's EBS_SDA_SIM modules (circuitry + circuit_params).

    Expected on-disk layout (matches your screenshot):
        <NEBULA_ROOT>/EBS_SDA_SIM-main/EBS_SDA_SIM/circuitry.py
        <NEBULA_ROOT>/EBS_SDA_SIM-main/EBS_SDA_SIM/circuit_params.py

    Parameters
    ----------
    ebs_repo_root:
        Path to the folder that contains the EBS_SDA_SIM package directory.
        Example: Path(r"C:/.../NEBULA/EBS_SDA_SIM-main")

    Returns
    -------
    (circuitry_module, circuit_params_module)
    """

    if ebs_repo_root is None:
        # Try to infer:
        # - If you place this file at NEBULA/Utility/EVENT_CAMERA/., then
        #   parents[2] == NEBULA root.
        this_file = Path(__file__).resolve()
        nebula_root_guess = this_file.parents[2]
        ebs_repo_root = nebula_root_guess / "EBS_SDA_SIM-main"

    ebs_repo_root = ebs_repo_root.resolve()
    if not ebs_repo_root.exists():
        raise FileNotFoundError(
            f"EBS_SDA_SIM repo root not found: {ebs_repo_root}\n"
            "Pass ebs_repo_root=Path('.../EBS_SDA_SIM-main') explicitly."
        )

    # Add the repo root so `import EBS_SDA_SIM.<module>` works.
    _add_to_syspath(ebs_repo_root)

    # Primary (package) import
    try:
        from EBS_SDA_SIM import circuitry as circuitry_mod  # type: ignore
        from EBS_SDA_SIM import circuit_params as circuit_params_mod  # type: ignore
        return circuitry_mod, circuit_params_mod
    except Exception:
        # Fallback: if the .py files are directly on sys.path
        import circuitry as circuitry_mod  # type: ignore
        import circuit_params as circuit_params_mod  # type: ignore
        return circuitry_mod, circuit_params_mod


# -----------------------------------------------------------------------------
# Minimal adapters required by circuitry.ebCircuitSim.generate_events
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ObserverAdapter:
    """Minimum observer API used by Rachel's circuitry model."""

    num_x_pix: int
    num_y_pix: int


class TimeArrayAdapter:
    """Minimum time-array API used by Rachel's circuitry model.

    Rachel's generate_events expects an object with attribute:
        t_array.utc : np.ndarray shape (6, N)
    with rows corresponding to [year, month, day, hour, minute, second].

    For this smoke test, we use *relative time only* and store elapsed seconds
    in the "second" row; the other rows remain 0.
    """

    def __init__(self, dt_s: float, n_frames: int):
        if n_frames < 2:
            raise ValueError(
                "n_frames must be >= 2 (Rachel's generate_events reads t_array.utc[:, 1])."
            )
        self.utc = np.zeros((6, n_frames), dtype=float)
        self.utc[5, :] = np.arange(n_frames, dtype=float) * float(dt_s)


# -----------------------------------------------------------------------------
# Test-input writers (HDF5 frames + attribution pickle)
# -----------------------------------------------------------------------------


def write_step_test_frames_h5(
    h5_path: Path,
    *,
    rows: int,
    cols: int,
    n_frames: int,
    base_rate_ph_s: float = 5e7,
    high_rate_ph_s: float = 1e8,
) -> None:
    """Create a tiny frames HDF5 file Rachel's circuitry model can read.

    The circuitry model expects datasets named like:
        ph_flux_time_itr_00000000
        ph_flux_time_itr_00000001
        ...
    Each dataset should be a 2D array [rows, cols] giving photon *rate*
    (photons/second) incident on each pixel *before QE*.

    We intentionally create a simple step change at pixel (x=0, y=0):
      - frames 0..1: base
      - frames 2..3: high
      - frames 4.. : base
    so we should see ON/OFF activity without relying on noise.
    """

    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(h5_path), "w") as hf:
        for i in range(n_frames):
            frame = np.full((rows, cols), float(base_rate_ph_s), dtype=np.float64)
            if 2 <= i < 4:
                frame[0, 0] = float(high_rate_ph_s)
            hf.create_dataset(f"ph_flux_time_itr_{i:08d}", data=frame, dtype="f8")


def write_minimal_attribution_pickle(
    pkl_path: Path,
    *,
    dt_s: float,
    n_frames: int,
) -> None:
    """Write a minimal attribution dictionary expected by Rachel's arbiter.

    In circuitry.process_event(), attribution is looked up as:
        attribution_dict[curr_sim_time][(x, y)]
    where curr_sim_time is an astropy Quantity (seconds).

    For the smoke test we keep it simple:
      - provide an entry for every time step
      - only pixel (0, 0) has attribution payload (others default to "NaN")
    """

    import pickle

    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    attribution_dict = {}
    for i in range(n_frames):
        t = (i * float(dt_s)) * u.s
        attribution_dict[t] = {(0, 0): {"note": "smoketest_step_pixel"}}

    with open(str(pkl_path), "wb") as f:
        pickle.dump(attribution_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------


def run_smoketest(
    *,
    ebs_repo_root: Optional[Path] = None,
    rows: int = 20,
    cols: int = 30,
    dt_s: float = 0.5,
    n_frames: int = 6,
) -> "object":
    """Run a minimal end-to-end call into Rachel's circuitry model.

    Returns the events DataFrame object produced by generate_events.
    """

    circuitry_mod, circuit_params_mod = import_ebs_sda_sim(ebs_repo_root)

    observer = ObserverAdapter(num_x_pix=int(cols), num_y_pix=int(rows))
    t_array = TimeArrayAdapter(dt_s=float(dt_s), n_frames=int(n_frames))

    # Use nominal defaults (no interactive prompts)
    circuit_para = circuit_params_mod.circuitParameters()

    with tempfile.TemporaryDirectory(prefix="nebula_ebs_smoketest_") as td:
        td_path = Path(td)
        frames_h5 = td_path / "frames_smoketest.h5"
        att_pkl = td_path / "attrib_smoketest.pkl"

        write_step_test_frames_h5(
            frames_h5,
            rows=observer.num_y_pix,
            cols=observer.num_x_pix,
            n_frames=int(n_frames),
        )
        write_minimal_attribution_pickle(att_pkl, dt_s=float(dt_s), n_frames=int(n_frames))

        # Rachel's generate_events writes plot directories under os.getcwd();
        # keep the smoke test self-contained by running in the temp dir.
        cwd0 = os.getcwd()
        os.chdir(str(td_path))
        try:
            sim = circuitry_mod.ebCircuitSim()
            events = sim.generate_events(
                observer,
                t_array,
                str(frames_h5),
                str(att_pkl),
                "SMOKETEST",
                circuit_para,
                plot=False,
                shot_noise=False,
                high_freq_noise=False,
                junction_leak=False,
                parasitic_leak=False,
            )
        finally:
            os.chdir(cwd0)

    # Minimal reporting (caller can print full df)
    try:
        n_events = len(events)
    except Exception:
        n_events = None
    print(f"[SMOKETEST] generate_events returned type={type(events)} len={n_events}")

    return events


if __name__ == "__main__":
    # If you're running this directly, consider passing an explicit ebs_repo_root
    # if the auto-inference doesn't match your folder layout.
    run_smoketest()
