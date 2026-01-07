import os, pickle
import numpy as np

def summarize(obj, indent=0, max_list=3):
    pad = " " * indent
    if isinstance(obj, dict):
        print(f"{pad}dict: {len(obj)} keys")
        for k in list(obj.keys())[:10]:
            v = obj[k]
            print(f"{pad}  - {k!r}: {type(v).__name__}")
    elif isinstance(obj, list):
        print(f"{pad}list: {len(obj)} items")
        for i, v in enumerate(obj[:max_list]):
            print(f"{pad}  [{i}] {type(v).__name__}")
    elif isinstance(obj, np.ndarray):
        print(f"{pad}ndarray: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{pad}{type(obj).__name__}: {obj}")

def inspect_gaia_cache(pkl_path):
    with open(pkl_path, "rb") as f:
        cache = pickle.load(f)

    print("Top-level:")
    summarize(cache)

    for obs_name, obs_entry in list(cache.items())[:1]:
        print("\nObserver:", obs_name)
        summarize(obs_entry, indent=2)

        windows = obs_entry.get("windows", [])
        print(f"  windows: {len(windows)}")
        if windows:
            w0 = windows[0]
            print("  First window keys:", sorted(w0.keys()))
            for key in ["window_index","status","n_rows","t_ref_utc","t_ref_mjd_utc"]:
                if key in w0:
                    print(f"    {key}: {w0[key]}")
            for arr_key in ["gaia_source_id","ra_deg","dec_deg","mag_G"]:
                if arr_key in w0 and isinstance(w0[arr_key], np.ndarray):
                    print(f"    {arr_key}: shape={w0[arr_key].shape}, dtype={w0[arr_key].dtype}")

# Example:
inspect_gaia_cache(r"C:\Users\prick\Desktop\Research\NEBULA\NEBULA_OUTPUT\STARS\GAIA_DR3_G\obs_gaia_cones.pkl")
