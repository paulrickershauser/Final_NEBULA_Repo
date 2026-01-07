import os
import glob
import h5py
import numpy as np
import pandas as pd
from astropy import units as u
import matplotlib.pyplot as plt
import re

# output_dir = "occultation_test/output_images"
output_dir = "occultation_test/output_images_0.2_Threshold"
input_dir = "occultation_test/occultation_hdf5"
fps = 500
# run for specific thresholds
positive_threshold = 0.2
negative_threshold = 0.2
events_grp_string = "events_pthreshold_{:f}_nthreshold_{:f}".format(positive_threshold,negative_threshold)


# FIXME: use the 500 fps variable to change the traverse rate of frames per pixel to pixels per second

# 1. Plot highres and lowres masks for each file
# for h5_path in glob.glob(os.path.join(input_dir, "*.h5")):
#     with h5py.File(h5_path, "r") as f:
#         for res in ["lowresmask", "highresmask"]:
#             if res in f:
#                 frames = [f[res][k][()] for k in list(f[res].keys())]
#                 if frames:
#                     plt.figure(figsize=(4,4))
#                     plt.imshow(frames[0], cmap="gray", vmin=0, vmax=1)
#                     plt.title(f"{os.path.basename(h5_path)} - {res}")
#                     plt.axis("off")
#                     plt.tight_layout()
#                     plt.savefig(os.path.join(output_dir, f"{os.path.basename(h5_path)}_{res}.pdf"), dpi=300)
#                     plt.close()

# # 2. Center pixel mask over time, varying speed (fixed object size)
# unique_sizes = sorted({h5_path.split("objectsize_")[1].split("_")[0] for h5_path in glob.glob(os.path.join(input_dir, "*objectsize_*.h5"))}, key=float)
# for fixed_size in unique_sizes:
#     center_pixel_traces = []
#     speeds = []
#     max_len = 0
#     for h5_path in sorted(glob.glob(os.path.join(input_dir, f"*objectsize_{fixed_size}_traversespeed_*.h5"))):
#         speed = float(h5_path.split("traversespeed_")[1].replace(".h5", ""))
#         with h5py.File(h5_path, "r") as f:
#             frames = [f["lowresmask"][k][()] for k in list(f["lowresmask"].keys())]
#             arr = np.stack(frames)
#             center = tuple(s//2 for s in arr.shape[1:])
#             trace = arr[:, center[0], center[1]]
#             max_len = max(max_len, len(trace))
#             center_pixel_traces.append(trace)
#             speeds.append(speed)
#     # Pad traces with 1s to the max length
#     center_pixel_traces_padded = []
#     for trace in center_pixel_traces:
#         pad_len = max_len - len(trace)
#         left = pad_len // 2
#         right = pad_len - left
#         padded = np.pad(trace, (left, right), constant_values=1)
#         center_pixel_traces_padded.append(padded)
#     center_pixel_traces_padded = np.array(center_pixel_traces_padded)
#     # Trim to the first and last columns where any value differs from 1
#     mask = (center_pixel_traces_padded != 1)
#     if np.any(mask):
#         col_any = mask.any(axis=0)
#         first = np.argmax(col_any)
#         last = len(col_any) - np.argmax(col_any[::-1])
#         center_pixel_traces_trimmed = center_pixel_traces_padded[:, first:last]
#     else:
#         center_pixel_traces_trimmed = center_pixel_traces_padded
#     trimmed_len = center_pixel_traces_trimmed.shape[1]
#     # Set time axis so that 0 is in the middle
#     half_time = trimmed_len / (2*fps)
#     time_axis = np.linspace(-half_time, half_time, trimmed_len)
#     # Plot as imshow
#     plt.figure(figsize=(10,6))
#     plt.imshow(center_pixel_traces_trimmed, aspect='auto', cmap="gray", vmin=0, vmax=1, extent=[time_axis[0], time_axis[-1], min(speeds), max(speeds)])
#     plt.xlabel("Time (s)")
#     plt.ylabel("Traverse Speed")
#     plt.title(f"Center Pixel Mask Over Time (Object Size {fixed_size})")
#     plt.colorbar(label="Transmission")
#     plt.savefig(os.path.join(output_dir, f"center_pixel_vs_time_vary_speed_objectsize_{fixed_size}_pthresh_{positive_threshold}_nthresh_{negative_threshold}_imshow.pdf"), dpi=300)
#     plt.close()
#     # Line plot: one line per speed
#     plt.figure(figsize=(10,6))
#     for trace, speed in zip(center_pixel_traces_trimmed, speeds):
#         plt.plot(time_axis, trace, label=f"Speed {speed}")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Transmission")
#     plt.title(f"Center Pixel Transmission Over Time (Object Size {fixed_size})")
#     plt.legend()
#     plt.savefig(os.path.join(output_dir, f"center_pixel_vs_time_vary_speed_objectsize_{fixed_size}_pthresh_{positive_threshold}_nthresh_{negative_threshold}_lines.pdf"), dpi=300)
#     plt.close()

# # 3. Center pixel mask over time, varying object size (fixed speed)
# unique_speeds = sorted({h5_path.split("traversespeed_")[1].replace(".h5", "") for h5_path in glob.glob(os.path.join(input_dir, "*traversespeed_*.h5"))}, key=float)
# for fixed_speed in unique_speeds:
#     center_pixel_traces = []
#     sizes = []
#     max_len = 0
#     for h5_path in sorted(glob.glob(os.path.join(input_dir, f"*traversespeed_{fixed_speed}.h5"))):
#         size = float(h5_path.split("objectsize_")[1].split("_")[0])
#         with h5py.File(h5_path, "r") as f:
#             frames = [f["lowresmask"][k][()] for k in list(f["lowresmask"].keys())]
#             arr = np.stack(frames)
#             center = tuple(s//2 for s in arr.shape[1:])
#             trace = arr[:, center[0], center[1]]
#             max_len = max(max_len, len(trace))
#             center_pixel_traces.append(trace)
#             sizes.append(size)
#     # Pad traces with 1s to the max length
#     center_pixel_traces_padded = []
#     for trace in center_pixel_traces:
#         pad_len = max_len - len(trace)
#         left = pad_len // 2
#         right = pad_len - left
#         padded = np.pad(trace, (left, right), constant_values=1)
#         center_pixel_traces_padded.append(padded)
#     center_pixel_traces_padded = np.array(center_pixel_traces_padded)
#     # Trim to the first and last columns where any value differs from 1
#     mask = (center_pixel_traces_padded != 1)
#     if np.any(mask):
#         col_any = mask.any(axis=0)
#         first = np.argmax(col_any)
#         last = len(col_any) - np.argmax(col_any[::-1])
#         center_pixel_traces_trimmed = center_pixel_traces_padded[:, first:last]
#     else:
#         center_pixel_traces_trimmed = center_pixel_traces_padded
#     trimmed_len = center_pixel_traces_trimmed.shape[1]
#     # Set time axis so that 0 is in the middle
#     half_time = trimmed_len / (2*fps)
#     time_axis = np.linspace(-half_time, half_time, trimmed_len)
#     # Plot as imshow
#     plt.figure(figsize=(10,6))
#     plt.imshow(center_pixel_traces_trimmed, aspect='auto', cmap="gray", vmin=0, vmax=1, extent=[time_axis[0], time_axis[-1], min(sizes), max(sizes)])
#     plt.xlabel("Time (s)")
#     plt.ylabel("Object Size")
#     plt.title(f"Center Pixel Mask Over Time (Traverse Speed {fixed_speed})")
#     plt.colorbar(label="Transmission")
#     plt.savefig(os.path.join(output_dir, f"center_pixel_vs_time_vary_objectsize_{fixed_speed}_pthresh_{positive_threshold}_nthresh_{negative_threshold}_imshow.pdf"), dpi=300)
#     plt.close()
#     # Line plot: one line per object size
#     plt.figure(figsize=(10,6))
#     for trace, size in zip(center_pixel_traces_trimmed, sizes):
#         plt.plot(time_axis, trace, label=f"Size {size}")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Transmission")
#     plt.title(f"Center Pixel Transmission Over Time (Traverse Speed {fixed_speed})")
#     plt.legend()
#     plt.savefig(os.path.join(output_dir, f"center_pixel_vs_time_vary_objectsize_{fixed_speed}_pthresh_{positive_threshold}_nthresh_{negative_threshold}_lines.pdf"), dpi=300)
#     plt.close()

# 4. Contour plots: total events vs. object size, traversal speed, and current (photon flux)

event_totals = pd.DataFrame(columns=["flux", "size", "speed", "event_totals", "event_rate"])
# Loop through the datasets in the 'events' group to get flux
for h5_path in glob.glob(os.path.join(input_dir, "*.h5")):
    with h5py.File(h5_path, "r") as f:
        size = float(h5_path.split("objectsize_")[1].split("_")[0])
        speed = float(h5_path.split("traversespeed_")[1].replace(".h5", ""))
        if events_grp_string in f:
            for dset in f[events_grp_string]:
                flux = float(dset.split('_')[-1].replace('current_', '').replace('neg', '-').replace('pos', '+').replace('dot', '.'))
                # Store total events for this file and 
                df = pd.read_hdf(h5_path, events_grp_string + '/' + dset)
                n_events = len(df)
                if n_events > 1:
                    pass
                event_rate = n_events / len(f["lowresmask"])*500
                event_totals.loc[len(event_totals)] = [flux, size, speed, n_events, event_rate]

# Find the simulation(s) with the most events
max_events = np.max(event_totals['event_totals'])
max_events_loc = np.where(event_totals['event_totals']==max_events)
for i, location in enumerate(max_events_loc[0]):
    flux = event_totals.loc[location]["flux"]
    size = event_totals.loc[location]["size"]
    speed = event_totals.loc[location]["speed"]

    # Find the corresponding h5 file
    h5_pattern = "photon_flux_1_objectsize_{:.2f}_traversespeed_{:.3f}.h5".format(size, speed)
    h5_files = glob.glob(os.path.join(input_dir, h5_pattern))
    if not h5_files:
        continue
    h5_path = h5_files[0]

    with h5py.File(h5_path, "r") as f:
        # Load lowresmask and stack into array
        frames = [f["lowresmask"][k][()] for k in list(f["lowresmask"].keys())]
        arr = np.stack(frames)
        center = tuple(s//2 for s in arr.shape[1:])
        center_trace = arr[:, center[0], center[1]]

        # Find the correct events group/dataset
        dset_name = "current_{:e}".format(flux).replace('-','neg').replace('.','dot').replace('+','pos')

        # Load events dataframe
        df = pd.read_hdf(h5_path, events_grp_string + '/' + dset_name)
        # Plot
        plt.figure(figsize=(10, 4))
        extent = [0, len(center_trace)/fps, 0, 1]
        plt.imshow(center_trace[None, :], aspect='auto', cmap='gray', vmin=0, vmax=1, extent=extent)
        # Overlay events
        if not df.empty and "t" in df and "p" in df:
            t = df["t"]
            p = df["p"]
            plt.scatter(t[p == 1.0], [0.5]*np.sum(p == 1.0), color='blue', label='p=1', marker='|', s=100)
            plt.scatter(t[p == -1.0], [0.5]*np.sum(p == -1.0), color='red', label='p=-1', marker='|', s=100)
        plt.xlabel("Time (s)")
        plt.yticks([])
        plt.title(f"Center Pixel Transmission & Events\nsize={size}, speed={speed}, current={flux}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"center_pixel_trace_events_size_{size}_speed_{speed}_current_{flux}.pdf"), dpi=300)
        plt.close()

        # Alternative depiction: line graph for transmission, vertical bars for "p" values
        plt.figure(figsize=(10, 4))
        time = np.arange(len(center_trace)) / fps
        plt.plot(time, center_trace, color='black', label='Transmission')
        if not df.empty and "t" in df and "p" in df:
            for polarity, color in zip([1.0, -1.0], ['blue', 'red']):
                tvals = df["t"][df["p"] == polarity]
                plt.vlines(tvals, ymin=0, ymax=1, color=color, alpha=0.7, linewidth=2, label=f'p={int(polarity)}')
        plt.xlabel("Time (s)")
        plt.ylabel("Transmission")
        plt.title(f"Center Pixel Transmission & Events \nsize={size}, speed={speed}, current={flux}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"center_pixel_trace_events_linebar_size_{size}_speed_{speed}_current_{flux}.pdf"), dpi=300)
        plt.close()


if not event_totals.empty:
    # 4a. Contour and imshow: object size vs. speed for a fixed current
    unique_flux = np.sort(event_totals['flux'].unique())
    for flux in unique_flux:
        df = event_totals[event_totals['flux'] == flux]
        sizes = np.sort(df['size'].unique())
        speeds = np.sort(df['speed'].unique())
        grid_x, grid_y = np.meshgrid(sizes, speeds, indexing='ij')
        grid_z = np.full(grid_x.shape, np.nan)
        for i, s in enumerate(sizes):
            for j, v in enumerate(speeds):
                match = df[(df['size'] == s) & (df['speed'] == v)]
                if not match.empty:
                    grid_z[i, j] = match['event_totals'].values[0]
        # Contour plot
        plt.figure(figsize=(8,6))
        cp = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap="viridis")
        plt.xlabel("Object Size")
        plt.ylabel("Traverse Speed")
        plt.title(f"Total Events (Current {flux:.2e} A)")
        cbar = plt.colorbar(cp, label="Total Events")
        plt.savefig(os.path.join(output_dir, f"total_events_vs_size_speed_current_{flux:.2e}.pdf"), dpi=300)
        plt.close()
        # Imshow plot (y-axis flipped)
        plt.figure(figsize=(8,6))
        plt.imshow(np.flipud(grid_z.T), aspect='auto', cmap="viridis", extent=[sizes[0], sizes[-1], speeds[0], speeds[-1]])
        plt.xlabel("Object Size")
        plt.ylabel("Traverse Speed")
        plt.title(f"Total Events (Current {flux:.2e} A)")
        plt.colorbar(label="Total Events")
        plt.savefig(os.path.join(output_dir, f"total_events_vs_size_speed_current_{flux:.2e}_imshow.pdf"), dpi=300)
        plt.close()
        
    # 4b. Contour and imshow: object size vs. current for a fixed speed
    unique_speed = np.sort(event_totals['speed'].unique())
    for speed in unique_speed:
        df = event_totals[event_totals['speed'] == speed]
        sizes = np.sort(df['size'].unique())
        fluxes = np.sort(df['flux'].unique())
        grid_x, grid_y = np.meshgrid(sizes, fluxes, indexing='ij')
        grid_z = np.full(grid_x.shape, np.nan)
        for i, s in enumerate(sizes):
            for j, fval in enumerate(fluxes):
                match = df[(df['size'] == s) & (df['flux'] == fval)]
                if not match.empty:
                    grid_z[i, j] = match['event_totals'].values[0]
        # Contour plot
        plt.figure(figsize=(8,6))
        cp = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap="viridis")
        plt.xlabel("Object Size")
        plt.ylabel("Current (A)")
        plt.title(f"Total Events (Traverse Speed {speed})")
        cbar = plt.colorbar(cp, label="Total Events")
        plt.gca().set_yticklabels([f"{tick:.2e}" for tick in plt.gca().get_yticks()])
        plt.savefig(os.path.join(output_dir, f"total_events_vs_size_current_speed_{speed}.pdf"), dpi=300)
        plt.close()
        # Imshow plot (y-axis flipped)
        fig, ax = plt.subplots(figsize=(8,6))
        plt.imshow(np.flipud(grid_z.T), aspect='auto', cmap="viridis", extent=[sizes[0], sizes[-1], 0, 20])
        plt.xlabel("Object Size")
        plt.ylabel("Current (A)")
        plt.title(f"Total Events (Traverse Speed {speed})")
        plt.colorbar(label="Total Events")
        ax.set_yticks(np.linspace(0,19,20),[f"{tick:.2e}" for tick in fluxes])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"total_events_vs_size_current_speed_{speed}_imshow.pdf"), dpi=300)
        plt.close()
    # 4c. Contour and imshow: speed vs. current for a fixed object size
    unique_size = np.sort(event_totals['size'].unique())
    for size in unique_size:
        df = event_totals[event_totals['size'] == size]
        speeds = np.sort(df['speed'].unique())
        fluxes = np.sort(df['flux'].unique())
        grid_x, grid_y = np.meshgrid(speeds, fluxes, indexing='ij')
        grid_z = np.full(grid_x.shape, np.nan)
        for i, s in enumerate(speeds):
            for j, fval in enumerate(fluxes):
                match = df[(df['speed'] == s) & (df['flux'] == fval)]
                if not match.empty:
                    grid_z[i, j] = match['event_totals'].values[0]
        # Contour plot
        plt.figure(figsize=(8,6))
        cp = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap="viridis")
        plt.xlabel("Traverse Speed")
        plt.ylabel("Current (A)")
        plt.title(f"Total Events (Object Size {size})")
        cbar = plt.colorbar(cp, label="Total Events")
        plt.gca().set_yticklabels([f"{tick:.2e}" for tick in plt.gca().get_yticks()])
        plt.savefig(os.path.join(output_dir, f"total_events_vs_speed_current_size_{size}.pdf"), dpi=300)
        plt.close()
        # Imshow plot (y-axis flipped)
        fig, ax = plt.subplots(figsize=(8,6))
        plt.imshow(np.flipud(grid_z.T), aspect='auto', cmap="viridis", extent=[sizes[0], sizes[-1], 0, 20])
        plt.xlabel("Traverse Speed")
        plt.ylabel("Current (A)")
        plt.title(f"Total Events (Object Size {size})")
        plt.colorbar(label="Total Events")
        ax.set_yticks(np.linspace(0,19,20),[f"{tick:.2e}" for tick in fluxes])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"total_events_vs_speed_current_size_{size}_imshow.pdf"), dpi=300)
        plt.close()