import numpy as np
import h5py
import os
from datetime import timedelta
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Parameters (edit as needed)
photon_flux_ranges = [1] #np.logspace(1,20,100)  # Example photon flux values (photons/s)
array_shape = (1,3)  # Size of the detector array
output_dir = "occultation_test/occultation_hdf5_2"

# Occulting object parameters
object_size_percent_ranges = np.linspace(.1,1.45,145) #np.linspace(1,1.45,45) out to 1.45 to cover full pixel np.linspace(.1,1,100)  # Fraction of pixel area (0.0-1.0)
traverse_speed_ranges = np.logspace(-4,-3,30)  # np.unique(np.append(np.logspace(-4,-3,30),np.logspace(-3,0,100))) Pixels per frame

# Camera timing parameters
fps = 500  # Frames per second
dt = 1.0 / fps  # Time step in seconds
start_time = 0.0  # Start time in seconds

def create_occultation_mask(array_shape, center, radius, subsample=10):
    """Create a high-res circular mask for the occulting object, then downsample to detector pixels."""
    high_shape = (array_shape[0]*subsample, array_shape[1]*subsample)
    Y, X = np.ogrid[:high_shape[0], :high_shape[1]]
    # Scale center and radius to high-res grid
    # high_center = (center[0]*subsample + subsample//2, center[1]*subsample + subsample//2)
    # high_radius = radius * subsample
    dist = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    mask = dist <= radius
    # Invert mask: 1=blocked, 0=open
    inv_mask = (~mask).astype(float)
    # Downsample by averaging 10x10 blocks
    downsampled = inv_mask.reshape(array_shape[0], subsample, array_shape[1], subsample).mean(axis=(1,3))
    # downsampled is the fraction of each detector pixel that is unblocked (transmittance)
    return inv_mask, downsampled

def simulate_occultation(photon_flux, array_shape, object_size_percent, traverse_speed):
    """Simulate a traversing occulting object over a photon flux array using high-res mask and downsampling."""
    # highres_frames = []
    # lowres_frames = []
    highres_masks = []
    lowres_masks = []
    # Base subsampling by the traversing speed
    subsample = int(np.ceil(1 / traverse_speed))
    # Center of the occulting object in the high-res grid
    min_dim = min(array_shape)*subsample
    radius = (min_dim * object_size_percent) / 2
    center_row = array_shape[0]*subsample // 2
    start_col = int(np.ceil(radius))
    end_col = array_shape[1]*subsample - int(np.ceil(radius))
    t = 0
    for center_col in range(start_col, end_col + 1, 1):
        # Downsampled mask: fraction of each detector pixel that is unblocked (transmittance)
        highres_mask, lowres_mask = create_occultation_mask(array_shape, (center_row, center_col), radius, subsample=subsample)
        # blocked_fraction = 1 - trans_mask
        # For each pixel, photon flux is reduced by the tranmission mask
        # frame = photon_flux * (lowres_mask)
        # highres_frame = photon_flux * (highres_mask)
        # lowres_frames.append(frame.astype(np.float32))
        # highres_frames.append(highres_frame.astype(np.float32))
        lowres_masks.append(lowres_mask.astype(np.float32))
        highres_masks.append(highres_mask.astype(np.float32))
        # Increment time step
        t += 1
    # lowres_frames = np.stack(lowres_frames, axis=0)
    # highres_frames = np.stack(highres_frames, axis=0)
    lowres_masks = np.stack(lowres_masks, axis=0)
    highres_masks = np.stack(highres_masks, axis=0)
    return lowres_masks, highres_masks

def save_hdf5_with_timestamps(lowres_masks, highres_masks, photon_flux, object_size_percent, traverse_speed, output_dir, dt, start_time):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"photon_flux_{int(photon_flux)}_objectsize_{object_size_percent:.2f}_traversespeed_{traverse_speed:.3f}.h5")
    with h5py.File(filename, "w") as f:
        # # Save low-res frames
        # lowres_grp = f.create_group("lowres")
        # for i, frame in enumerate(lowres_frames):
        #     timestamp = start_time + i * dt
        #     dset_name = f"frame_{timestamp:.9f}"
        #     lowres_grp.create_dataset(dset_name, data=frame)
        # # Save high-res frames
        # highres_grp = f.create_group("highres")
        # for i, frame in enumerate(highres_frames):
        #     timestamp = start_time + i * dt
        #     dset_name = f"frame_{timestamp:.9f}"
        #     highres_grp.create_dataset(dset_name, data=frame)
        # Save low-res frames
        lowres_grp_2 = f.create_group("lowresmask")
        for i, frame in enumerate(lowres_masks):
            timestamp = start_time + i * dt
            dset_name = f"frame_{timestamp:.9f}"
            lowres_grp_2.create_dataset(dset_name, data=frame, compression="gzip", compression_opts=9)
        # Save high-res frames
        highres_grp_2 = f.create_group("highresmask")
        for i, frame in enumerate(highres_masks):
            timestamp = start_time + i * dt
            dset_name = f"frame_{timestamp:.9f}"
            highres_grp_2.create_dataset(dset_name, data=frame, compression="gzip", compression_opts=9)
        # f.attrs["photon_flux"] = photon_flux
        f.attrs["object_size_percent"] = object_size_percent
        f.attrs["fps"] = 1.0 / dt
        f.attrs["traverse_speed_pixels_per_frame"] = traverse_speed
        f.attrs["description"] = "Simulated transmission frames for occultation observations. Each dataset is a frame with a timestamp in seconds. Contains both lowres and highres outputs."
    f.close()
    print(f"Saved: {filename}")

def main():
    for photon_flux in photon_flux_ranges:
        for object_size_percent in object_size_percent_ranges:  # Reduce to 5 steps for brevity
            for traverse_speed in traverse_speed_ranges:
                lowres_masks, highres_masks = simulate_occultation(
                    photon_flux,
                    array_shape,
                    object_size_percent,
                    traverse_speed,
                )
                save_hdf5_with_timestamps(lowres_masks, highres_masks, photon_flux, object_size_percent, traverse_speed, output_dir, dt, start_time)


def visualize_all_hdf5_as_videos(output_dir, frame_rate=30, cmap="viridis"):
    """Visualize all HDF5 files in the output directory as videos and save as MP4."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import glob

    writer_mp4 = FFMpegWriter(fps=frame_rate,metadata={'artist':'Me'},bitrate=1800)

    h5_files = sorted(glob.glob(os.path.join(output_dir, "*.h5")))
    if not h5_files:
        print("No HDF5 files found in output directory.")
        return
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            for res in ["lowresmask", "highresmask"]:
                if res not in f:
                    continue
                frame_keys = sorted(f[res].keys(), key=lambda k: float(k.split('_')[1]))
                frames = [f[res][k][:] for k in frame_keys]

                fig, ax = plt.subplots()
                im = ax.imshow(frames[0], cmap=cmap, animated=True)
                plt.colorbar(im, ax=ax,orientation='horizontal', label='Percent Tranmission [0-1]')
                ax.set_title(f"{os.path.basename(h5_file)} - {res}")

                def update(frame):
                    im.set_array(frame)
                    return [im]

                ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=1000/frame_rate)
                video_path = os.path.join(output_dir, os.path.splitext(os.path.basename(h5_file))[0] + f"_{res}.mp4")
                ani.save(video_path, writer=writer_mp4, dpi=300)
                # ani.save(video_path.replace('.mp4','.gif'), fps=30)
                plt.close(fig)
                print(f"Video saved to {video_path}")

if __name__ == "__main__":
    main()

    # Visualize all HDF5 files as videos
    # visualize_all_hdf5_as_videos(output_dir, frame_rate=30)
