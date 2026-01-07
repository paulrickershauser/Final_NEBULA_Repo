"""
nebula_3d_geometry_anim.py

High-level 3D geometry visualization for NEBULA.
Robust version: Auto-switches to GIF if FFmpeg is not installed.
"""

import logging
import sys
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
from tqdm.auto import tqdm
from skyfield.api import load

# NEBULA imports
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR
from Configuration.NEBULA_ENV_CONFIG import R_EARTH, GEO_RADIUS_KM
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER
from Utility.RADIOMETRY import NEBULA_SKYFIELD_ILLUMINATION

# Turn off interactive plotting to prevent Spyder freezing during save
plt.ioff()

def _build_logger():
    logger = logging.getLogger("NEBULA_GEOM3D")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def _unit_vector(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _build_fov_basis(ra_deg, dec_deg):
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    u_bore = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    u_bore = _unit_vector(u_bore)
    
    # Reference up (Z unless looking at pole)
    ref = np.array([0., 1., 0.]) if abs(u_bore[2]) > 0.9 else np.array([0., 0., 1.])
    u_right = _unit_vector(np.cross(ref, u_bore))
    u_up = _unit_vector(np.cross(u_bore, u_right))
    return u_bore, u_right, u_up

def compute_geo_footprint_corners(r_obs_km, ra_deg, dec_deg, fov_h, rows, cols, target_R=GEO_RADIUS_KM):
    """Ray-Sphere intersection to find where the FOV hits the GEO belt."""
    u_bore, u_right, u_up = _build_fov_basis(ra_deg, dec_deg)
    
    hfov = np.deg2rad(fov_h) / 2.0
    vfov = np.deg2rad(fov_h * (rows/cols)) / 2.0
    tan_h, tan_v = np.tan(hfov), np.tan(vfov)
    
    corners = []
    # O = r_obs_km (Observer position)
    O_sq = np.dot(r_obs_km, r_obs_km)
    R_sq = target_R**2
    C = O_sq - R_sq 

    # Ray-cast 4 corners
    for sx, sy in [(1,1), (-1,1), (-1,-1), (1,-1)]:
        D = _unit_vector(u_bore + sx*tan_h*u_right + sy*tan_v*u_up)
        B = 2.0 * np.dot(r_obs_km, D)
        disc = B*B - 4.0*C
        
        # If ray hits sphere, solve quadratic; else project far out
        t = (-B + np.sqrt(disc))/2.0 if disc >= 0 else target_R
        corners.append(r_obs_km + t*D)
        
    corners.append(corners[0]) # Close loop
    return np.array(corners)

def build_sun(times, eph_path, log):
    log.info("Loading Ephemeris...")
    eph = load(eph_path)
    t = load.timescale().from_datetimes(times)
    return eph['earth'].at(t).observe(eph['sun']).position.km.T

def create_geometry_movie(force_recompute_pixels=False, frame_stride=10, filename_base="nebula_geometry"):
    log = _build_logger()
    
    # 1. Load Data
    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=force_recompute_pixels, sensor_config=ACTIVE_SENSOR, logger=log
    )
    if not obs_tracks: return

    obs_names = list(obs_tracks.keys())
    tar_names = list(tar_tracks.keys())
    times = np.asarray(obs_tracks[obs_names[0]]["times"])
    frames = np.arange(len(times))[::frame_stride]
    
    # 2. Pre-calculate Sun
    r_sun = build_sun(times, NEBULA_SKYFIELD_ILLUMINATION.EPHEMERIS_PATH_DEFAULT, log)

    # 3. Setup Writer (Auto-Fallback)
    if animation.writers.is_available('ffmpeg'):
        writer = animation.writers['ffmpeg'](fps=20, bitrate=3000)
        out_file = NEBULA_OUTPUT_DIR / "GEOM3D_Animations" / f"{filename_base}.mp4"
        log.info(f"FFmpeg found. Saving to MP4: {out_file}")
    else:
        writer = animation.PillowWriter(fps=20)
        out_file = NEBULA_OUTPUT_DIR / "GEOM3D_Animations" / f"{filename_base}.gif"
        log.warning("FFmpeg NOT found. Falling back to GIF (slower but reliable).")
        log.info(f"Saving to GIF: {out_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 4. Setup Plot
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.set_axis_off()
    
    # Static Earth
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    ax.plot_surface(
        R_EARTH*np.cos(u)*np.sin(v), 
        R_EARTH*np.sin(u)*np.sin(v), 
        R_EARTH*np.cos(v), 
        color='blue', alpha=0.2, edgecolor='none'
    )
    
    # Limits
    lim = 1.2 * GEO_RADIUS_KM
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

    # Artists
    sun_line, = ax.plot([],[],[], 'y--', lw=1.5, label='Sun')
    obs_artists = {
        n: (ax.plot([],[],[], 'c^')[0], ax.plot([],[],[], 'c:', lw=1)[0], ax.plot([],[],[], 'c-', lw=1.5)[0])
        for n in obs_names
    }
    scatter = ax.scatter([],[],[], s=3)

    # Pre-calc arrays to speed up loop
    obs_data = {
        n: {
            'pos': np.array(t['r_eci_km']),
            'ra': np.array(t['pointing_boresight_ra_deg']),
            'dec': np.array(t['pointing_boresight_dec_deg']),
            'valid': np.array(t.get('pointing_valid_for_projection', np.ones(len(times), bool)))
        } for n, t in obs_tracks.items()
    }
    
    tar_pos = np.array([tar_tracks[n]['r_eci_km'] for n in tar_names]) # shape (n_tars, n_times, 3)

    # 5. Render Loop
    with writer.saving(fig, str(out_file), dpi=120):
        for i in tqdm(frames, desc="Rendering Frames"):
            # Update Sun
            s = _unit_vector(r_sun[i]) * lim
            sun_line.set_data_3d([0, s[0]], [0, s[1]], [0, s[2]])

            # Update Observers
            for n, (mark, los, box) in obs_artists.items():
                d = obs_data[n]
                if not d['valid'][i]:
                    mark.set_data_3d([],[],[]); los.set_data_3d([],[],[]); box.set_data_3d([],[],[])
                    continue
                
                pos = d['pos'][i]
                corners = compute_geo_footprint_corners(
                    pos, d['ra'][i], d['dec'][i], 
                    ACTIVE_SENSOR.fov_deg, ACTIVE_SENSOR.rows, ACTIVE_SENSOR.cols
                )
                
                mark.set_data_3d([pos[0]], [pos[1]], [pos[2]])
                box.set_data_3d(corners[:,0], corners[:,1], corners[:,2])
                
                # Line from Obs to Center of Box
                ctr = np.mean(corners[:-1], axis=0)
                los.set_data_3d([pos[0], ctr[0]], [pos[1], ctr[1]], [pos[2], ctr[2]])

            # Update Targets (Green if visible to ANY observer)
            # Logic: We check the pre-computed flags in tar_tracks
            t_cols = []
            for t_idx, t_name in enumerate(tar_names):
                is_vis = False
                by_obs = tar_tracks[t_name].get('by_observer', {})
                for o_name in obs_names:
                    # Safe get for boolean flag
                    flags = by_obs.get(o_name, {}).get('on_detector_visible_sunlit')
                    if flags is not None and flags[i]:
                        is_vis = True
                        break
                t_cols.append('lime' if is_vis else 'red')
            
            # Scatter update
            current_tar_pos = tar_pos[:, i, :]
            scatter._offsets3d = (current_tar_pos[:,0], current_tar_pos[:,1], current_tar_pos[:,2])
            scatter.set_color(t_cols)
            
            ax.set_title(f"NEBULA Frame {i}", color='white')
            writer.grab_frame()

    print(f"\nDone! Saved to: {out_file}")

if __name__ == "__main__":
    create_geometry_movie()