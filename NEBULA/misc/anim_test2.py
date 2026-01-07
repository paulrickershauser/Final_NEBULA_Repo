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
from Configuration.NEBULA_SENSOR_CONFIG import EVK4_SENSOR
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER
from Utility.RADIOMETRY import NEBULA_SKYFIELD_ILLUMINATION

# Turn off interactive plotting
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
    ref = np.array([0., 1., 0.]) if abs(u_bore[2]) > 0.9 else np.array([0., 0., 1.])
    u_right = _unit_vector(np.cross(ref, u_bore))
    u_up = _unit_vector(np.cross(u_bore, u_right))
    return u_bore, u_right, u_up

def intersect_ray_sphere(origin, direction, radius):
    # Solve |O + tD|^2 = R^2 for t > 0
    # t^2 + 2(O.D)t + (O^2 - R^2) = 0
    O_sq = np.dot(origin, origin)
    R_sq = radius**2
    C = O_sq - R_sq
    B = 2.0 * np.dot(origin, direction)
    disc = B*B - 4.0*C
    if disc < 0: return None
    t = (-B + np.sqrt(disc))/2.0
    return origin + t * direction

def compute_geo_footprint_corners(r_obs_km, ra_deg, dec_deg, fov_h, rows, cols, target_R=GEO_RADIUS_KM):
    u_bore, u_right, u_up = _build_fov_basis(ra_deg, dec_deg)
    hfov = np.deg2rad(fov_h) / 2.0
    vfov = np.deg2rad(fov_h * (rows/cols)) / 2.0
    tan_h, tan_v = np.tan(hfov), np.tan(vfov)
    corners = []
    # Ray-cast 4 corners
    for sx, sy in [(1,1), (-1,1), (-1,-1), (1,-1)]:
        D = _unit_vector(u_bore + sx*tan_h*u_right + sy*tan_v*u_up)
        P = intersect_ray_sphere(r_obs_km, D, target_R)
        if P is None: P = r_obs_km + target_R * D # Fallback
        corners.append(P)
    corners.append(corners[0])
    return np.array(corners)

def build_sun(times, eph_path, log):
    log.info("Loading Ephemeris...")
    eph = load(eph_path)
    t = load.timescale().from_datetimes(times)
    return eph['earth'].at(t).observe(eph['sun']).position.km.T

def create_geometry_movie(force_recompute_pixels=False, frame_stride=10, filename_base="nebula_celestial_check"):
    log = _build_logger()
    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=force_recompute_pixels, sensor_config=EVK4_SENSOR, logger=log
    )
    if not obs_tracks: return

    obs_names = list(obs_tracks.keys())
    tar_names = list(tar_tracks.keys())
    times = np.asarray(obs_tracks[obs_names[0]]["times"])
    frames = np.arange(len(times))[::frame_stride]
    
    r_sun = build_sun(times, NEBULA_SKYFIELD_ILLUMINATION.EPHEMERIS_PATH_DEFAULT, log)

    if animation.writers.is_available('ffmpeg'):
        writer = animation.writers['ffmpeg'](fps=20, bitrate=3000)
        out_file = NEBULA_OUTPUT_DIR / "GEOM3D_Animations" / f"{filename_base}.mp4"
    else:
        writer = animation.PillowWriter(fps=20)
        out_file = NEBULA_OUTPUT_DIR / "GEOM3D_Animations" / f"{filename_base}.gif"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.set_axis_off()
    
    # CELESTIAL SPHERE SETUP
    R_CEL = 4.0 * GEO_RADIUS_KM  # ~168,000 km radius
    
    # Draw Celestial Sphere Wireframe
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(R_CEL*np.cos(u)*np.sin(v), R_CEL*np.sin(u)*np.sin(v), R_CEL*np.cos(v), color='white', alpha=0.05, linewidth=0.3)
    
    # Draw GEO Belt Wireframe (for reference)
    ax.plot_wireframe(GEO_RADIUS_KM*np.cos(u)*np.sin(v), GEO_RADIUS_KM*np.sin(u)*np.sin(v), GEO_RADIUS_KM*np.cos(v), color='gray', alpha=0.1, linewidth=0.5)

    # Earth
    ax.plot_surface(R_EARTH*np.cos(u)*np.sin(v), R_EARTH*np.sin(u)*np.sin(v), R_EARTH*np.cos(v), color='blue', alpha=0.4, edgecolor='none')
    
    limit = 1.1 * R_CEL
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)

    # Artists
    sun_line, = ax.plot([],[],[], 'y--', lw=1.5, label='Sun')
    
    # Observers & Celestial Pointers
    obs_artists = {}
    for n in obs_names:
        # Obs Marker, GEO Footprint, Celestial Ray, Celestial Marker
        obs_artists[n] = {
            'mk': ax.plot([],[],[], 'c^')[0],
            'box': ax.plot([],[],[], 'c-', lw=1)[0],
            'ray': ax.plot([],[],[], 'm:', lw=0.5)[0], # Magenta ray to celestial sphere
            'star': ax.plot([],[],[], 'm*', ms=5)[0]   # Star on celestial sphere
        }
    
    scatter = ax.scatter([],[],[], s=2)

    obs_data = {
        n: {
            'pos': np.array(t['r_eci_km']),
            'ra': np.array(t['pointing_boresight_ra_deg']),
            'dec': np.array(t['pointing_boresight_dec_deg']),
            'valid': np.array(t.get('pointing_valid_for_projection', np.ones(len(times), bool)))
        } for n, t in obs_tracks.items()
    }
    tar_pos = np.array([tar_tracks[n]['r_eci_km'] for n in tar_names])

    with writer.saving(fig, str(out_file), dpi=100):
        for i in tqdm(frames, desc="Rendering"):
            s = _unit_vector(r_sun[i]) * limit
            sun_line.set_data_3d([0, s[0]], [0, s[1]], [0, s[2]])

            for n, art in obs_artists.items():
                d = obs_data[n]
                if not d['valid'][i]:
                    for k in art: art[k].set_data_3d([],[],[])
                    continue
                
                pos = d['pos'][i]
                ra, dec = d['ra'][i], d['dec'][i]
                
                # 1. GEO Footprint
                corners = compute_geo_footprint_corners(pos, ra, dec, EVK4_SENSOR.fov_deg, EVK4_SENSOR.rows, EVK4_SENSOR.cols)
                art['mk'].set_data_3d([pos[0]], [pos[1]], [pos[2]])
                art['box'].set_data_3d(corners[:,0], corners[:,1], corners[:,2])
                
                # 2. Celestial Intersection
                u_bore, _, _ = _build_fov_basis(ra, dec)
                P_cel = intersect_ray_sphere(pos, u_bore, R_CEL)
                if P_cel is not None:
                    # Draw line from Obs to Celestial Point
                    art['ray'].set_data_3d([pos[0], P_cel[0]], [pos[1], P_cel[1]], [pos[2], P_cel[2]])
                    art['star'].set_data_3d([P_cel[0]], [P_cel[1]], [P_cel[2]])

            t_cols = []
            for t_name in tar_names:
                is_vis = False
                by_obs = tar_tracks[t_name].get('by_observer', {})
                for o_name in obs_names:
                    flags = by_obs.get(o_name, {}).get('on_detector_visible_sunlit')
                    if flags is not None and flags[i]:
                        is_vis = True; break
                t_cols.append('lime' if is_vis else 'red')
            
            cp = tar_pos[:, i, :]
            scatter._offsets3d = (cp[:,0], cp[:,1], cp[:,2])
            scatter.set_color(t_cols)
            
            ax.set_title(f"Celestial Check Frame {i}", color='white')
            writer.grab_frame()

    print(f"Saved: {out_file}")

if __name__ == "__main__":
    create_geometry_movie()