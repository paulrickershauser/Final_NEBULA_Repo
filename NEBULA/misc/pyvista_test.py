from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER
from Simulation_Viewer import NEBULA_PYVISTA_DATA, NEBULA_PYVISTA_SCENE

obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
    force_recompute=False
)

# dataset = NEBULA_PYVISTA_DATA.build_viewer_dataset(
#     observer_tracks=obs_tracks,
#     target_tracks=tar_tracks,
#     visibility_field="los_visible",
# )
# earth_texture = r"C:\Users\prick\Desktop\Research\NEBULA\Input\NEBULA_PYVISTA_DATA\earth_noClouds.0330_web.png"
# scene = NEBULA_PYVISTA_SCENE.create_scene(dataset,earth_texture_path=earth_texture)
# plotter = scene.plotter

# active_observer = dataset["observer_names"][0]

# times = dataset["times"]   # list/array of datetimes

# # Add a text actor in the lower-left that we'll update every slider move
# time_text_actor = plotter.add_text(
#     "",
#     position="lower_left",
#     font_size=12,
#     color="white",
# )

# def slider_callback(value):
#     # Map slider float to integer index
#     idx = int(round(value))

#     # Update scene geometry/colors
#     NEBULA_PYVISTA_SCENE.update_scene_time_index(
#         scene_state=scene,
#         index=idx,
#         active_observer=active_observer,
#     )

#     # Build a human-readable time label
#     t = times[idx]
#     if hasattr(t, "strftime"):
#         label = t.strftime("%Y-%m-%d %H:%M:%S (UTC)")
#     else:
#         # Fallback (if they're not datetime objects)
#         label = f"Index {idx}"

#     # Update the text actor in-place
#     time_text_actor.SetInput(label)

# # Slider at the *bottom* of the window
# plotter.add_slider_widget(
#     slider_callback,
#     rng=[0, dataset["n_times"] - 1],
#     title="Time index",
#     pointa=(0.1, 0.1),   # normalized display coords (x,y)
#     pointb=(0.9, 0.1),
# )

# plotter.show()

