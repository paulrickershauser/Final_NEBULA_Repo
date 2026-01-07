import occultation_simulation as occ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
import os
import glob
import multiprocessing as mp
import h5py
import time
import pdb
import pickle


def init_pool(
    photocurrents,
    dark,
    leak,
    pix_x,
    pix_y,
    sim_freq,
    freq,
    pos_thres,
    neg_thres,
    rng_seed,
    file_loc,
):
    global I_dark
    global I_moon
    global Leak_rate
    global num_pix_x
    global num_pix_y
    global num_steps_per_second
    global cuttoff_freq
    global pos_threshold
    global neg_threshold
    global seed
    global file_location

    I_dark = dark
    I_moon = photocurrents
    Leak_rate = leak
    num_pix_x = 1  # pix_x
    num_pix_y = 1  # pix_y
    num_steps_per_second = sim_freq
    cuttoff_freq = freq
    pos_threshold = pos_thres
    neg_threshold = neg_thres
    seed = rng_seed
    file_location = file_loc


def get_center_pixel(shape):
    return tuple(s // 2 for s in shape)


def occultation_sim_current_variation(h5_path, lock):
    """Simulate the occultation for a given HDF5 file over various currents."""
    occ_sim = occ.OccultationSimulation()
    results = []
    # os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # Disable file locking for multiprocessing
    with h5py.File(h5_path, "a") as f:
        lowres_grp = f["lowresmask"]
        frame_names = list(lowres_grp.keys())
        # Stack all frames into a 3D array (time, y, x)
        lowres_stack = np.stack([lowres_grp[name][()] for name in frame_names])
        # Stack all frames for the center pixel into a 1D array (time,)
        center_pixel = get_center_pixel(lowres_stack.shape[1:])
        lowres_stack = lowres_stack[:, center_pixel[0], center_pixel[1]]
        for i, flux in enumerate(I_moon):
            # Scale mask by photon flux (broadcasts over all frames)
            incident_flux = lowres_stack * flux  # shape: (t, y, x)
            # Simulate events: run occultation simulation
            events, simulation_time = occ_sim.occultation_sim(
                incident_flux,
                I_dark,
                Leak_rate,
                num_pix_x,
                num_pix_y,
                num_steps_per_second,
                cuttoff_freq,
                pos_threshold,
                neg_threshold,
                seed,
                extra_noise=True,
                varying_cuttoff=True,
                junction_leak=True,
                parasitic_leak=False,
            )
            # Save DataFrame to HDF5 under group 'events' with flux as name
            events_grp_string = "events_pthreshold_{:f}_nthreshold_{:f}".format(
                pos_threshold, neg_threshold
            )
            events_grp = f.require_group(events_grp_string)
            dset_name = (
                "current_{:e}".format(flux.value)
                .replace("-", "neg")
                .replace(".", "dot")
                .replace("+", "pos")
            )
            # Remove if exists
            if dset_name in events_grp:
                del events_grp[dset_name]
            events.to_hdf(h5_path, key=events_grp_string + "/" + dset_name, mode="a")
            # Store center pixel event rate as attribute
            events_grp[dset_name].attrs["center_pixel_event_rate"] = len(events) / len(
                lowres_stack
            )
            results.append((h5_path, flux.value, len(events) / len(lowres_stack)))
    return results


if __name__ == "__main__":
    pdb.set_trace()
    # Choose if output should include event array
    raw_events_out = True

    # Ensure all local variables exist to run the simulation
    with open("collection_parameters_occultation_tests.pickle", "rb") as picklefile:
        collect_params = pickle.load(picklefile)
    num_pix_x = collect_params["x pix"]
    num_pix_y = collect_params["y pix"]
    num_steps_per_second = collect_params["Sim Steps Per Second"]
    cuttoff_freq = collect_params["Cutoff Frequency"]
    pos_threshold = collect_params["Positive Event Threshold"]
    neg_threshold = collect_params["Negative Event Threshold"]
    # pos_threshold = .1
    # neg_threshold = .1
    I_dark = collect_params["Dark Current"]
    leak_rate = collect_params["Leak Rate"]
    try:
        seed = collect_params["Seed"]
        print("Seed: {}".format(seed))
    except NameError:
        # Establish seed
        seed_seq = np.random.SeedSequence()
        seed = seed_seq.generate_state(1)
        seed = int(seed)

    # Create a lock object to enable saving
    m = mp.Manager()
    lock = m.Lock()

    # Generate a tuple with each input variable for all the runs
    argsList = []
    # Use glob to find all .h5 files in the directory
    OCC_HDF5_DIR = "occultation_test/occultation_hdf5"
    h5_files = glob.glob(os.path.join(OCC_HDF5_DIR, "*.h5"))
    # h5_files = ['occultation_test/occultation_hdf5/photon_flux_1_objectsize_1.00_traversespeed_0.001.h5']

    # Recieved Photon Flux
    Incident_flux = (
        np.logspace(1, 20, 20) * u.ph / u.s
    )  # np.logspace(1,20,20) * u.ph / u.s  # Varying photon flux
    # Induced Current using quantum efficiency of 0.27
    e = 1.602176634e-19 * u.A * u.s
    eta = 0.27 * (1 / u.ph)
    # Caclulate the circuit input current for each photon flux level
    I_in = eta * e * Incident_flux

    for i, h5_file in enumerate(h5_files):
        argsList.append((h5_file, lock))

    with mp.pool.Pool(
        processes=1,
        initializer=init_pool,
        initargs=(
            I_in,
            I_dark,
            leak_rate,
            num_pix_x,
            num_pix_y,
            num_steps_per_second,
            cuttoff_freq,
            pos_threshold,
            neg_threshold,
            seed,
            OCC_HDF5_DIR,
        ),
    ) as pool:
        pool.starmap(occultation_sim_current_variation, argsList)
        pool.close()
        pool.join()

        print("Simulation complete. Results saved in HDF5 files.")

# Induced photocurrent go from I_in = np.logspace(1,20,1) u.ph/u.s recieved to the induced current
