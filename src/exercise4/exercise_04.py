"""GNSS Exercise 4 -- student starter. Execute one problem at a time.
python exercise_04.py 1|2|3|4|hw
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pow
from gnss_helpers import load_data, skyplot, mask_experiment, plot_mask_results

DATA = load_data()
A_E = 6378137.0
F = 1.0 / 298.257223563
E2 = 2.0*F - F**2

# PROBLEM 1: implement ONLY ecef_to_geodetic(). task1() is supplied.
def ecef_to_geodetic(r_ecef_m):
    """Return (geodetic latitude [deg], longitude [deg], height [m])."""
    x, y, z = np.asarray(r_ecef_m, dtype=float)
    # TODO_DONE 1: compute p, longitude, iterated geodetic latitude, and height.
    p = np.sqrt(x * x + y * y)
    lam = np.arctan2(y,x)
    phi = np.arctan2(z, p * (1 - (np.e * np.e)))
    for i in range(0,30):
        N_phi = (A_E / (np.sqrt(1 - E2 * (np.sin(phi) * np.sin(phi)))))
        phi = np.arctan2(z + E2 * N_phi * np.sin(phi), p) 

    h = (p / np.cos(phi)) - N_phi

    return (np.degrees(phi), np.degrees(lam), h)


def task1():
    r = DATA['receiver_ecef_m']
    print('Receiver ECEF [m]:', r)
    phi, lam, h = ecef_to_geodetic(r)
    print(f'Latitude = {phi:.6f} deg; longitude = {lam:.6f} deg; height = {h:.2f} m')


# PROBLEM 2: implement these two functions AND the loop in task2().
def ecef_to_enu(sat_ecef_m, rec_ecef_m, phi_deg, lam_deg):
    """Return local line-of-sight vector [east, north, up] in metres."""
    # TODO 2a: displacement satellite - receiver, followed by ENU rotation.
    d_i = sat_ecef_m - rec_ecef_m
    lam = np.radians(lam_deg)
    phi = np.radians(phi_deg)

    d_enu = np.array([0,0,0])
    R_enu = np.array([-np.sin(lam), np.cos(lam), 0],
                     [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
                     [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])

    return R_enu @ d_enu

    #raise NotImplementedError('Problem 2: complete ecef_to_enu')


def enu_to_angles(enu_m):
    """Return (azimuth [deg], elevation [deg])."""
    east, north, up = enu_m
    # TODO 2b: azimuth clockwise from north; elevation above horizon.
    
    azi_i = np.arctan2(enu_m[0], enu_m[1])
    elev_i = np.arctan2(enu_m[2], np.sqrt(enu_m[0]*enu_m[0] + enu_m[1] * enu_m[1]))
    # convert to degrees
    azi_i = np.degrees(azi_i) % 360
    elev_i = np.degrees(elev_i)

    return (azi_i, elev_i)
    #raise NotImplementedError('Problem 2: complete enu_to_angles')


def task2():
    rec = DATA['receiver_ecef_m']
    phi, lam, h = DATA['receiver_geodetic_deg_m']
    sats = DATA['sat_ecef_m']; labels = DATA['sv']
    # TODO 2c: loop over satellites; collect angles; print a satellite/az/el table.
    az_deg = 0 
    el_deg = 0
    for sat in sats: 
        az_deg, el_deg = enu_to_angles(ecef_to_enu(sat, rec, phi, lam))


    # TODO 2d: call skyplot(az_deg, el_deg, labels, mask_deg=15); plt.show().

    #raise NotImplementedError('Problem 2: complete task2')


# PROBLEM 3: implement ONLY task3(); simulation and plotting helpers supplied.
def task3():
    sats = DATA['sat_ecef_m']; rec = DATA['receiver_ecef_m']
    el = DATA['sat_el_deg']; labels = DATA['sv']
    masks = (0, 10, 20, 30)
    # TODO 3a: loop over masks. keep = el >= mask; print count and IDs.
    # TODO 3b: call mask_experiment twice with noise_model='constant'/'elevation'.
    # TODO 3c: plot both result arrays with plot_mask_results(..., ax=ax).
    #           Finish with plt.show(). Use a shared axes object.
    raise NotImplementedError('Problem 3: complete task3')


# PROBLEM 4: implement spreading, mixture and correlation; NO correlation bar chart.
def task4():
    codes = DATA['gps_ca_codes'].astype(float)       # shape (3, 1023); +/-1
    symbols = DATA['data_symbols'].astype(float)     # shape (3,); +/-1
    amps = DATA['signal_amplitudes'].astype(float)   # shape (3,); positive
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 4.0, size=codes.shape[1])
    # TODO 4a: spread[i, :] = amps[i] * symbols[i] * codes[i, :]
    # TODO 4b: combined = sum across satellites; y = combined + noise.
    # TODO 4b: plot FIRST 40 samples of combined and y in separate figures.
    # TODO 4c: score[i] = np.mean(y * codes[i, :]); d_hat = np.sign(score).
    # TODO 4c: print score, transmitted symbols and recovered symbols.
    raise NotImplementedError('Problem 4: complete task4')


# HOMEWORK: implement the error model; no code-delay estimation.
def elevation_error_model(el_deg):
    """Return arrays (sigma_m, bias_m) for each satellite elevation [deg]."""
    el_deg = np.asarray(el_deg, dtype=float)
    # TODO HW a: sigma = 2 + 20 exp(-el/12); bias = 12 exp(-el/10), in metres.
    raise NotImplementedError('Homework: complete elevation_error_model')


def homework():
    sats=DATA['sat_ecef_m'];rec=DATA['receiver_ecef_m'];el=DATA['sat_el_deg']
    masks=(0,10,20,30)
    # TODO HW b: calculate model values for elevations 10, 30, 60 degrees.
    # TODO HW c: run mask_experiment(...,error_model=elevation_error_model).
    # TODO HW d: compare with the built-in elevation model and print maximum RMSE difference.
    # TODO HW e: optional: change sigma_excess 20 -> 10 and compare RMSE.
    raise NotImplementedError('Homework: complete homework')


if __name__ == '__main__':
    commands = {'1':task1, '2':task2, '3':task3, '4':task4, 'hw':homework}
    if len(sys.argv)!=2 or sys.argv[1] not in commands:
        print(__doc__); sys.exit(2)
    commands[sys.argv[1]]()
