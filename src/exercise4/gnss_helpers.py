"""GNSS Exercise 4 -- supplied helpers (not assessed). NumPy/Matplotlib only.

Positions are ECEF in metres. Toy pseudoranges are SIMULATED, not measured.
The broadcast satellite ECEF snapshots are derived from the supplied Galileo
RINEX 3.04 navigation file and kept fixed for this short experiment.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

def load_data():
    """Returns an np.load(...) data bundle; no georinex is required."""
    return np.load(HERE / 'gnss_ex04_data.npz')

def geodetic_to_ecef(phi_deg, lam_deg, h_m):
    """Reference conversion for helper internals only; student task is inverse."""
    phi, lam = np.deg2rad([phi_deg, lam_deg])
    aE, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    N = aE / np.sqrt(1 - e2 * np.sin(phi)**2)
    return np.array([(N + h_m)*np.cos(phi)*np.cos(lam),
                     (N + h_m)*np.cos(phi)*np.sin(lam),
                     (N*(1 - e2) + h_m)*np.sin(phi)])

def enu_rotation(phi_deg, lam_deg):
    """Rows of R map ECEF line-of-sight displacement into East, North, Up."""
    phi, lam = np.deg2rad([phi_deg, lam_deg])
    return np.array([[-np.sin(lam), np.cos(lam), 0],
                     [-np.sin(phi)*np.cos(lam), -np.sin(phi)*np.sin(lam), np.cos(phi)],
                     [np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)]])

def skyplot(az_deg, el_deg, labels, mask_deg=15, ax=None):
    """Polar plot: N at top, clockwise azimuth, zenith at center."""
    if ax is None: _, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 5))
    az_deg=np.asarray(az_deg);el_deg=np.asarray(el_deg)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_ylim(0,90);ax.set_yticks([0,30,60,90]); ax.set_yticklabels(['90°','60°','30°','0°'])
    ok=el_deg>=mask_deg
    ax.scatter(np.deg2rad(az_deg[ok]),90-el_deg[ok],s=60,marker='o',label='Used')
    ax.scatter(np.deg2rad(az_deg[~ok]),90-el_deg[~ok],s=55,marker='x',label='Masked')
    ang=np.linspace(0,2*np.pi,361)
    ax.plot(ang,np.full_like(ang,90-mask_deg),'--',linewidth=1,label=f'{mask_deg}° mask')
    for az,el,name in zip(az_deg,el_deg,labels):
        ax.annotate(str(name),(np.deg2rad(az),90-el),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.legend(loc='upper left',bbox_to_anchor=(1.03,1.0),fontsize=8)
    ax.set_title('Galileo satellite sky plot',pad=20)
    return ax

def _least_squares_position(sats_m, rho_m, x0_m):
    """Pedagogical, UNWEIGHTED 4-unknown pseudorange least-squares solver.

    Estimated parameters are x,y,z [m] and receiver clock range bias b [m].
    The students are not required to implement this solver.
    """
    x=np.r_[np.asarray(x0_m,dtype=float).copy(),0.0]
    for _ in range(12):
        d=x[:3]-sats_m
        rng=np.linalg.norm(d,axis=1)
        H=np.column_stack((d/rng[:,None],np.ones(len(sats_m))))
        correction=np.linalg.lstsq(H,rho_m-(rng+x[3]),rcond=None)[0]
        x+=correction
        if np.linalg.norm(correction[:3])<1e-5 and abs(correction[3])<1e-5:break
    return x

def mask_experiment(sat_positions_m, receiver_ecef_m, elevation_deg,
                    mask_values=(0,10,20,30), n_trials=200,
                    noise_model='elevation', seed=2026, error_model=None):
    """Monte-Carlo position RMSE for each mask. SAME noise draws across masks.

    Synthetic observations: rho_i = ||s_i-r_true|| + b_true + noise_i + bias_i.
    Noise is in METRES; model is intentionally illustrative, not observed data.
    'constant' -> sigma=3 m, bias=0 m.
    'elevation' -> sigma=2+20 exp(-el/12deg) m; bias=12 exp(-el/10deg) m.
    If error_model is supplied, its (sigma_m, bias_m) arrays replace these assumptions.
    No satellite-clock/ionosphere/troposphere/Earth-rotation corrections are modeled.
    """
    sats=np.asarray(sat_positions_m,dtype=float)
    r=np.asarray(receiver_ecef_m,dtype=float)
    el=np.asarray(elevation_deg,dtype=float)
    if len(sats)!=len(el):raise ValueError('Satellite and elevation arrays differ')
    if error_model is not None:
        sigma,bias=error_model(el)
        sigma=np.asarray(sigma,dtype=float);bias=np.asarray(bias,dtype=float)
        if sigma.shape!=el.shape or bias.shape!=el.shape or np.any(sigma<=0):
            raise ValueError('Custom model must return positive sigma and matching sigma/bias arrays')
    elif noise_model=='constant':sigma=np.full(len(el),3.);bias=np.zeros(len(el))
    elif noise_model=='elevation':
        sigma=2.0+20.0*np.exp(-el/12.)
        bias=12.0*np.exp(-el/10.)
    else: raise ValueError("Select 'constant' or 'elevation'")
    rng=np.random.default_rng(seed)
    draws=rng.normal(size=(n_trials,len(el)))
    rho_ideal=np.linalg.norm(sats-r,axis=1)+75000.0
    out=[]
    for mask in mask_values:
        selected=el>=mask
        n=int(selected.sum())
        if n<4:out.append((mask,n,np.nan,np.nan));continue
        errors=[]
        for z in draws:
            rho=rho_ideal+sigma*z+bias
            est=_least_squares_position(sats[selected],rho[selected],r+np.array([200.,-120.,160.]))
            errors.append(np.linalg.norm(est[:3]-r))
        errors=np.asarray(errors)
        out.append((mask,n,np.sqrt(np.mean(errors**2)),np.percentile(errors,95)))
    return np.asarray(out,dtype=float)

def plot_mask_results(results, ax=None, label='Elevation-dependent errors'):
    """Each point: 3D position RMSE [m] versus elevation mask [deg]."""
    if ax is None: _,ax=plt.subplots(figsize=(6.0,3.7))
    ax.plot(results[:,0],results[:,2],marker='o',label=label)
    for mask,count,rmse,_ in results:
        if np.isfinite(rmse):ax.annotate(f'{int(count)} SV',(mask,rmse),xytext=(3,8),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Elevation mask [degrees]');ax.set_ylabel('3D position RMSE [m]')
    ax.grid(True,alpha=.25);ax.legend()
    return ax
