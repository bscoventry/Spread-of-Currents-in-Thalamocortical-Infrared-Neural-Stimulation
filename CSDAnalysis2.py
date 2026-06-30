"""
CSD_CoM_Tracking.py
====================================================================
Combines:
  - The rigorous center-of-mass / trajectory-linking engine from
    step02.py (connected-component mass centers, Donoho threshold,
    weight-based trajectory linking, gap-filling interpolation)
  - The reporting, multi-wave breakdown, and video-generation
    structure from CSDAnalysis2.py

Operates directly on a single 101 x 101 x 1526 CSD cube (rows, cols,
time) from a 2x8 Utah-style array interpolated onto a 101x101 grid.

NOTE ON ELECTRODE LAYOUT
-------------------------
step02.py's original depuration_x_distance()/min_cm_weight() assumed
a 64x64 raw-channel layout (4096 channels). That does not apply here:
your data is already a 101x101 *interpolated* CSD grid built from a
2-row x 8-column array (16 channels). This script replaces that
64x64 assumption with electrode pixel coordinates derived the same
way CSDAnalysis2.py maps physical mm -> pixels.

ACD = [1,2,...,16] channel numbers are mapped onto physical (row, col)
positions on the 2x8 array via the explicit ELECTRODE_CHANNEL_MAP defined
below build_electrode_positions() (not a simple row-major scheme).
====================================================================
"""

import os
import math
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')   # use 'QtAgg' if you prefer Qt-based windows
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import ndimage
import pdb
# =====================================================================
# 1. SETUP PARAMETERS & CONFIGURATION
# =====================================================================
xdims, ydims, total_samples = 101, 101, 1526

df1 = pd.read_pickle('LFPCSD.pkl')
pdb.set_trace()
csd_data = np.asarray(df1['estCSD'][10], dtype=np.float64)   # (101, 101, 1526)

baseline_end_sample = 305

# Electrode / array geometry (2 rows x 8 columns, 0.25mm x-pitch, 0.50mm y-pitch)
N_ROWS_ELEC, N_COLS_ELEC = 2, 8
ACD = list(range(1, 17))   # [1, 2, ..., 16]

grid_physical_x_mm = (N_COLS_ELEC - 1) * 0.25
grid_physical_y_mm = (N_ROWS_ELEC - 1) * 0.375

mm_per_pixel_x = grid_physical_x_mm / (xdims - 1)
mm_per_pixel_y = grid_physical_y_mm / (ydims - 1)
seconds_per_sample = 1.0 / total_samples

# Center-of-mass / trajectory-linking parameters (mirrors STEP02.ipynb cell_0004)
minchannels = 3     # minimum connected pixels per CSD blob               *
tol_dist    = 4.0   # spatial tolerance (pixels) for trajectory linking   *
tol_time    = 3      # temporal tolerance (frames) for trajectory linking  *
Tmin        = 3      # minimum trajectory length (frames)                  *

SAVE_AND_PLAY_VIDEO = True
VIDEO_FILENAME = "csd_propagation_trajectory.mp4"


# =====================================================================
# 2. STATISTICAL / THRESHOLD HELPERS  (from step02.py)
# =====================================================================
def donoho(x) -> float:
    """Donoho-Johnstone robust noise estimator: |median(|x|) / 0.6745|"""
    return float(abs(np.median(np.abs(np.asarray(x, dtype=float).ravel())) / 0.6745))


def sigma_data(data) -> float:
    """sqrt(2 * log(n))  - universal threshold scale factor."""
    return float(np.sqrt(2.0 * np.log(len(np.asarray(data).ravel()))))


def donoho_matrix_3d(data: np.ndarray) -> np.ndarray:
    """Apply the Donoho estimator along axis 2 of an (a x b x c) array."""
    a, b, _ = data.shape
    R = np.empty((a, b), dtype=np.float64)
    for i in range(a):
        for j in range(b):
            R[i, j] = donoho(data[i, j, :])
    return R


def fences(data):
    data = np.asarray(data, dtype=float).ravel()
    q1, q3 = float(np.percentile(data, 25)), float(np.percentile(data, 75))
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _jackknife_mean_error(data):
    data = np.asarray(data, dtype=float)
    n = len(data)
    jk = np.array([(data[:i].sum() + data[i + 1:].sum()) / (n - 1) for i in range(n)])
    mu = jk.mean()
    err = np.sqrt((n - 1) / n * np.sum((jk - mu) ** 2))
    return mu, err


def thresholding(aux) -> np.ndarray:
    """Battery of threshold candidates; only those inside [min,max] kept."""
    from skimage import filters as sk_filters

    W = np.round(np.asarray(aux, dtype=float).ravel(), 3)
    t = np.zeros(19)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xmean, xerror = _jackknife_mean_error(W)
        t[0] = donoho(W) * sigma_data(W)
        t[1] = W.mean() + 2.0 * W.std()
        t[2] = W.mean() - 2.0 * W.std()
        t[3], t[4] = fences(W)

        for method_name, idx in [
            ("li", 5), ("entropy", 6), ("isodata", 7), ("minimum", 8),
            ("triangle", 9), ("yen", 10), ("otsu", 11), ("mean", 12),
        ]:
            try:
                fn = getattr(sk_filters, f"threshold_{method_name}")
                t[idx] = float(fn(W))
            except Exception:
                pass

        t[14] = xmean + xerror
        t[15] = xmean - xerror
        t[16] = float(np.percentile(W, 0.33))
        t[17] = float(np.median(W)) + xerror
        t[18] = float(np.median(W)) - xerror

    lo, hi = W.min(), W.max()
    t = t[(t >= lo) & (t <= hi)]
    return np.sort(np.unique(np.round(t, 10)))


def clean_dictionary(D: dict) -> dict:
    to_del = []
    for k, v in D.items():
        try:
            is_empty = hasattr(v, "__len__") and len(v) == 0
            if (not hasattr(v, "__len__") and (v == "null" or v == "")) or is_empty:
                to_del.append(k)
        except Exception:
            pass
    for k in to_del:
        del D[k]
    return D


# =====================================================================
# 3. CONNECTED-COMPONENT / MASS-CENTRE ANALYSIS  (from step02.py,
#    generalised to any grid size - no longer hard-coded to 64x64)
# =====================================================================
def eight_neigh(channel, max_row, max_col):
    """Eight-neighbourhood of a point, 0-indexed, bounded by grid size."""
    j, k = int(channel[0]), int(channel[1])
    result = set()
    for dj in (-1, 0, 1):
        for dk in (-1, 0, 1):
            if dj == 0 and dk == 0:
                continue
            nj, nk = j + dj, k + dk
            if 0 <= nj < max_row and 0 <= nk < max_col:
                result.add((nj, nk))
    return result


def disjoint_components(cartesian_channels, max_row, max_col):
    """Connected-component labelling using 8-connectivity."""
    temp_set = {tuple(c) for c in cartesian_channels}
    components = []
    while temp_set:
        seed = temp_set.pop()
        queue = [seed]
        component = [list(seed)]
        while queue:
            current = queue.pop()
            for nb in eight_neigh(current, max_row, max_col):
                if nb in temp_set:
                    temp_set.discard(nb)
                    queue.append(nb)
                    component.append(list(nb))
        components.append(component)
    return components


def mass_centers(csd: np.ndarray, t: int, channels: list, min_channels: int) -> np.ndarray:
    """Weighted centroids (x, y, Omega) for connected components >= min_channels."""
    rows, cols, _ = csd.shape
    if len(channels) == 0:
        return np.empty((0, 3), dtype=np.float64)

    DCs = disjoint_components(channels, rows, cols)
    rows_list = []
    for component in DCs:
        if len(component) < min_channels:
            continue
        Omega, x_sum, y_sum = 0.0, 0.0, 0.0
        for ch in component:
            row, col = int(ch[0]), int(ch[1])
            omega = csd[row, col, t]
            Omega += omega
            x_sum += col * omega
            y_sum += row * omega
        if Omega != 0:
            x_c, y_c = x_sum / Omega, y_sum / Omega
        else:
            x_c, y_c = 0.0, 0.0
        rows_list.append([x_c, y_c, Omega])

    if not rows_list:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(rows_list, dtype=np.float64)


def get_centers_of_mass(csd: np.ndarray, min_channels: int, epsilon=0):
    """
    For each time frame, find negative (sink) and positive (source)
    super-threshold CSD blobs and compute their weighted centroids.

    Returns
    -------
    cmn : dict {t -> ndarray (N,3)}  negative centres (sinks)
    cmp : dict {t -> ndarray (N,3)}  positive centres (sources)
    """
    rows, cols, t_n = csd.shape
    cmp, cmn = {}, {}

    if np.ndim(epsilon) == 0 and epsilon == 0:
        epsilon = np.abs(np.std(csd, axis=2)) * 3.0

    for t in range(t_n):
        neg_chs, pos_chs = [], []
        frame = csd[:, :, t]
        for r in range(rows):
            for c in range(cols):
                thr = float(epsilon[r, c]) if hasattr(epsilon, "__len__") else float(epsilon)
                val = frame[r, c]
                if val <= -thr:
                    neg_chs.append([r, c])
                elif val >= thr:
                    pos_chs.append([r, c])
        cmn[t] = mass_centers(csd, t, neg_chs, min_channels)
        cmp[t] = mass_centers(csd, t, pos_chs, min_channels)

    return cmn, cmp


# =====================================================================
# 4. DISTANCE HELPERS  (from step02.py)
# =====================================================================
def distance_vectors(v, M: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    M = np.asarray(M, dtype=float)
    return np.sqrt((v[0] - M[:, 0]) ** 2 + (v[1] - M[:, 1]) ** 2)


def distance_coords(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2))


# =====================================================================
# 5. ELECTRODE LAYOUT  (replaces step02.py's hard-coded 64x64 _LAYOUT)
# =====================================================================
# Explicit (row, col) -> channel-number map for the physical 2x8 array,
# as provided by the user. (row, col) are 1-indexed positions on the array.
ELECTRODE_CHANNEL_MAP = {
    (1, 1): 1,  (1, 2): 3,  (1, 3): 5,  (1, 4): 7,
    (1, 5): 2,  (1, 6): 4,  (1, 7): 6,  (1, 8): 8,
    (2, 1): 10, (2, 2): 12, (2, 3): 14, (2, 4): 16,
    (2, 5): 9,  (2, 6): 11, (2, 7): 13, (2, 8): 15,
}


def build_electrode_positions(acd, n_rows=N_ROWS_ELEC, n_cols=N_COLS_ELEC,
                               grid_rows=ydims, grid_cols=xdims,
                               channel_map=ELECTRODE_CHANNEL_MAP):
    """
    Map channel numbers in `acd` onto pixel (row, col) coordinates in the
    interpolated grid_rows x grid_cols CSD grid, using the explicit
    physical (row, col) -> channel-number layout in `channel_map`.
    """
    row_pix = np.linspace(0, grid_rows - 1, n_rows)
    col_pix = np.linspace(0, grid_cols - 1, n_cols)

    # channel -> pixel position, derived from the physical layout
    positions = {}
    for (r, c), ch in channel_map.items():
        positions[ch] = (row_pix[r - 1], col_pix[c - 1])

    return np.array([positions[ch] for ch in acd if ch in positions], dtype=float)


ELECTRODE_POSITIONS = build_electrode_positions(ACD)   # (16, 2) -> (row_px, col_px)

# Minimum spatial tolerance for "near active tissue", in pixel units.
# Half the smallest electrode pitch in pixels (x-pitch is denser than y-pitch here).
_col_pitch_px = (xdims - 1) / (N_COLS_ELEC - 1)
mindistance = _col_pitch_px / 2.0


def depuration_x_distance(electrode_positions: np.ndarray, CM: dict, min_distance: float) -> dict:
    """Remove centres of mass farther than min_distance from any electrode."""
    all_cms = {}
    for key in CM:
        cms_list = []
        for i in range(CM[key].shape[0]):
            CMS = CM[key][i, :2]                      # (x_col, y_row)
            CMS_rc = np.array([CMS[1], CMS[0]])        # convert to (row, col)
            dist = distance_vectors(CMS_rc, electrode_positions)
            if np.any(dist <= min_distance):
                cms_list.append(CM[key][i, :])
        all_cms[key] = np.array(cms_list, dtype=float) if cms_list else np.empty((0, 3), dtype=float)
    return all_cms


def min_cm_weight(CM: dict, electrode_positions: np.ndarray, min_distance: float) -> float:
    """Minimum significant CM weight via thresholding + biggest log-jump."""
    ncm = clean_dictionary(dict(CM))
    ncm = depuration_x_distance(electrode_positions, ncm, min_distance)

    all_weights = [ncm[k][:, 2] for k in ncm if ncm[k].shape[0] > 0]
    if not all_weights:
        return 0.0

    V = np.concatenate(all_weights)
    ts = np.abs(thresholding(V))
    if len(ts) < 2:
        return float(ts[0]) if len(ts) == 1 else 0.0

    diffs = np.diff(np.log(ts + 1e-300))
    its = int(np.argmax(diffs))
    return float(ts[its + 1])


# =====================================================================
# 6. TRAJECTORY LINKING  (from step02.py, unchanged - grid-size agnostic)
# =====================================================================
def weight_selection(Fr: np.ndarray, min_weight: float) -> np.ndarray:
    if Fr.shape[0] == 0:
        return Fr
    mask = np.abs(Fr[:, 2]) > min_weight
    return Fr[mask, :]


def _velocities(Tjs, Locs, fr: int):
    x = Locs[fr - 2, :]
    y = Locs[fr - 1, :]
    delta_loc = distance_coords(x, y)
    delta_t = Tjs[fr - 1] - Tjs[fr - 2]
    V = delta_loc / delta_t if delta_t != 0 else 0.0
    return V, delta_t


def _auxiliary_trajectories(Tj, cms, tiempo, tol_dist, More, t_aux,
                             tol_time, all_tjs, n_ts, T_min):
    if len(Tj) < 5:
        effective_tol = tol_dist
    else:
        locs = Tj[-5:, :2]
        tjs = Tj[-5:, 3]
        arr_vel = []
        for fr in range(2, 6):
            V, _ = _velocities(tjs, locs, fr)
            arr_vel.append(V)
        tjs_sub = tjs[1:]
        X2 = np.column_stack([tjs_sub, np.ones(4)])
        arr_vel = np.asarray(arr_vel, dtype=float)
        try:
            coeff, *_ = np.linalg.lstsq(X2, arr_vel, rcond=None)
        except np.linalg.LinAlgError:
            coeff = np.zeros(2)
        pred_tol = (coeff[0] * (tjs_sub[-1] + 1) + coeff[1]) * 1.5
        effective_tol = max(pred_tol, tol_dist / 5.0)

    Last = Tj[-1, :]
    next_frame = cms.get(tiempo + 1, np.empty((0, 3)))
    if next_frame.shape[0] == 0:
        return Tj, cms, More, t_aux, all_tjs, n_ts

    dist = distance_vectors(Last[:2], next_frame[:, :2])
    closer = int(np.argmin(dist))
    dist_cl = float(dist[closer])

    if dist_cl < effective_tol:
        temporal = np.hstack([next_frame[closer, :], [tiempo + 1]])
        Chain = np.vstack([Tj, temporal])
        mask = np.ones(next_frame.shape[0], dtype=bool)
        mask[closer] = False
        cms[tiempo + 1] = next_frame[mask, :]
        t_aux = 0
    else:
        Chain = Tj
        if t_aux == tol_time:
            More = False
            if len(Tj) > T_min:
                all_tjs[n_ts] = Tj
                n_ts += 1
        t_aux += 1

    return Chain, cms, More, t_aux, all_tjs, n_ts


def trajectories(CM: dict, tol_dist: float, tol_time: int, T_min: int, min_weight: float) -> dict:
    cms = {k: np.array(v, dtype=float) for k, v in CM.items()}
    t_frs = max(cms.keys()) + 1 if cms else 0

    for k in range(1, 6):
        cms[t_frs + k - 1] = np.empty((0, 3), dtype=float)

    all_tjs = {}
    n_ts = 0

    for tiempo in range(t_frs):
        conj_en_fr = weight_selection(cms.get(tiempo, np.empty((0, 3))), min_weight)
        num_conj = conj_en_fr.shape[0]
        if num_conj == 0:
            continue

        for j in range(num_conj):
            t_prim = tiempo
            More = True
            row0 = np.hstack([conj_en_fr[j, :], [tiempo]])
            Tj = row0.reshape(1, -1)
            t_aux = 0
            ultimo_fr = False

            while More and t_aux <= tol_time:
                if t_prim >= t_frs - 1:
                    ultimo_fr = True
                if t_prim < t_frs - 1:
                    conj_en_sig_fr = weight_selection(cms.get(t_prim + 1, np.empty((0, 3))), min_weight)
                    num_conj_sig = conj_en_sig_fr.shape[0]
                else:
                    num_conj_sig = 0
                    t_aux = tol_time

                if num_conj_sig > 0:
                    (Tj, cms, More, t_aux, all_tjs, n_ts) = _auxiliary_trajectories(
                        Tj, cms, t_prim, tol_dist, More, t_aux, tol_time, all_tjs, n_ts, T_min,
                    )
                    if ultimo_fr and len(Tj) > T_min:
                        all_tjs[n_ts] = Tj
                        n_ts += 1
                        More = False
                    t_prim += 1
                else:
                    if t_aux == tol_time:
                        if len(Tj) > T_min:
                            all_tjs[n_ts] = Tj
                            n_ts += 1
                        More = False
                    t_aux += 1
                    t_prim += 1

    return all_tjs


def start_stop(Tjs: dict):
    t0 = {i: float(Tjs[i][0, -1]) for i in Tjs}
    tN = {i: float(Tjs[i][-1, -1]) for i in Tjs}
    return t0, tN


def fixing_gaps(Tjs: dict, Starts: dict, Stops: dict) -> dict:
    for i in Tjs:
        expected = int(Stops[i] - Starts[i]) + 1
        actual = Tjs[i].shape[0]
        if expected == actual:
            continue

        Frs = Tjs[i][:, -1]
        j = 0
        while j < Tjs[i].shape[0] - 1:
            dif = Frs[j + 1] - Frs[j]
            if dif > 1:
                Head = Tjs[i][:j + 1, :]
                Body = Tjs[i][j + 1:, :]
                extras = []
                for k in range(1, int(dif)):
                    K = k / dif
                    h1 = Tjs[i][j, 0] + (Tjs[i][j + 1, 0] - Tjs[i][j, 0]) * K
                    h2 = Tjs[i][j, 1] + (Tjs[i][j + 1, 1] - Tjs[i][j, 1]) * K
                    h3 = Tjs[i][j, 2] + (Tjs[i][j + 1, 2] - Tjs[i][j, 2]) * K
                    h4 = Starts[i] + j + k
                    extras.append([h1, h2, h3, h4])
                Tjs[i] = np.vstack([Head] + extras + [Body])
                Frs = Tjs[i][:, -1]
            j += 1
    return Tjs


# =====================================================================
# 7. RUN THE DETECTION PIPELINE
# =====================================================================
baseline_matrix = csd_data[:, :, 0:baseline_end_sample + 1]
thr = donoho_matrix_3d(baseline_matrix)
eps = thr * sigma_data(baseline_matrix[0, 0, :])   # (101, 101) threshold matrix

print("Computing centers of mass per frame ...")
CMN, CMP = get_centers_of_mass(csd_data, minchannels, eps)   # CMN=sinks, CMP=sources

print("Linking sink (negative) trajectories ...")
min_weight_n = min_cm_weight(CMN, ELECTRODE_POSITIONS, mindistance)
TN = trajectories(CMN, tol_dist, tol_time, Tmin, min_weight_n)
Starts_n, Stops_n = start_stop(TN)
TNf = fixing_gaps(TN, Starts_n, Stops_n)

print("Linking source (positive) trajectories ...")
min_weight_p = min_cm_weight(CMP, ELECTRODE_POSITIONS, mindistance)
TP = trajectories(CMP, tol_dist, tol_time, Tmin, min_weight_p)
Starts_p, Stops_p = start_stop(TP)
TPf = fixing_gaps(TP, Starts_p, Stops_p)

num_sink_waves = len(TNf)
num_source_waves = len(TPf)


# =====================================================================
# 8. KINEMATICS PER TRAJECTORY ("WAVE")
# =====================================================================
def analyze_trajectory_kinematics(Tjs: dict) -> list:
    """
    Tjs: dict {wave_id -> ndarray (M,4)} columns [x_px, y_px, weight, frame]
    Returns a list of metric dicts, one per trajectory/wave.
    """
    wave_metrics = []
    for wave_id in sorted(Tjs.keys()):
        traj = Tjs[wave_id]
        frames = traj[:, 3]
        start_frame = frames[0]
        end_frame = frames[-1]
        duration_ms = len(frames) * (seconds_per_sample * 1000)

        x_mm = traj[:, 0] * mm_per_pixel_x
        y_mm = traj[:, 1] * mm_per_pixel_y
        trajectory_mm = np.column_stack([x_mm, y_mm, frames])

        diffs = np.diff(trajectory_mm[:, 0:2], axis=0)
        step_distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        total_distance_mm = np.sum(step_distances)
        mean_velocity_mm_s = (np.mean(step_distances / seconds_per_sample)
                               if len(step_distances) > 0 else 0.0)

        if len(trajectory_mm) >= 2:
            delta_x = trajectory_mm[-1, 0] - trajectory_mm[0, 0]
            delta_y = trajectory_mm[-1, 1] - trajectory_mm[0, 1]
            angle_degrees = np.degrees(np.arctan2(delta_y, delta_x))
        else:
            angle_degrees = np.nan

        wave_metrics.append({
            "wave_number": wave_id + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_ms": duration_ms,
            "distance_mm": total_distance_mm,
            "velocity_mm_s": mean_velocity_mm_s,
            "angle_deg": angle_degrees,
            "trajectory_coords": trajectory_mm,   # x_mm, y_mm, frame
        })
    return wave_metrics


sink_wave_analysis = analyze_trajectory_kinematics(TNf)
source_wave_analysis = analyze_trajectory_kinematics(TPf)


# =====================================================================
# 9. REPORT GENERATION
# =====================================================================
print("\n" + "=" * 50)
print("CSD MULTI-WAVE BREAKDOWN REPORT")
print("=" * 50)
print(f"Total SINK Waves Identified: {num_sink_waves}")
for wave in sink_wave_analysis:
    print(f"  - Wave #{wave['wave_number']}: Frames {wave['start_frame']:.0f}-{wave['end_frame']:.0f} | "
          f"Duration: {wave['duration_ms']:.1f}ms | Dist: {wave['distance_mm']:.2f}mm | "
          f"Vel: {wave['velocity_mm_s']:.1f}mm/s | Angle: {wave['angle_deg']:.1f}deg")

print("\n" + "-" * 50)
print(f"Total SOURCE Waves Identified: {num_source_waves}")
for wave in source_wave_analysis:
    print(f"  - Wave #{wave['wave_number']}: Frames {wave['start_frame']:.0f}-{wave['end_frame']:.0f} | "
          f"Duration: {wave['duration_ms']:.1f}ms | Dist: {wave['distance_mm']:.2f}mm | "
          f"Vel: {wave['velocity_mm_s']:.1f}mm/s | Angle: {wave['angle_deg']:.1f}deg")
print("=" * 50 + "\n")

# Persist results, mirroring STEP02.ipynb's CMS output structure
Results = {"CMN": CMN, "CMP": CMP, "TN": TNf, "TP": TPf,
           "sink_wave_analysis": sink_wave_analysis,
           "source_wave_analysis": source_wave_analysis}
with open("CSD_CoM_Results.pkl", "wb") as fout:
    pickle.dump({"Data": Results}, fout)
print("Results saved to CSD_CoM_Results.pkl")


# =====================================================================
# 10. ANIMATION GENERATION
# =====================================================================
if SAVE_AND_PLAY_VIDEO:
    # Build per-frame lookup: frame -> list of (wave_id, x_mm, y_mm) active at that frame
    def build_frame_lookup(wave_analysis):
        lookup = {}
        for wave in wave_analysis:
            coords = wave["trajectory_coords"]   # x_mm, y_mm, frame
            for row in coords:
                f = int(round(row[2]))
                lookup.setdefault(f, []).append((wave["wave_number"], row[0], row[1]))
        return lookup

    sink_frame_lookup = build_frame_lookup(sink_wave_analysis)
    source_frame_lookup = build_frame_lookup(source_wave_analysis)

    grid_x = np.linspace(0, grid_physical_x_mm, xdims)
    grid_y = np.linspace(0, grid_physical_y_mm, ydims)
    GRID_X, GRID_Y = np.meshgrid(grid_x, grid_y)

    fig, ax = plt.subplots(figsize=(8, 5))

    v_max = max(np.max(csd_data), abs(np.min(csd_data)))
    mesh = ax.pcolormesh(GRID_X, GRID_Y, csd_data[:, :, 0], cmap='bwr',
                          vmin=-v_max, vmax=v_max, shading='auto')
    fig.colorbar(mesh, ax=ax, label='CSD Intensity')

    sink_scatter = ax.scatter([], [], color='black', marker='o', s=80,
                               edgecolors='white', label='Sink CoM')
    source_scatter = ax.scatter([], [], color='yellow', marker='^', s=80,
                                 edgecolors='black', label='Source CoM')

    ax.set_xlim(0, grid_physical_x_mm)
    ax.set_ylim(0, grid_physical_y_mm)
    ax.set_xlabel("X Profile Range (mm)")
    ax.set_ylabel("Y Profile Range (mm)")

    title_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black', weight='bold')
    stats_text = ax.text(0.02, 0.05, '', transform=ax.transAxes, color='black', fontsize=9,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.legend(loc='upper right')

    def update_frame(t):
        mesh.set_array(csd_data[:, :, t].ravel())
        current_time_ms = t * (seconds_per_sample * 1000)
        title_text.set_text(f"Frame: {t:04d} | Time: {current_time_ms:.1f} ms")

        sink_pts = sink_frame_lookup.get(t, [])
        source_pts = source_frame_lookup.get(t, [])

        if sink_pts:
            sink_scatter.set_offsets(np.array([[p[1], p[2]] for p in sink_pts]))
            sink_id_str = ", ".join(f"#{p[0]}" for p in sink_pts)
        else:
            sink_scatter.set_offsets(np.empty((0, 2)))
            sink_id_str = "Baseline/None"

        if source_pts:
            source_scatter.set_offsets(np.array([[p[1], p[2]] for p in source_pts]))
            source_id_str = ", ".join(f"#{p[0]}" for p in source_pts)
        else:
            source_scatter.set_offsets(np.empty((0, 2)))
            source_id_str = "Baseline/None"

        stats_text.set_text(
            f"Active Sink Wave(s): {sink_id_str}\n"
            f"Active Source Wave(s): {source_id_str}"
        )
        return mesh, sink_scatter, source_scatter, title_text, stats_text

    # ---- Robust writer selection -----------------------------------
    # ffmpeg on Windows is notoriously easy to break (PATH issues,
    # missing DLLs like cairo.dll from conda-forge builds, etc). We
    # try, in order: (1) matplotlib's default ffmpeg discovery via
    # rcParams, (2) imageio_ffmpeg's bundled self-contained binary
    # (no external DLL dependencies), (3) fall back to an animated
    # GIF via PillowWriter so the pipeline never silently loses the
    # video output entirely.
    writer = None
    output_filename = VIDEO_FILENAME

    try:
        import imageio_ffmpeg
        matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Brandon S Coventry'),
                                         bitrate=2000)
        print(f"Using bundled ffmpeg binary: {imageio_ffmpeg.get_ffmpeg_exe()}")
    except Exception as e:
        print(f"imageio_ffmpeg unavailable or failed to configure ({e}).")
        writer = None

    if writer is None:
        try:
            writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Brandon S Coventry'),
                                             bitrate=2000)
            print("Falling back to matplotlib's default ffmpeg discovery.")
        except Exception as e:
            print(f"Default ffmpeg discovery failed too ({e}).")
            writer = None

    print(f"Compiling tracking video: '{output_filename}' across {total_samples} frames...")
    ani = animation.FuncAnimation(fig, update_frame, frames=total_samples, blit=False)

    try:
        if writer is None:
            raise RuntimeError("No ffmpeg writer available.")
        ani.save(output_filename, writer=writer)
        print(f"Video compilation successful. File written to: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"ffmpeg-based save failed ({e}). Falling back to animated GIF via Pillow.")
        gif_filename = os.path.splitext(output_filename)[0] + ".gif"
        pillow_writer = animation.PillowWriter(fps=30)
        ani.save(gif_filename, writer=pillow_writer)
        print(f"GIF compilation successful. File written to: {os.path.abspath(gif_filename)}")

    plt.close(fig)