"""
IMU → 3D Cartesian Trajectory Visualiser  (Accelerometer + Gyroscope Fusion)
=============================================================================
Fuses accelerometer (ax, ay, az in m/s²) and gyroscope (gx, gy, gz in rad/s)
readings using a Madgwick AHRS filter to produce an accurate orientation
quaternion at every timestep.  The orientation is used to:

  1. Rotate sensor-frame accelerations into the world frame.
  2. Subtract the gravity vector (no need for --remove-gravity guesswork).
  3. Double-integrate world-frame linear acceleration → position (x, y, z).

Falls back gracefully to accelerometer-only mode if gyroscope columns are
absent from the CSV (or when running with --accel-only).

Input  : CSV with columns  time, ax, ay, az  [, gx, gy, gz]
          - time       : seconds (float, monotonically increasing)
          - ax/ay/az   : acceleration in g  (multiples of 9.80665 m/s², gravity included)
          - gx/gy/gz   : angular velocity in °/s  (optional but recommended)

Output : interactive HTML plot  (and optional PNG via --save-png)

Usage examples
--------------
# Quickstart – generate synthetic IMU data and plot it
python accel_3d_plot.py --demo

# Real data with gyroscope columns
python accel_3d_plot.py data.csv

# Accel-only fallback (old behaviour)
python accel_3d_plot.py data.csv --accel-only

# Tune the Madgwick beta gain and save PNG
python accel_3d_plot.py data.csv --beta 0.033 --save-png output.png
"""

import argparse
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Madgwick AHRS filter (pure NumPy, no external IMU library required)
# ---------------------------------------------------------------------------

class MadgwickAHRS:
    """
    Sebastian Madgwick's gradient-descent AHRS filter.

    Reference: S. O. H. Madgwick, "An efficient orientation filter for
    inertial and inertial/magnetic sensor arrays", 2010.

    Parameters
    ----------
    beta : float
        Filter gain – higher = faster convergence, more noise sensitivity.
        Typical range 0.01 – 0.1.  Default 0.033.
    """

    def __init__(self, beta: float = 0.033):
        self.beta = beta
        # Initial quaternion: identity (no rotation)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gx: float, gy: float, gz: float,
               ax: float, ay: float, az: float,
               dt: float) -> np.ndarray:
        """
        Update orientation with one IMU sample.

        Parameters
        ----------
        gx, gy, gz : gyroscope  (rad/s)
        ax, ay, az : accelerometer  (any consistent unit – will be normalised)
        dt         : time step (seconds)

        Returns
        -------
        Updated quaternion [w, x, y, z] (also stored in self.q).
        """
        q = self.q
        q0, q1, q2, q3 = q

        # Normalise accelerometer; skip update if near-zero (free-fall)
        a_norm = np.sqrt(ax*ax + ay*ay + az*az)
        if a_norm < 1e-10:
            # gyro-only integration
            self.q = self._integrate_gyro(q, gx, gy, gz, dt)
            return self.q
        ax /= a_norm; ay /= a_norm; az /= a_norm

        # Gradient of objective function F_g (eq. 25 Madgwick 2010)
        _2q0 = 2.0 * q0;  _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2;  _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0;  _4q1 = 4.0 * q1;  _4q2 = 4.0 * q2
        _8q1 = 8.0 * q1;  _8q2 = 8.0 * q2

        s0 = _4q0*q2*q2 + _2q2*ax + _4q0*q1*q1 - _2q1*ay
        s1 = (_4q1*q3*q3 - _2q3*ax + 4.0*q0*q0*q1
              - _2q0*ay - _4q1 + _8q1*q1*q1 + _8q1*q2*q2 + _4q1*az)
        s2 = (4.0*q0*q0*q2 + _2q0*ax + _4q2*q3*q3
              - _2q3*ay - _4q2 + _8q2*q1*q1 + _8q2*q2*q2 + _4q2*az)
        s3 = 4.0*q1*q1*q3 - _2q1*ax + 4.0*q2*q2*q3 - _2q2*ay

        s_norm = np.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3)
        if s_norm > 1e-10:
            s0 /= s_norm; s1 /= s_norm; s2 /= s_norm; s3 /= s_norm

        # Rate of change of quaternion from gyroscope
        qDot0 = 0.5 * (-q1*gx - q2*gy - q3*gz) - self.beta * s0
        qDot1 = 0.5 * ( q0*gx + q2*gz - q3*gy) - self.beta * s1
        qDot2 = 0.5 * ( q0*gy - q1*gz + q3*gx) - self.beta * s2
        qDot3 = 0.5 * ( q0*gz + q1*gy - q2*gx) - self.beta * s3

        q0 += qDot0 * dt; q1 += qDot1 * dt
        q2 += qDot2 * dt; q3 += qDot3 * dt

        # Normalise quaternion
        q_norm = np.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        self.q = np.array([q0, q1, q2, q3]) / q_norm
        return self.q

    @staticmethod
    def _integrate_gyro(q: np.ndarray,
                        gx: float, gy: float, gz: float,
                        dt: float) -> np.ndarray:
        q0, q1, q2, q3 = q
        qDot0 = 0.5 * (-q1*gx - q2*gy - q3*gz)
        qDot1 = 0.5 * ( q0*gx + q2*gz - q3*gy)
        qDot2 = 0.5 * ( q0*gy - q1*gz + q3*gx)
        qDot3 = 0.5 * ( q0*gz + q1*gy - q2*gx)
        q_new = np.array([q0 + qDot0*dt, q1 + qDot1*dt,
                          q2 + qDot2*dt, q3 + qDot3*dt])
        return q_new / np.linalg.norm(q_new)


def quaternion_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q using sandwich product q ⊗ v ⊗ q*.
    q : [w, x, y, z]
    v : [x, y, z]
    """
    w, x, y, z = q
    # Convert v to pure quaternion
    vq = np.array([0.0, v[0], v[1], v[2]])

    def qmul(a, b):
        return np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
        ])

    q_conj = np.array([w, -x, -y, -z])
    rotated = qmul(qmul(q, vq), q_conj)
    return rotated[1:]  # drop scalar part


# ---------------------------------------------------------------------------
# Numerical integration helper
# ---------------------------------------------------------------------------

def integrate(values: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (values[:-1] + values[1:]) * dt)])


# ---------------------------------------------------------------------------
# Main fusion pipeline
# ---------------------------------------------------------------------------

GRAVITY = 9.80665  # m/s²


def fuse_imu(
    time: np.ndarray,
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    gx: np.ndarray, gy: np.ndarray, gz: np.ndarray,
    beta: float = 0.033,
) -> pd.DataFrame:
    """
    Madgwick-fused IMU → world-frame position.

    Steps
    -----
    1. Run Madgwick filter to get orientation quaternion per sample.
    2. Rotate sensor-frame acceleration into world frame.
    3. Subtract gravity vector [0, 0, g] from world-frame acceleration.
    4. Double-integrate world-frame linear acceleration → position.

    Returns DataFrame with columns:
        time, ax, ay, az, gx, gy, gz,
        roll, pitch, yaw  (degrees),
        wa_x, wa_y, wa_z  (world-frame linear acceleration),
        vx, vy, vz, x, y, z
    """
    N = len(time)
    madgwick = MadgwickAHRS(beta=beta)

    quaternions = np.zeros((N, 4))
    euler       = np.zeros((N, 3))  # roll, pitch, yaw

    # --- Pass 1: estimate orientation ---
    for i in range(N):
        dt = float(time[i] - time[i-1]) if i > 0 else 1.0 / 100.0
        q = madgwick.update(gx[i], gy[i], gz[i], ax[i], ay[i], az[i], dt)
        quaternions[i] = q

        # Convert quaternion to Euler angles (ZYX convention)
        w, qx, qy, qz = q
        roll  = np.degrees(np.arctan2(2*(w*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
        pitch = np.degrees(np.arcsin (np.clip(2*(w*qy - qz*qx), -1, 1)))
        yaw   = np.degrees(np.arctan2(2*(w*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
        euler[i] = [roll, pitch, yaw]

    # --- Pass 2: rotate accel into world frame and remove gravity ---
    wa = np.zeros((N, 3))
    gravity_world = np.array([0.0, 0.0, GRAVITY])

    for i in range(N):
        sensor_accel = np.array([ax[i], ay[i], az[i]])
        world_accel  = quaternion_rotate(quaternions[i], sensor_accel)
        wa[i] = world_accel - gravity_world

    # --- Pass 3: double-integrate world-frame linear acceleration ---
    dt_arr = np.diff(time)

    vx = integrate(wa[:, 0], dt_arr)
    vy = integrate(wa[:, 1], dt_arr)
    vz = integrate(wa[:, 2], dt_arr)

    px_ = integrate(vx, dt_arr)
    py_ = integrate(vy, dt_arr)
    pz_ = integrate(vz, dt_arr)

    return pd.DataFrame({
        "time":  time,
        "ax": ax, "ay": ay, "az": az,
        "gx": gx, "gy": gy, "gz": gz,
        "roll":  euler[:, 0],
        "pitch": euler[:, 1],
        "yaw":   euler[:, 2],
        "wa_x": wa[:, 0], "wa_y": wa[:, 1], "wa_z": wa[:, 2],
        "vx": vx, "vy": vy, "vz": vz,
        "x": px_, "y": py_, "z": pz_,
    })


def accel_only_position(
    time: np.ndarray,
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    remove_gravity: bool = True,
) -> pd.DataFrame:
    """Accel-only fallback (original behaviour)."""
    if remove_gravity:
        ax = ax - ax.mean()
        ay = ay - ay.mean()
        az = az - az.mean()
    dt = np.diff(time)
    vx = integrate(ax, dt); vy = integrate(ay, dt); vz = integrate(az, dt)
    return pd.DataFrame({
        "time": time,
        "ax": ax, "ay": ay, "az": az,
        "vx": vx, "vy": vy, "vz": vz,
        "x": integrate(vx, dt),
        "y": integrate(vy, dt),
        "z": integrate(vz, dt),
    })


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def generate_demo_data(duration: float = 8.0, fs: float = 100.0) -> pd.DataFrame:
    """
    Synthetic IMU data for a helical ascent trajectory.
    Returns DataFrame with columns: time, ax, ay, az, gx, gy, gz
    """
    t    = np.arange(0, duration, 1.0 / fs)
    omega = 2 * np.pi * 0.4          # 0.4 Hz circular motion
    r     = 0.8                       # metres radius
    vz_c  = 0.15                      # metres/s upward drift

    # True world-frame position
    x_true = r * np.cos(omega * t)
    y_true = r * np.sin(omega * t)
    z_true = vz_c * t

    # True world-frame acceleration (centripetal + gravity)
    ax_w = -omega**2 * x_true
    ay_w = -omega**2 * y_true
    az_w = np.zeros_like(t)           # linear z acc ≈ 0 (constant velocity)

    # Add gravity in world frame before rotating to sensor frame
    ax_w_g = ax_w
    ay_w_g = ay_w
    az_w_g = az_w + GRAVITY

    # Simulate a slowly tilting sensor (pitch oscillates ±15°)
    pitch_true = np.radians(15.0 * np.sin(2 * np.pi * 0.1 * t))
    roll_true  = np.radians(8.0  * np.sin(2 * np.pi * 0.07 * t + 0.5))
    yaw_true   = omega * t  # sensor yaws with the circular motion

    # Build rotation matrices (ZYX) and rotate world → sensor frame
    ax_s = np.zeros_like(t)
    ay_s = np.zeros_like(t)
    az_s = np.zeros_like(t)
    gx_s = np.zeros_like(t)
    gy_s = np.zeros_like(t)
    gz_s = np.zeros_like(t)

    for i in range(len(t)):
        r_val = roll_true[i];  p_val = pitch_true[i];  y_val = yaw_true[i]
        cr, sr = np.cos(r_val), np.sin(r_val)
        cp, sp = np.cos(p_val), np.sin(p_val)
        cy, sy = np.cos(y_val), np.sin(y_val)
        # ZYX rotation matrix (world → sensor = R^T)
        R = np.array([
            [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,     cp*sr,             cp*cr            ],
        ])
        # sensor = R^T @ world
        Rt = R.T
        a_w = np.array([ax_w_g[i], ay_w_g[i], az_w_g[i]])
        a_s = Rt @ a_w
        ax_s[i], ay_s[i], az_s[i] = a_s

        # Gyro = angular velocity in sensor frame (approx from finite diff)
        if i > 0:
            dt_i = t[i] - t[i-1]
            gx_s[i] = (roll_true[i]  - roll_true[i-1])  / dt_i
            gy_s[i] = (pitch_true[i] - pitch_true[i-1]) / dt_i
            gz_s[i] = (yaw_true[i]   - yaw_true[i-1])   / dt_i

    # Convert to user-facing units before adding noise:
    #   acceleration: m/s²  ->  g  (divide by GRAVITY)
    #   angular vel:  rad/s ->  deg/s
    ax_s /= GRAVITY;  ay_s /= GRAVITY;  az_s /= GRAVITY
    gx_s = np.degrees(gx_s)
    gy_s = np.degrees(gy_s)
    gz_s = np.degrees(gz_s)

    rng = np.random.default_rng(42)
    a_noise = 0.005   # g       (~0.05 m/s2)
    g_noise = 0.17    # deg/s   (~0.003 rad/s)
    ax_s += rng.normal(0, a_noise, len(t))
    ay_s += rng.normal(0, a_noise, len(t))
    az_s += rng.normal(0, a_noise, len(t))
    gx_s += rng.normal(0, g_noise, len(t))
    gy_s += rng.normal(0, g_noise, len(t))
    gz_s += rng.normal(0, g_noise, len(t))

    return pd.DataFrame({
        "time": t,
        "ax": ax_s, "ay": ay_s, "az": az_s,   # g
        "gx": gx_s, "gy": gy_s, "gz": gz_s,   # deg/s
    })


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def build_figure(df: pd.DataFrame,
                 title: str = "IMU 3D Trajectory",
                 fused: bool = True) -> go.Figure:
    """3D trajectory coloured by time, with orientation subplot if fused."""

    if fused and all(c in df.columns for c in ("roll", "pitch", "yaw")):
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
            column_widths=[0.65, 0.35],
            subplot_titles=["3D Trajectory", "Orientation (Euler angles)"],
        )

        # --- 3D trajectory trace ---
        fig.add_trace(go.Scatter3d(
            x=df["x"], y=df["y"], z=df["z"],
            mode="lines",
            line=dict(color=df["time"], colorscale="Viridis", width=4,
                      colorbar=dict(title="Time (s)", x=0.62, len=0.8)),
            name="Trajectory", showlegend=False,
        ), row=1, col=1)

        # --- Euler angle traces ---
        for col, colour, name in [
            ("roll",  "#e74c3c", "Roll"),
            ("pitch", "#2ecc71", "Pitch"),
            ("yaw",   "#3498db", "Yaw"),
        ]:
            fig.add_trace(go.Scatter(
                x=df["time"], y=df[col],
                mode="lines", name=name,
                line=dict(color=colour, width=1.5),
            ), row=1, col=2)

        fig.update_layout(
            title=title,
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),

            margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(x=0.66, y=0.95),
            height=550,
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=df["x"], y=df["y"], z=df["z"],
            mode="lines",
            line=dict(color=df["time"], colorscale="Viridis", width=4,
                      colorbar=dict(title="Time (s)")),
            name="Trajectory", showlegend=False,
        ))
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
            margin=dict(l=0, r=0, t=50, b=0),
        )

    # Start / End markers on 3D scene
    for idx, label, colour in [(0, "Start", "limegreen"), (-1, "End", "crimson")]:
        trace = go.Scatter3d(
            x=[df["x"].iloc[idx]],
            y=[df["y"].iloc[idx]],
            z=[df["z"].iloc[idx]],
            mode="markers+text",
            marker=dict(size=8, color=colour),
            text=[label], textposition="top center",
            name=label, showlegend=True,
        )
        if fused:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", help="Path to input CSV file")
    p.add_argument("--demo", action="store_true",
                   help="Use generated synthetic IMU demo data")
    p.add_argument("--accel-only", action="store_true",
                   help="Skip gyroscope fusion even if gyro columns are present")
    p.add_argument("--beta", type=float, default=0.033,
                   help="Madgwick filter gain  [%(default)s]  (0.01–0.1 typical)")

    # Column name overrides
    p.add_argument("--time-col", default="time")
    p.add_argument("--ax-col",   default="ax")
    p.add_argument("--ay-col",   default="ay")
    p.add_argument("--az-col",   default="az")
    p.add_argument("--gx-col",   default="gx")
    p.add_argument("--gy-col",   default="gy")
    p.add_argument("--gz-col",   default="gz")

    p.add_argument("--save-html", default="trajectory.html")
    p.add_argument("--save-png",  default=None,
                   help="Also save a static PNG (requires kaleido)")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open the browser window")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    if args.demo:
        raw = generate_demo_data()
        print("Demo mode: synthetic helical-ascent IMU data.")
    else:
        if not args.csv:
            print("ERROR: provide a CSV file path, or use --demo", file=sys.stderr)
            sys.exit(1)
        raw = pd.read_csv(args.csv)
        raw = raw.rename(columns={
            args.time_col: "time",
            args.ax_col: "ax", args.ay_col: "ay", args.az_col: "az",
            args.gx_col: "gx", args.gy_col: "gy", args.gz_col: "gz",
        })
        for col in ("time", "ax", "ay", "az"):
            if col not in raw.columns:
                print(f"ERROR: column '{col}' not found. Use --{col}-col to remap.",
                      file=sys.stderr)
                sys.exit(1)
        print(f"Loaded {len(raw):,} samples from '{args.csv}'.")

    time = raw["time"].to_numpy(dtype=float)

    # ── Unit conversion ────────────────────────────────────────────────────
    # Input: acceleration in g, angular velocity in deg/s
    # Internal physics: m/s² and rad/s
    accel_cols = (raw["ax"].to_numpy(float) * GRAVITY,
                  raw["ay"].to_numpy(float) * GRAVITY,
                  raw["az"].to_numpy(float) * GRAVITY)

    has_gyro = all(c in raw.columns for c in ("gx", "gy", "gz"))
    use_fusion = has_gyro and not args.accel_only

    # ── Fusion or fallback ─────────────────────────────────────────────────
    if use_fusion:
        print(f"Gyroscope columns detected — running Madgwick fusion (beta={args.beta}).")
        gyro_cols = (np.radians(raw["gx"].to_numpy(float)),
                     np.radians(raw["gy"].to_numpy(float)),
                     np.radians(raw["gz"].to_numpy(float)))
        df = fuse_imu(time, *accel_cols, *gyro_cols, beta=args.beta)
        mode_label = f"Madgwick fusion (β={args.beta})"
    else:
        reason = "--accel-only flag" if args.accel_only else "no gyro columns found"
        print(f"Accel-only mode ({reason}) — subtracting mean gravity from each axis.")
        df = accel_only_position(time, *accel_cols, remove_gravity=True)
        mode_label = "Accel-only (mean gravity removed)"

    print(f"Position range  X: [{df['x'].min():.3f}, {df['x'].max():.3f}] m")
    print(f"                Y: [{df['y'].min():.3f}, {df['y'].max():.3f}] m")
    print(f"                Z: [{df['z'].min():.3f}, {df['z'].max():.3f}] m")

    # ── Plot ───────────────────────────────────────────────────────────────
    source = "Demo data" if args.demo else args.csv
    fig = build_figure(df,
                       title=f"3D Trajectory — {source}  [{mode_label}]",
                       fused=use_fusion)

    fig.write_html(args.save_html)
    print(f"Interactive plot saved → {args.save_html}")

    if args.save_png:
        try:
            fig.write_image(args.save_png)
            print(f"Static PNG saved      → {args.save_png}")
        except Exception as e:
            print(f"PNG export failed (pip install kaleido): {e}", file=sys.stderr)

    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()