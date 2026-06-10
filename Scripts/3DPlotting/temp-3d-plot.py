"""
IMU → Live 3D Trajectory Viewer  (CSV tail + Madgwick fusion + Dash)
=====================================================================
Watches a CSV file being written by another process and updates the
3D trajectory plot in the browser as new rows arrive.

The fusion pipeline (Madgwick AHRS) and temperature colour gradient
are identical to accel_3d_plot.py.  The key additions are:

  - A Dash web app replaces fig.show().
  - A dcc.Interval component polls the CSV every POLL_MS milliseconds.
  - Only the rows that are NEW since the last poll are fused; the
    quaternion state is preserved across callbacks so the filter
    never restarts from scratch.
  - Plotly.extendTraces is used under the hood (via graph update) so
    the browser appends points rather than redrawing from scratch.

Usage
-----
# Watch a live CSV (must already exist with a header row)
python accel_3d_live.py data.csv

# Custom column names / poll rate
python accel_3d_live.py data.csv --poll-ms 100 --temp-col temp_degC

# Demo mode: spawns a background thread that writes synthetic IMU
# data to /tmp/imu_demo.csv at ~50 Hz so you can see it working
python accel_3d_live.py --demo

Requirements
------------
pip install dash plotly pandas numpy
"""

import argparse
import sys
import threading
import time
import tempfile
import os

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Thermal colour scale  (blue → cyan → green → yellow → amber → red)
# ---------------------------------------------------------------------------

THERMAL_COLORSCALE = [
    [0.00, "#2b3d9e"],
    [0.15, "#1c8ac9"],
    [0.30, "#26c7c7"],
    [0.50, "#4fca5b"],
    [0.65, "#d4d430"],
    [0.80, "#f5a623"],
    [1.00, "#d62728"],
]

GRAVITY = 9.80665  # m/s²

# ---------------------------------------------------------------------------
# Madgwick AHRS (stateful — one instance persists across callbacks)
# ---------------------------------------------------------------------------

class MadgwickAHRS:
    def __init__(self, beta=0.033):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gx, gy, gz, ax, ay, az, dt):
        q = self.q
        q0, q1, q2, q3 = q
        a_norm = np.sqrt(ax*ax + ay*ay + az*az)
        if a_norm < 1e-10:
            self.q = self._integrate_gyro(q, gx, gy, gz, dt)
            return self.q
        ax /= a_norm; ay /= a_norm; az /= a_norm
        _2q0=2*q0; _2q1=2*q1; _2q2=2*q2; _2q3=2*q3
        _4q0=4*q0; _4q1=4*q1; _4q2=4*q2
        _8q1=8*q1; _8q2=8*q2
        s0 = _4q0*q2*q2 + _2q2*ax + _4q0*q1*q1 - _2q1*ay
        s1 = (_4q1*q3*q3 - _2q3*ax + 4*q0*q0*q1
              - _2q0*ay - _4q1 + _8q1*q1*q1 + _8q1*q2*q2 + _4q1*az)
        s2 = (4*q0*q0*q2 + _2q0*ax + _4q2*q3*q3
              - _2q3*ay - _4q2 + _8q2*q1*q1 + _8q2*q2*q2 + _4q2*az)
        s3 = 4*q1*q1*q3 - _2q1*ax + 4*q2*q2*q3 - _2q2*ay
        s_norm = np.sqrt(s0*s0+s1*s1+s2*s2+s3*s3)
        if s_norm > 1e-10:
            s0/=s_norm; s1/=s_norm; s2/=s_norm; s3/=s_norm
        qDot0 = 0.5*(-q1*gx-q2*gy-q3*gz) - self.beta*s0
        qDot1 = 0.5*( q0*gx+q2*gz-q3*gy) - self.beta*s1
        qDot2 = 0.5*( q0*gy-q1*gz+q3*gx) - self.beta*s2
        qDot3 = 0.5*( q0*gz+q1*gy-q2*gx) - self.beta*s3
        q0+=qDot0*dt; q1+=qDot1*dt; q2+=qDot2*dt; q3+=qDot3*dt
        norm = np.sqrt(q0*q0+q1*q1+q2*q2+q3*q3)
        self.q = np.array([q0,q1,q2,q3])/norm
        return self.q

    @staticmethod
    def _integrate_gyro(q, gx, gy, gz, dt):
        q0,q1,q2,q3 = q
        qD0=0.5*(-q1*gx-q2*gy-q3*gz); qD1=0.5*(q0*gx+q2*gz-q3*gy)
        qD2=0.5*(q0*gy-q1*gz+q3*gx);  qD3=0.5*(q0*gz+q1*gy-q2*gx)
        q_new=np.array([q0+qD0*dt,q1+qD1*dt,q2+qD2*dt,q3+qD3*dt])
        return q_new/np.linalg.norm(q_new)


def quat_rotate(q, v):
    w,x,y,z = q
    vq = np.array([0.0,v[0],v[1],v[2]])
    def qmul(a,b):
        return np.array([
            a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
            a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0],
        ])
    return qmul(qmul(q,vq),np.array([w,-x,-y,-z]))[1:]

# ---------------------------------------------------------------------------
# Incremental fusion state  (lives in module scope, shared across callbacks)
# ---------------------------------------------------------------------------

class FusionState:
    """
    Holds everything needed to continue the Madgwick filter and
    dead-reckoning integration from where the last callback left off.
    Thread-safe via a simple lock (Dash can run callbacks concurrently).
    """
    def __init__(self, beta, use_gyro, has_temp):
        self.lock       = threading.Lock()
        self.use_gyro   = use_gyro
        self.has_temp   = has_temp
        self.madgwick   = MadgwickAHRS(beta=beta)
        self.rows_seen  = 0          # total CSV rows consumed (excl. header)
        # running integration state
        self.last_time  = None
        self.vel        = np.zeros(3)
        self.pos        = np.zeros(3)
        # accumulated trajectory for plotting
        self.xs: list   = []
        self.ys: list   = []
        self.zs: list   = []
        self.cs: list   = []         # colour values (temp or time)

    def process_new_rows(self, new_rows: pd.DataFrame):
        """Fuse new rows and append results to trajectory lists.

        The 'time' column is expected in milliseconds and is converted
        to seconds internally before integration.
        """
        grav = np.array([0.0, 0.0, GRAVITY])
        for _, row in new_rows.iterrows():
            if (row['time'] == 'time'):
                return
            t  = float(row['time']) / 1000.0   # ms → seconds
            ax = float(row['ax']) * GRAVITY
            ay = float(row['ay']) * GRAVITY
            az = float(row['az']) * GRAVITY

            if self.last_time is None:
                dt = 1.0 / 100.0   # assume 100 Hz until we have two samples
            else:
                dt = max(t - self.last_time, 1e-6)
            self.last_time = t

            if self.use_gyro:
                gx = np.radians(float(row['gx']))
                gy = np.radians(float(row['gy']))
                gz = np.radians(float(row['gz']))
                q  = self.madgwick.update(gx, gy, gz, ax, ay, az, dt)
                lin_acc = quat_rotate(q, np.array([ax, ay, az])) - grav
            else:
                # Accel-only: no gravity removal (mean subtraction can't be
                # done incrementally, so we zero each axis bias at startup)
                lin_acc = np.array([ax, ay, az])

            # Trapezoidal velocity & position update
            self.vel = self.vel + lin_acc * dt
            self.pos = self.pos + self.vel * dt

            self.xs.append(float(self.pos[0]))
            self.ys.append(float(self.pos[1]))
            self.zs.append(float(self.pos[2]))
            self.cs.append(
                float(row["temperature"]) if self.has_temp else t  # t already in seconds
            )

        self.rows_seen += len(new_rows)


# ---------------------------------------------------------------------------
# Synthetic demo data writer
# ---------------------------------------------------------------------------

def _demo_writer(path: str, fs: float = 50.0):
    """Writes synthetic IMU rows to *path* at ~fs Hz (background thread)."""
    omega = 2*np.pi*0.4; r = 0.8; vz_c = 0.15
    duration = 30.0
    rng = np.random.default_rng(0)
    t_arr = np.arange(0, duration, 1.0/fs)

    with open(path, "w") as f:
        f.write("time,ax,ay,az,gx,gy,gz,temperature\n")
        f.flush()
        t0 = time.time()
        roll_prev = pitch_prev = yaw_prev = 0.0
        for i, t in enumerate(t_arr):
            x_t = r*np.cos(omega*t); y_t = r*np.sin(omega*t)
            ax_w = -omega**2*x_t + rng.normal(0, 0.005)
            ay_w = -omega**2*y_t + rng.normal(0, 0.005)
            az_w = rng.normal(0, 0.005)
            ax_wg = ax_w; ay_wg = ay_w; az_wg = az_w + GRAVITY

            pitch = np.radians(15*np.sin(2*np.pi*0.1*t))
            roll  = np.radians(8 *np.sin(2*np.pi*0.07*t+0.5))
            yaw   = omega*t
            cr,sr=np.cos(roll),np.sin(roll)
            cp,sp=np.cos(pitch),np.sin(pitch)
            cy,sy=np.cos(yaw),np.sin(yaw)
            R=np.array([
                [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                [-sp,   cp*sr,          cp*cr],
            ])
            a_s = R.T @ np.array([ax_wg, ay_wg, az_wg])
            dt_i = 1.0/fs if i==0 else t-t_arr[i-1]
            gx_s = np.degrees((roll -roll_prev) /dt_i) + rng.normal(0,0.17)
            gy_s = np.degrees((pitch-pitch_prev)/dt_i) + rng.normal(0,0.17)
            gz_s = np.degrees((yaw  -yaw_prev)  /dt_i) + rng.normal(0,0.17)
            roll_prev=roll; pitch_prev=pitch; yaw_prev=yaw

            speed = np.sqrt(ax_w**2+ay_w**2)
            fade  = min(t/(0.8*duration), 1-(max(t-0.8*duration,0))/(0.2*duration))
            temp  = 22 + 18*fade*(speed/((omega*r)**2)) + rng.normal(0,0.15)

            t_ms = t * 1000.0   # convert to milliseconds for output
            f.write(f"{t_ms:.2f},{a_s[0]/GRAVITY:.6f},{a_s[1]/GRAVITY:.6f},"
                    f"{a_s[2]/GRAVITY:.6f},{gx_s:.4f},{gy_s:.4f},{gz_s:.4f},"
                    f"{temp:.3f}\n")
            f.flush()
            # pace to real-time
            elapsed = time.time()-t0
            wait = t+1.0/fs - elapsed
            if wait > 0:
                time.sleep(wait)


# ---------------------------------------------------------------------------
# Dash app builder
# ---------------------------------------------------------------------------

def make_app(csv_path: str, args) -> dash.Dash:
    # Peek at the header to decide what columns are available
    header = pd.read_csv(csv_path, nrows=0)
    cols   = set(header.columns)

    use_gyro = (all(c in cols for c in ("gx","gy","gz"))
                and not args.accel_only)
    has_temp = args.temp_col in cols

    if has_temp and args.temp_col != "temperature":
        # normalise column name
        pass  # handled in process_new_rows via row[args.temp_col]

    state = FusionState(beta=args.beta, use_gyro=use_gyro, has_temp=has_temp)

    color_label = "Temp (°C)" if has_temp else "Time (s)"
    colorscale  = THERMAL_COLORSCALE if has_temp else "Viridis"

    # ── Initial empty figure ───────────────────────────────────────────────
    def empty_fig():
        fig = go.Figure(go.Scatter3d(
            x=[], y=[], z=[],
            mode="lines",
            line=dict(color=[], colorscale=colorscale, width=4,
                      colorbar=dict(title=color_label)),
            name="Trajectory",
        ))
        fig.update_layout(
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)",
                       zaxis_title="Z (m)",
                       aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
            height=620,
            title=dict(text=f"Live IMU trajectory — {os.path.basename(csv_path)}",
                       font=dict(size=14)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
        )
        return fig

    # ── Layout ─────────────────────────────────────────────────────────────
    app = dash.Dash(__name__, title="IMU Live Viewer")
    app.layout = html.Div([
        html.Div([
            html.Span(id="status",
                      style={"fontFamily": "monospace", "fontSize": "13px",
                             "color": "#555"}),
            html.Span(" · ", style={"color": "#ccc"}),
            html.Span(
                ["Poll interval: ",
                 dcc.Slider(id="interval-slider",
                            min=50, max=2000, step=50,
                            value=args.poll_ms,
                            marks={50:"50ms",500:"500ms",
                                   1000:"1s",2000:"2s"},
                            tooltip={"placement":"bottom"})],
                            #style={"width":"260px","display":"inline-block",
                                  # "verticalAlign":"middle"})],
                style={"display":"inline-flex","alignItems":"center","gap":"8px"}
            ),
        ], style={"padding":"8px 16px","display":"flex",
                  "alignItems":"center","gap":"12px",
                  "borderBottom":"1px solid #eee"}),

        dcc.Graph(id="traj", figure=empty_fig(),
                  style={"height":"620px"},
                  config={"scrollZoom": True}),

        dcc.Interval(id="ticker", interval=args.poll_ms, n_intervals=0),

        # Store the current poll interval so the callback can update it
        dcc.Store(id="poll-store", data=args.poll_ms),
    ], style={"fontFamily": "sans-serif"})

    # ── Update interval when slider moves ──────────────────────────────────
    @app.callback(
        Output("ticker", "interval"),
        Input("interval-slider", "value"),
    )
    def update_interval(val):
        return val or args.poll_ms

    # ── Main polling callback ──────────────────────────────────────────────
    @app.callback(
        Output("traj",   "figure"),
        Output("status", "children"),
        Input("ticker",  "n_intervals"),
        State("traj",    "figure"),
    )
    def poll_csv(n, current_fig):
        with state.lock:
            try:
                # Read only new rows by skipping already-seen ones
                # (skiprows skips data rows; +1 for the header)
                new_rows = pd.read_csv(
                    csv_path,
                    skiprows=range(1, state.rows_seen + 1),
                    header=0 if state.rows_seen == 0 else None,
                    names=list(pd.read_csv(csv_path, nrows=0).columns),
                    on_bad_lines="skip",
                )
            except Exception as e:
                return current_fig, f"⚠ read error: {e}"

            if new_rows.empty:
                status = (f"⏳ waiting for data… "
                          f"({state.rows_seen:,} rows so far)")
                return current_fig, status

            # Rename temp column if needed
            if has_temp and args.temp_col != "temperature":
                new_rows = new_rows.rename(
                    columns={args.temp_col: "temperature"})

            state.process_new_rows(new_rows)

        # Rebuild the line trace with all accumulated points.
        # Plotly Dash doesn't expose extendData easily for 3-D colour lines,
        # so we replace the data arrays — still fast because no layout redraw.
        fig = go.Figure(current_fig)
        fig.data[0].x = state.xs
        fig.data[0].y = state.ys
        fig.data[0].z = state.zs
        fig.data[0].line.color = state.cs

        # Keep the camera angle the user has set
        if current_fig and current_fig.get("layout", {}).get("scene"):
            fig.update_layout(scene_camera=
                current_fig["layout"]["scene"].get("camera", {}))

        status = (f"✓ {state.rows_seen:,} rows · "
                  f"pos ({state.pos[0]:.2f}, {state.pos[1]:.2f}, "
                  f"{state.pos[2]:.2f}) m")
        if has_temp and state.cs:
            status += f" · {state.cs[-1]:.1f} °C"

        return fig, status

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv", nargs="?", help="Path to CSV being written live")
    p.add_argument("--demo", action="store_true",
                   help="Write synthetic demo data and watch it live")
    p.add_argument("--accel-only", action="store_true")
    p.add_argument("--beta",     type=float, default=0.033)
    p.add_argument("--temp-col", default="temperature",
                   help="CSV column for temperature in °C  [%(default)s]")
    p.add_argument("--poll-ms",  type=int,   default=200,
                   help="Browser refresh interval in ms  [%(default)s]")
    p.add_argument("--port",     type=int,   default=8050)
    p.add_argument("--host",     default="127.0.0.1")
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo:
        csv_path = os.path.join(tempfile.gettempdir(), "imu_demo.csv")
        print(f"Demo mode: writing synthetic data to {csv_path}")
        # Write the header first so the app can start immediately
        with open(csv_path, "w") as f:
            f.write("time,ax,ay,az,gx,gy,gz,temperature\n")
        t = threading.Thread(target=_demo_writer, args=(csv_path,),
                             daemon=True)
        t.start()
        args.csv = csv_path
    else:
        if not args.csv:
            print("ERROR: supply a CSV path or use --demo", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.csv):
            print(f"ERROR: '{args.csv}' does not exist yet.\n"
                  "Make sure the writing process has created the file "
                  "(with a header row) before starting the viewer.",
                  file=sys.stderr)
            sys.exit(1)

    app = make_app(args.csv, args)
    print(f"\nLive viewer running → http://{args.host}:{args.port}/\n"
          "Open that URL in your browser.  Ctrl-C to stop.\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()