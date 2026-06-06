import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="UAV Telemetry Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .metric-container { background: #f8f9fa; border-radius: 8px; padding: 12px; }
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }
    .status-normal { color: #198754; font-weight: 600; }
    .status-warning { color: #ffc107; font-weight: 600; }
    .status-danger { color: #dc3545; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    div[data-testid="stMetricDelta"] { font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ── Demo data generators ──────────────────────────────────────────────────────

def make_flight_log(n=60):
    """Generate n rows of fake telemetry at 1-second intervals."""
    now = datetime.now()
    timestamps = [now - timedelta(seconds=n - i) for i in range(n)]
    t = np.linspace(0, n, n)

    altitude   = np.round(100 + np.sin(t * 0.15) * 20 + t * 0.8 + np.random.normal(0, 0.5, n), 1)
    airspeed   = np.round(65  + np.sin(t * 0.2)  * 8  + np.random.normal(0, 0.3, n), 1)
    pitch      = np.round(np.sin(t * 0.25) * 6   + np.random.normal(0, 0.2, n), 2)
    roll       = np.round(np.cos(t * 0.2)  * 5   + np.random.normal(0, 0.2, n), 2)
    heading    = np.round(np.mod(180 + t * 0.5, 360), 1)
    throttle   = np.round(np.clip(55 + np.sin(t * 0.1) * 15, 30, 95), 1)

    # flight mode changes at fixed points for demo
    modes = ["Manual" if i < 20 else ("AI Assist" if i < 45 else "Manual") for i in range(n)]

    # status based on pitch/roll exceedance
    def get_status(p, r):
        if abs(p) > 10 or abs(r) > 10:
            return "Envelope limit"
        if abs(p) > 7 or abs(r) > 7:
            return "Warning"
        return "Normal"

    statuses = [get_status(pitch[i], roll[i]) for i in range(n)]

    return pd.DataFrame({
        "timestamp":   [t.strftime("%H:%M:%S") for t in timestamps],
        "altitude_m":  altitude,
        "airspeed_kmh":airspeed,
        "pitch_deg":   pitch,
        "roll_deg":    roll,
        "heading_deg": heading,
        "throttle_pct":throttle,
        "flight_mode": modes,
        "status":      statuses,
    })


def make_reward_log(episodes=60):
    """Generate fake RL reward history per episode."""
    ep = np.arange(1, episodes + 1)
    r_task      = np.round(np.clip(0.2 + ep * 0.008 + np.random.normal(0, 0.04, episodes), 0, 1), 3)
    r_smooth    = np.round(np.clip(0.3 + ep * 0.006 + np.random.normal(0, 0.05, episodes), 0, 1), 3)
    r_safety    = np.round(np.clip(0.4 + ep * 0.005 + np.random.normal(0, 0.03, episodes), 0, 1), 3)
    w1, w2, w3  = 0.4, 0.3, 0.3
    total       = np.round(w1 * r_task + w2 * r_smooth + w3 * r_safety, 3)
    violations  = np.maximum(0, np.round(10 - ep * 0.12 + np.random.normal(0, 0.5, episodes)).astype(int))

    return pd.DataFrame({
        "episode":    ep,
        "r_task":     r_task,
        "r_smooth":   r_smooth,
        "r_safety":   r_safety,
        "total_reward": total,
        "violations": violations,
    })


# ── Plotly chart helpers ──────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    margin=dict(l=8, r=8, t=8, b=8),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False),
)


def line_chart(df, x, ys, colors, names, height=200, y_label=""):
    fig = go.Figure()
    for y, color, name in zip(ys, colors, names):
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y], name=name,
            line=dict(color=color, width=1.8),
            mode="lines", hovertemplate=f"{name}: %{{y}}<extra></extra>"
        ))
    fig.update_layout(**CHART_LAYOUT, height=height, yaxis_title=y_label)
    fig.update_xaxes(tickangle=-30, nticks=8)
    return fig


def reward_area_chart(df, height=200):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["episode"], y=df["total_reward"],
        fill="tozeroy", name="Total reward",
        line=dict(color="#7F77DD", width=1.8),
        fillcolor="rgba(127,119,221,0.15)",
        hovertemplate="Episode %{x}: %{y}<extra></extra>"
    ))
    fig.update_layout(**CHART_LAYOUT, height=height)
    return fig


def component_bar_chart(df, height=200):
    last = df.iloc[-1]
    components = ["R_task", "R_smooth", "R_safety"]
    values     = [last["r_task"], last["r_smooth"], last["r_safety"]]
    colors     = ["#1D9E75", "#378ADD", "#7F77DD"]
    fig = go.Figure(go.Bar(
        x=components, y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y}<extra></extra>"
    ))
    # build layout without yaxis first, then override yaxis separately
    layout = {k: v for k, v in CHART_LAYOUT.items() if k != "yaxis"}
    fig.update_layout(**layout, height=height)
    fig.update_yaxes(range=[0, 1.1], showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def violations_chart(df, height=200):
    fig = go.Figure(go.Scatter(
        x=df["episode"], y=df["violations"],
        fill="tozeroy", name="Violations",
        line=dict(color="#E24B4A", width=1.5),
        fillcolor="rgba(226,75,74,0.12)",
        hovertemplate="Episode %{x}: %{y} violations<extra></extra>"
    ))
    fig.update_layout(**CHART_LAYOUT, height=height, yaxis_title="Count")
    return fig


def status_color(s):
    if s == "Normal":
        return "🟢"
    if s == "Warning":
        return "🟡"
    return "🔴"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ✈️ UAV Dashboard")
    st.markdown("---")

    st.markdown("**Data source**")
    data_source = st.radio(
        "Mode",
        ["Demo (fake data)", "Load CSV file"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    if data_source == "Load CSV file":
        uploaded_file = st.file_uploader(
            "Upload flight_log.csv", type=["csv"],
            help="CSV must have columns: timestamp, altitude_m, airspeed_kmh, pitch_deg, roll_deg, flight_mode, status"
        )

    st.markdown("---")
    st.markdown("**Display settings**")
    show_n = st.slider("Rows in telemetry view", 10, 60, 30)
    auto_refresh = st.checkbox("Auto-refresh demo (5s)", value=False)

    st.markdown("---")
    st.markdown("**Reward weights**")
    w1 = st.slider("w1 — R_task",   0.0, 1.0, 0.4, 0.05)
    w2 = st.slider("w2 — R_smooth", 0.0, 1.0, 0.3, 0.05)
    w3 = st.slider("w3 — R_safety", 0.0, 1.0, 0.3, 0.05)
    total_w = round(w1 + w2 + w3, 2)
    if total_w != 1.0:
        st.warning(f"Weights sum to {total_w} (should be 1.0)")
    else:
        st.success("Weights sum to 1.0 ✓")

    st.markdown("---")
    st.caption("Capstone I — UAV Co-pilot Project\nRija Bhomi · Data Architect")


# ── Load data ─────────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(0.05)
    st.rerun()

if data_source == "Load CSV file" and uploaded_file:
    flight_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {len(flight_df)} rows")
else:
    flight_df = make_flight_log(60)

reward_df = make_reward_log(60)
# recalculate total with sidebar weights
reward_df["total_reward"] = np.round(
    w1 * reward_df["r_task"] + w2 * reward_df["r_smooth"] + w3 * reward_df["r_safety"], 3
)

latest = flight_df.iloc[-1]
prev   = flight_df.iloc[-2]


# ── Top bar ───────────────────────────────────────────────────────────────────

col_title, col_badges = st.columns([2, 3])
with col_title:
    st.markdown("## UAV Telemetry & AI Monitoring")
with col_badges:
    mode = latest["flight_mode"]
    mode_color = "#ffc107" if mode == "AI Assist" else "#0d6efd"
    s = latest["status"]
    sc = {"Normal": "#198754", "Warning": "#ffc107", "Envelope limit": "#dc3545"}.get(s, "#6c757d")
    st.markdown(f"""
    <div style='display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap;'>
      <div style='background:{mode_color}22;border:1px solid {mode_color}55;border-radius:8px;
        padding:7px 16px;white-space:nowrap;'>
        <span style='font-size:13px;font-weight:600;color:{mode_color};'>✈ {mode}</span>
      </div>
      <div style='background:{sc}22;border:1px solid {sc}55;border-radius:8px;
        padding:7px 16px;white-space:nowrap;'>
        <span style='font-size:13px;font-weight:600;color:{sc};'>● {s}</span>
      </div>
      <div style='background:#6c757d22;border:1px solid #6c757d44;border-radius:8px;
        padding:7px 16px;white-space:nowrap;'>
        <span style='font-size:13px;color:#aaa;'>🕐 {latest["timestamp"]}</span>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════
# SECTION 1 — LIVE FLIGHT TELEMETRY
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📡 Live flight telemetry</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Altitude",  f"{latest['altitude_m']} m",    f"{round(latest['altitude_m']-prev['altitude_m'],1)} m")
m2.metric("Airspeed",  f"{latest['airspeed_kmh']} km/h", f"{round(latest['airspeed_kmh']-prev['airspeed_kmh'],1)}")
m3.metric("Pitch",     f"{latest['pitch_deg']}°",       f"{round(latest['pitch_deg']-prev['pitch_deg'],2)}")
m4.metric("Roll",      f"{latest['roll_deg']}°",        f"{round(latest['roll_deg']-prev['roll_deg'],2)}")
m5.metric("Heading",   f"{latest['heading_deg']}°")
m6.metric("Throttle",  f"{latest['throttle_pct']}%")

recent_df = flight_df.tail(show_n)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Altitude over time**")
    st.plotly_chart(
        line_chart(recent_df, "timestamp", ["altitude_m"], ["#378ADD"], ["Altitude (m)"], y_label="m"),
        use_container_width=True
    )
with c2:
    st.markdown("**Pitch & Roll over time**")
    st.plotly_chart(
        line_chart(recent_df, "timestamp", ["pitch_deg", "roll_deg"],
                   ["#1D9E75", "#D4537E"], ["Pitch", "Roll"], y_label="degrees"),
        use_container_width=True
    )

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Airspeed over time**")
    st.plotly_chart(
        line_chart(recent_df, "timestamp", ["airspeed_kmh"], ["#BA7517"], ["Airspeed (km/h)"], y_label="km/h"),
        use_container_width=True
    )
with c4:
    st.markdown("**Throttle over time**")
    st.plotly_chart(
        line_chart(recent_df, "timestamp", ["throttle_pct"], ["#7F77DD"], ["Throttle (%)"], y_label="%"),
        use_container_width=True
    )

# safety alert banner
n_warnings   = (flight_df["status"] == "Warning").sum()
n_violations = (flight_df["status"] == "Envelope limit").sum()
if n_violations > 0:
    st.error(f"⚠️  {n_violations} envelope limit breach(es) detected in this session. Review flight log below.")
elif n_warnings > 0:
    st.warning(f"🟡  {n_warnings} warning event(s) in this session. Check pitch/roll values.")
else:
    st.success("✅  All flight parameters within safe envelope — no warnings.")

st.markdown("---")


# ══════════════════════════════════════════════════════════════
# SECTION 2 — AI CO-PILOT MONITORING
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🤖 AI co-pilot monitoring</div>', unsafe_allow_html=True)

ai1, ai2, ai3, ai4 = st.columns(4)
last_reward = reward_df.iloc[-1]
ai1.metric("Total reward (latest ep)",   f"{last_reward['total_reward']:.3f}",
           f"{round(last_reward['total_reward'] - reward_df.iloc[-2]['total_reward'], 3)}")
ai2.metric("Episodes completed",         f"{int(reward_df['episode'].max())}")
ai3.metric("Envelope violations (total)",f"{int(reward_df['violations'].sum())}")
ai4.metric("Convergence trend",
           "Improving" if reward_df["total_reward"].iloc[-5:].mean() > reward_df["total_reward"].iloc[:5].mean() else "Needs tuning")

r1, r2 = st.columns(2)
with r1:
    st.markdown("**Reward convergence over episodes**")
    st.plotly_chart(reward_area_chart(reward_df), use_container_width=True)
with r2:
    st.markdown("**Reward components — latest episode**")
    st.plotly_chart(component_bar_chart(reward_df), use_container_width=True)

r3, r4 = st.columns(2)
with r3:
    st.markdown("**Envelope violations per episode**")
    st.plotly_chart(violations_chart(reward_df), use_container_width=True)
with r4:
    st.markdown("**All reward components over episodes**")
    st.plotly_chart(
        line_chart(reward_df, "episode",
                   ["r_task", "r_smooth", "r_safety"],
                   ["#1D9E75", "#378ADD", "#7F77DD"],
                   ["R_task", "R_smooth", "R_safety"], y_label="reward"),
        use_container_width=True
    )

# reward weight summary
st.info(f"Current reward weights → w1 (R_task): **{w1}**  ·  w2 (R_smooth): **{w2}**  ·  w3 (R_safety): **{w3}**  — adjust in sidebar to experiment.")

st.markdown("---")


# ══════════════════════════════════════════════════════════════
# SECTION 3 — FLIGHT LOG TABLE
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📋 Flight log</div>', unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns(3)
with fc1:
    mode_filter = st.multiselect("Filter by flight mode", options=flight_df["flight_mode"].unique().tolist(),
                                  default=flight_df["flight_mode"].unique().tolist())
with fc2:
    status_filter = st.multiselect("Filter by status", options=flight_df["status"].unique().tolist(),
                                    default=flight_df["status"].unique().tolist())
with fc3:
    st.markdown("")
    st.markdown("")
    show_all = st.checkbox("Show all rows", value=False)

filtered = flight_df[
    flight_df["flight_mode"].isin(mode_filter) &
    flight_df["status"].isin(status_filter)
]
if not show_all:
    filtered = filtered.tail(20)

display_df = filtered.copy()
display_df.insert(0, "", display_df["status"].map(status_color))
display_df = display_df.rename(columns={
    "timestamp": "Time",
    "altitude_m": "Alt (m)",
    "airspeed_kmh": "Speed (km/h)",
    "pitch_deg": "Pitch (°)",
    "roll_deg": "Roll (°)",
    "heading_deg": "Heading (°)",
    "throttle_pct": "Throttle (%)",
    "flight_mode": "Mode",
    "status": "Status"
})
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════
# SECTION 4 — SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📊 Session summary</div>', unsafe_allow_html=True)

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Max altitude",    f"{flight_df['altitude_m'].max()} m")
s2.metric("Avg airspeed",    f"{round(flight_df['airspeed_kmh'].mean(), 1)} km/h")
s3.metric("Max pitch",       f"{flight_df['pitch_deg'].abs().max()}°")
s4.metric("Max roll",        f"{flight_df['roll_deg'].abs().max()}°")
s5.metric("Total log rows",  len(flight_df))

# download button
csv_data = flight_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️  Download flight log as CSV",
    data=csv_data,
    file_name=f"flight_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)