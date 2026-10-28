import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="EV Motor Digital Twin", page_icon="⚡", layout="wide")

st.title("⚡ EV Motor Digital Twin & AI/ML Monitoring")
st.caption("Software-only prototype • Simulated data only • Not connected to a physical EV motor")

with st.sidebar:
    st.header("Simulation controls")
    n_points = st.slider("Number of time samples", 100, 1000, 300, 50)
    anomaly_strength = st.slider("Anomaly strength", 1.0, 3.5, 2.0, 0.1)
    anomaly_fraction = st.slider("Percentage of samples with injected anomalies", 1, 20, 8)
    random_seed = st.number_input("Random seed", min_value=0, max_value=10000, value=42, step=1)
    run_sim = st.button("🔄 Generate / refresh simulation", type="primary")

# Keep results stable until controls change or button is pressed.
@st.cache_data
def make_data(n, strength, fraction, seed, run_id):
    rng = np.random.default_rng(seed + run_id)
    t = np.arange(n)
    setpoint = 1000 + 80 * np.sin(2 * np.pi * t / 100)
    speed = setpoint + rng.normal(0, 18, n)
    current = 1.4 + 0.00035 * np.abs(setpoint - 900) + rng.normal(0, 0.08, n)
    temperature = 30 + 0.025 * t + 1.2 * np.sin(2 * np.pi * t / 180) + rng.normal(0, 0.5, n)
    voltage = 48 + rng.normal(0, 0.35, n)

    injected = np.zeros(n, dtype=bool)
    count = max(1, int(n * fraction / 100))
    # Place anomalies in a later segment so the demonstration is easy to see.
    start = max(0, int(n * 0.65))
    candidates = np.arange(start, n)
    if len(candidates) < count:
        candidates = np.arange(n)
    idx = rng.choice(candidates, size=min(count, len(candidates)), replace=False)
    injected[idx] = True
    speed[idx] -= rng.uniform(100, 180, len(idx)) * strength
    current[idx] += rng.uniform(0.5, 1.1, len(idx)) * strength
    temperature[idx] += rng.uniform(3, 7, len(idx)) * strength
    voltage[idx] -= rng.uniform(1, 3, len(idx)) * min(strength, 2.5)
    df = pd.DataFrame({
        "Time (s)": t,
        "Commanded speed (RPM)": np.round(setpoint, 1),
        "Speed (RPM)": np.round(speed, 1),
        "Current (A)": np.round(current, 3),
        "Temperature (°C)": np.round(temperature, 2),
        "DC bus voltage (V)": np.round(voltage, 2),
        "Injected anomaly (simulation ground truth)": injected
    })
    return df

if "run_id" not in st.session_state:
    st.session_state.run_id = 0
if run_sim:
    st.session_state.run_id += 1

df = make_data(n_points, anomaly_strength, anomaly_fraction, int(random_seed), st.session_state.run_id)

# Simple software Digital Twin: expected speed follows the commanded speed,
# with nominal current/temperature baselines estimated from early simulated samples.
baseline_n = max(30, int(len(df) * 0.35))
baseline = df.iloc[:baseline_n]
df["Twin predicted speed (RPM)"] = df["Commanded speed (RPM)"]
df["Speed residual (RPM)"] = df["Speed (RPM)"] - df["Twin predicted speed (RPM)"]
df["Twin baseline current (A)"] = baseline["Current (A)"].median()
df["Twin baseline temperature (°C)"] = (
    baseline["Temperature (°C)"].median()
    + 0.025 * (df["Time (s)"] - baseline["Time (s)"].median())
)
df["Current residual (A)"] = df["Current (A)"] - df["Twin baseline current (A)"]
df["Temperature residual (°C)"] = df["Temperature (°C)"] - df["Twin baseline temperature (°C)"]

features = df[["Speed (RPM)", "Current (A)", "Temperature (°C)", "DC bus voltage (V)"]]
train = features.iloc[:baseline_n]
model = IsolationForest(n_estimators=150, contamination=0.08, random_state=7)
model.fit(train)
df["AI/ML anomaly"] = model.predict(features) == -1
df["Anomaly score"] = -model.decision_function(features)

anomaly_count = int(df["AI/ML anomaly"].sum())
injected_count = int(df["Injected anomaly (simulation ground truth)"].sum())
flagged_injected = int((df["AI/ML anomaly"] & df["Injected anomaly (simulation ground truth)"]).sum())
precision_like = flagged_injected / max(1, anomaly_count)
recall_like = flagged_injected / max(1, injected_count)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest speed", f"{df['Speed (RPM)'].iloc[-1]:.0f} RPM")
c2.metric("Latest current", f"{df['Current (A)'].iloc[-1]:.2f} A")
c3.metric("Latest temperature", f"{df['Temperature (°C)'].iloc[-1]:.1f} °C")
latest_flag = bool(df["AI/ML anomaly"].iloc[-1])
c4.metric("Latest AI/ML status", "⚠️ Anomaly" if latest_flag else "✅ Normal")

st.info("All readings are generated in software. The Digital Twin here is an educational baseline model; it is not a validated EV motor model.")
tab1, tab2, tab3, tab4 = st.tabs(["Live-style dashboard", "Digital Twin comparison", "AI/ML results", "Data table & export"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Time (s)"], y=df["Speed (RPM)"], name="Simulated speed", mode="lines"))
    fig.add_trace(go.Scatter(x=df["Time (s)"], y=df["Commanded speed (RPM)"], name="Commanded speed", mode="lines"))
    fig.update_layout(title="Motor speed over time", xaxis_title="Time (s)", yaxis_title="Speed (RPM)", height=400, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=df["Time (s)"], y=df["Current (A)"], name="Current", mode="lines"))
        f2.update_layout(title="Motor current", xaxis_title="Time (s)", yaxis_title="Current (A)", height=330)
        st.plotly_chart(f2, use_container_width=True)
    with right:
        f3 = go.Figure()
        f3.add_trace(go.Scatter(x=df["Time (s)"], y=df["Temperature (°C)"], name="Temperature", mode="lines"))
        f3.update_layout(title="Motor temperature", xaxis_title="Time (s)", yaxis_title="Temperature (°C)", height=330)
        st.plotly_chart(f3, use_container_width=True)

with tab2:
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=df["Time (s)"], y=df["Speed (RPM)"], name="Measured/simulated speed", mode="lines"))
    f4.add_trace(go.Scatter(x=df["Time (s)"], y=df["Twin predicted speed (RPM)"], name="Twin predicted speed", mode="lines"))
    f4.update_layout(title="Measured vs Digital Twin predicted speed", xaxis_title="Time (s)", yaxis_title="RPM", height=420)
    st.plotly_chart(f4, use_container_width=True)

    f5 = go.Figure()
    f5.add_trace(go.Scatter(x=df["Time (s)"], y=df["Speed residual (RPM)"], name="Speed residual", mode="lines"))
    f5.add_hline(y=0, line_dash="dash")
    f5.update_layout(title="Digital Twin speed residual", xaxis_title="Time (s)", yaxis_title="Actual − predicted (RPM)", height=330)
    st.plotly_chart(f5, use_container_width=True)
    st.write("Residual = simulated measurement − Digital Twin prediction. Large or persistent residuals may indicate a deviation; they do not prove a specific failure.")

with tab3:
    a, b, c = st.columns(3)
    a.metric("AI/ML samples flagged", f"{anomaly_count} / {len(df)}")
    b.metric("Injected simulated anomalies", injected_count)
    c.metric("Flagged injected samples", f"{flagged_injected} / {injected_count}")
    st.caption(f"Illustrative overlap score: {precision_like:.2f} of flagged samples were injected anomalies; {recall_like:.2f} of injected anomalies were flagged. This is a demonstration metric on synthetic data, not real-world model validation.")
    display_df = df[df["AI/ML anomaly"]].copy()
    display_df["Injected anomaly?"] = display_df["Injected anomaly (simulation ground truth)"].map({True: "Yes", False: "No"})
    st.subheader("AI/ML flagged samples")
    st.dataframe(display_df[["Time (s)", "Speed (RPM)", "Current (A)", "Temperature (°C)", "DC bus voltage (V)", "Anomaly score", "Injected anomaly?"]], use_container_width=True)
    f6 = go.Figure()
    f6.add_trace(go.Scatter(x=df["Time (s)"], y=df["Anomaly score"], name="Anomaly score", mode="lines"))
    f6.add_hline(y=0, line_dash="dash")
    f6.update_layout(title="Isolation Forest anomaly score", xaxis_title="Time (s)", yaxis_title="Score (higher means more unusual)", height=350)
    st.plotly_chart(f6, use_container_width=True)

with tab4:
    st.dataframe(df, use_container_width=True, height=420)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download simulation data (CSV)", data=csv, file_name="ev_motor_digital_twin_simulation.csv", mime="text/csv")
    st.download_button("⬇️ Download AI/ML flagged rows (CSV)", data=df[df["AI/ML anomaly"]].to_csv(index=False).encode("utf-8"), file_name="ev_motor_anomalies.csv", mime="text/csv")

st.subheader("How to explain this in your demo")
st.markdown("""
1. Explain that sensor values are simulated because this version is software-only.
2. Show normal speed/current/temperature graphs.
3. Open **Digital Twin comparison** and explain the speed residual.
4. Open **AI/ML results** and show the Isolation Forest flags.
5. Download the CSV to demonstrate stored telemetry and event analysis.
""")
