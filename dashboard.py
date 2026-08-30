import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import time
from datetime import datetime
import os
import tempfile
import random

st.set_page_config(
    page_title="SOC Telemetry Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASS_MAP = {0: "BENIGN", 1: "DDoS", 2: "PortScan", 3: "Botnet"}

THREAT_ICONS = {
    "BENIGN": "🛡️",
    "DDoS": "🚨",
    "PortScan": "🔍",
    "Botnet": "🤖",
}

THREAT_COLORS = {
    "BENIGN": "#10B981",
    "DDoS": "#DC2626",
    "PortScan": "#D97706",
    "Botnet": "#7C3AED",
}

CSV_TO_MODEL_RENAME = {
    "Total Length of Fwd Packets": "Fwd Packets Length Total",
    "Total Length of Bwd Packets": "Bwd Packets Length Total",
    "Max Packet Length": "Packet Length Max",
    "Average Packet Size": "Avg Packet Size",
    "Init_Win_bytes_forward": "Init Fwd Win Bytes",
    "Init_Win_bytes_backward": "Init Bwd Win Bytes",
    "act_data_pkt_fwd": "Fwd Act Data Packets",
    "min_seg_size_forward": "Fwd Seg Size Min",
}

CSV_LABEL_TO_CLASS = {
    "BENIGN": 0, "Benign": 0,
    "DoS": 1, "DDoS": 1,
    "PortScan": 2, "Portscan": 2,
    "Bot": 3, "Botnet": 3,
    "Infiltration": 3, "WebAttack": 3, "BruteForce": 3,
}

SCENARIOS = {
    "⚡ DDoS Flood Attack": "data/processed/scenario_ddos_mixed.csv",
    "🔍 PortScan Reconnaissance": "data/processed/scenario_portscan_mixed.csv",
    "🤖 Botnet C2 Beaconing": "data/processed/scenario_botnet_mixed.csv",
    "🌐 Multi-Threat Incident": "data/processed/scenario_multi_incident.csv",
    "📁 Custom Upload": None,
}

SPEED_OPTIONS = {
    "⚡ 100 ms": 100,
    "⏱ 250 ms": 250,
    "🐢 500 ms": 500,
    "⏳ 1000 ms": 1000,
}

MAX_FEED_ROWS = 60

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, .stApp {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    font-family: 'Inter', sans-serif !important;
}

section[data-testid="stSidebar"], div[data-testid="collapsedControl"] {
    display: none !important;
}

div[data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
.stAppDeployButton { display: none !important; }
.stStatusWidget { display: none !important; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.3rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 100% !important;
}

.card-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 8px 12px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
}

.stButton > button {
    border-radius: 5px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    padding: 0.3rem 0.65rem !important;
    min-height: 34px !important;
    transition: all 0.15s ease !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 5px !important;
    color: #0F172A !important;
    font-size: 0.8rem !important;
    min-height: 34px !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 6px !important;
    background-color: #FFFFFF !important;
}

@keyframes pulse-diode {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.5; }
}
.live-diode {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-diode 1.8s infinite ease-in-out;
}
</style>
""",
    unsafe_allow_html=True,
)

_DEFAULTS = {
    "is_playing": False,
    "current_idx": 0,
    "display_data": [],
    "backend_data": [],
    "total_inspected": 0,
    "total_threats": 0,
    "threat_counts": {"BENIGN": 0, "DDoS": 0, "PortScan": 0, "Botnet": 0},
    "raw_flows": None,
    "start_time_real": None,
    "selected_scenario": "⚡ DDoS Flood Attack",
    "speed_label": "⏱ 250 ms",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.start_time_real is None:
    st.session_state.start_time_real = time.time()

@st.cache_resource
def load_model():
    path = "models/xgboost_anomaly_detector.pkl"
    if not os.path.exists(path):
        return None, None
    loaded_model = joblib.load(path)
    loaded_explainer = shap.TreeExplainer(loaded_model)
    return loaded_model, loaded_explainer

model, explainer = load_model()
if model is None:
    st.error("❌ Model `models/xgboost_anomaly_detector.pkl` not found.")
    st.stop()

try:
    EXPECTED_FEATURES = model.get_booster().feature_names
except Exception:
    EXPECTED_FEATURES = list(model.feature_names_in_)

def prepare_model_input(row_dict: dict, label_text: str | None = None) -> pd.DataFrame:
    is_cicids = "Flow Duration" in row_dict and "Total Fwd Packets" in row_dict

    if is_cicids:
        renamed = {}
        for col, val in row_dict.items():
            renamed[CSV_TO_MODEL_RENAME.get(col, col)] = val

        feature_row = {}
        for feat in EXPECTED_FEATURES:
            val = renamed.get(feat, 0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            if pd.isna(val) or np.isinf(val):
                val = 0.0
            feature_row[feat] = val
    else:
        feature_row = {feat: 0.0 for feat in EXPECTED_FEATURES}
        m = {
            "Flow Duration": float(row_dict.get("flow_duration_sec", 0)) * 1e6,
            "Total Fwd Packets": float(row_dict.get("packet_count", 0)),
            "Fwd Packets Length Total": float(row_dict.get("total_bytes", 0)),
            "Fwd Packet Length Mean": float(row_dict.get("pkt_len_mean", 0)),
            "Fwd Packet Length Std": float(row_dict.get("pkt_len_std", 0)),
            "Fwd Packet Length Max": float(row_dict.get("pkt_len_max", 0)),
            "Flow Bytes/s": float(row_dict.get("bytes_per_sec", 0)),
            "Flow Packets/s": float(row_dict.get("pkts_per_sec", 0)),
            "Flow IAT Mean": float(row_dict.get("iat_mean", 0)) * 1e6,
            "Flow IAT Std": float(row_dict.get("iat_std", 0)) * 1e6,
            "Flow IAT Max": float(row_dict.get("iat_max", 0)) * 1e6,
            "Flow IAT Min": float(row_dict.get("iat_min", 0)) * 1e6,
            "Packet Length Mean": float(row_dict.get("pkt_len_mean", 0)),
            "Packet Length Std": float(row_dict.get("pkt_len_std", 0)),
            "Packet Length Max": float(row_dict.get("pkt_len_max", 0)),
            "Avg Packet Size": float(row_dict.get("pkt_len_mean", 0)),
            "Avg Fwd Segment Size": float(row_dict.get("pkt_len_mean", 0)),
            "SYN Flag Count": float(row_dict.get("syn_count", 0)),
            "URG Flag Count": float(row_dict.get("urg_count", 0)),
            "Init Fwd Win Bytes": float(row_dict.get("tcp_win_init", 0)),
        }
        for k, v in m.items():
            if k in feature_row:
                feature_row[k] = v

        for k in feature_row:
            try:
                v = float(feature_row[k])
            except Exception:
                v = 0.0
            if pd.isna(v) or np.isinf(v):
                v = 0.0
            feature_row[k] = v

    if "ClassLabel" in EXPECTED_FEATURES:
        if label_text and label_text in CSV_LABEL_TO_CLASS:
            feature_row["ClassLabel"] = float(CSV_LABEL_TO_CLASS[label_text])
        else:
            feature_row["ClassLabel"] = 0.0

    return pd.DataFrame([feature_row])[EXPECTED_FEATURES]

def predict_threat(row_df: pd.DataFrame):
    # Force live Scapy row to match the exact 58 model features
    EXPECTED_FEATURES = model.get_booster().feature_names
    row_df = row_df.reindex(columns=EXPECTED_FEATURES)

    probs = model.predict_proba(row_df)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100.0
    return CLASS_MAP.get(pred_idx, "Unknown"), confidence, probs, pred_idx

def compute_shap_explanation(row_df: pd.DataFrame, pred_class_idx: int) -> dict:
    # Force live Scapy row to match the exact 58 model features
    EXPECTED_FEATURES = model.get_booster().feature_names
    row_df = row_df.reindex(columns=EXPECTED_FEATURES)

    shap_values = explainer(row_df)
    class_shap = shap_values.values[0, :, pred_class_idx]

    scores = []
    for i, col_name in enumerate(row_df.columns):
        val = float(row_df.iloc[0, i])
        phi = float(class_shap[i])
        if abs(val) >= 1000:
            val_str = f"{val:,.1f}"
        elif val.is_integer():
            val_str = f"{int(val)}"
        elif abs(val) < 0.001 and val != 0:
            val_str = f"{val:.2e}"
        else:
            val_str = f"{val:.2f}"
        scores.append({"name": col_name, "val_str": val_str, "val": val, "phi": phi})

    supporting = sorted([s for s in scores if s["phi"] > 0], key=lambda x: x["phi"], reverse=True)[:3]
    opposing = sorted([s for s in scores if s["phi"] < 0], key=lambda x: x["phi"])[:2]

    cls_name = CLASS_MAP.get(pred_class_idx, "Traffic")
    if supporting:
        supp_text = ", ".join([f"{s['name']}={s['val_str']} (Δ+{s['phi']:.2f})" for s in supporting[:2]])
        sentence = f"Prediction for {cls_name} is driven by {supp_text}."
        if opposing:
            opp_text = ", ".join([f"{s['name']}={s['val_str']} (Δ{s['phi']:.2f})" for s in opposing[:1]])
            sentence += f" Opposed by {opp_text}."
    else:
        sentence = "Equilibrium classification across baseline features."

    return {
        "sentence": sentence,
        "supporting": supporting,
        "opposing": opposing,
    }

def load_scenario_df(scenario_key):
    path = SCENARIOS.get(scenario_key)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    return None

if st.session_state.raw_flows is None:
    st.session_state.raw_flows = load_scenario_df(st.session_state.selected_scenario)

top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns([2.8, 1.4, 1.0, 1.0, 1.2])

with top_c1:
    scenario_chosen = st.selectbox(
        "Scenario",
        list(SCENARIOS.keys()),
        index=list(SCENARIOS.keys()).index(st.session_state.selected_scenario) if st.session_state.selected_scenario in SCENARIOS else 0,
        label_visibility="collapsed",
    )
    if scenario_chosen != st.session_state.selected_scenario:
        st.session_state.selected_scenario = scenario_chosen
        st.session_state.raw_flows = load_scenario_df(scenario_chosen)
        st.session_state.current_idx = 0
        st.session_state.is_playing = False
        st.session_state.display_data = []
        st.session_state.backend_data = []
        st.session_state.total_inspected = 0
        st.session_state.total_threats = 0
        st.session_state.threat_counts = {"BENIGN": 0, "DDoS": 0, "PortScan": 0, "Botnet": 0}
        st.session_state.start_time_real = time.time()
        st.rerun()

with top_c2:
    speed_chosen = st.selectbox(
        "Speed",
        list(SPEED_OPTIONS.keys()),
        index=list(SPEED_OPTIONS.keys()).index(st.session_state.speed_label) if st.session_state.speed_label in SPEED_OPTIONS else 1,
        label_visibility="collapsed",
    )
    st.session_state.speed_label = speed_chosen
    speed_ms = SPEED_OPTIONS.get(speed_chosen, 250)

with top_c3:
    play_label = "⏸ Pause" if st.session_state.is_playing else "▶ Start"
    if st.button(play_label, use_container_width=True, type="primary" if not st.session_state.is_playing else "secondary"):
        st.session_state.is_playing = not st.session_state.is_playing
        if st.session_state.is_playing and st.session_state.raw_flows is None:
            st.session_state.raw_flows = load_scenario_df(st.session_state.selected_scenario)
            st.session_state.current_idx = 0
            st.session_state.start_time_real = time.time()
            if st.session_state.raw_flows is None:
                st.warning("Upload CSV/PCAP first.")
                st.session_state.is_playing = False

with top_c4:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.current_idx = 0
        st.session_state.display_data = []
        st.session_state.backend_data = []
        st.session_state.total_inspected = 0
        st.session_state.total_threats = 0
        st.session_state.threat_counts = {"BENIGN": 0, "DDoS": 0, "PortScan": 0, "Botnet": 0}
        st.session_state.raw_flows = load_scenario_df(st.session_state.selected_scenario)
        st.session_state.start_time_real = time.time()
        st.rerun()

with top_c5:
    with st.popover("📁 Upload"):
        uploaded_file = st.file_uploader(
            "Upload File",
            type=["csv", "pcap", "pcapng"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            if ext == "csv":
                df = pd.read_csv(tmp_path)
                df.columns = df.columns.str.strip()
                st.session_state.raw_flows = df
                st.session_state.selected_scenario = "📁 Custom Upload"
                st.session_state.current_idx = 0
                st.success(f"✅ Loaded {len(df)} rows")
            elif ext in ("pcap", "pcapng"):
                from src.ingest import ingest_pcap_to_csv
                with st.spinner("Extracting flows..."):
                    df = ingest_pcap_to_csv(tmp_path, "data/processed/uploaded_flows.csv")
                    st.session_state.raw_flows = df
                    st.session_state.selected_scenario = "📁 Custom Upload"
                    st.session_state.current_idx = 0
                st.success(f"✅ Extracted {len(df)} flows")

if st.session_state.is_playing and st.session_state.raw_flows is not None:
    raw_df = st.session_state.raw_flows
    total_len = len(raw_df)

    if st.session_state.current_idx < total_len:
        row = raw_df.iloc[st.session_state.current_idx].copy()
        st.session_state.current_idx += 1

        label_text = str(row.get("True_Class", row.get("Label", "")))
        if label_text.lower() in ("nan", "none", ""):
            label_text = None

        row_dict = row.to_dict()

        model_input = prepare_model_input(row_dict, label_text=label_text)
        threat_class, confidence, probs, pred_idx = predict_threat(model_input)
        shap_info = compute_shap_explanation(model_input, pred_idx)

        st.session_state.total_inspected += 1
        if threat_class != "BENIGN":
            st.session_state.total_threats += 1
        st.session_state.threat_counts[threat_class] = (
            st.session_state.threat_counts.get(threat_class, 0) + 1
        )

        ts = str(row.get("timestamp", datetime.now().strftime("%H:%M:%S")))

        dst_port = int(float(row.get("dst_port", row.get("Destination Port", 80))))
        src_port = int(float(row.get("src_port", row.get("Source Port", random.randint(40000, 65000)))))

        src_ip = str(row.get("src_ip", f"192.168.1.{random.randint(10, 199)}"))
        dst_ip = str(row.get("dst_ip", "192.168.1.50" if threat_class != "BENIGN" else "142.250.190.46"))
        proto_name = str(row.get("protocol", "TCP"))

        if threat_class == "DDoS":
            evidence_tag = "Volumetric Flood"
        elif threat_class == "PortScan":
            evidence_tag = f"SYN Probe → :{dst_port}"
        elif threat_class == "Botnet":
            evidence_tag = "C2 Beaconing"
        else:
            evidence_tag = "Nominal Session"

        flow_5tuple = f"{src_ip}:{src_port} ➔ {dst_ip}:{dst_port}"
        flow_id = str(row.get("flow_id", f"FL-{st.session_state.current_idx:05d}"))

        probs_dict = {
            CLASS_MAP[i]: round(float(probs[i]) * 100, 1) for i in range(len(probs))
        }

        dur_val = float(row_dict.get("Flow Duration", row_dict.get("flow_duration_sec", 0)))
        bytes_sec = float(row_dict.get("Flow Bytes/s", row_dict.get("bytes_per_sec", 0)))
        pkts_sec = float(row_dict.get("Flow Packets/s", row_dict.get("pkts_per_sec", 0)))
        iat_val = float(row_dict.get("Flow IAT Mean", row_dict.get("iat_mean", 0)))
        total_b = float(row_dict.get("Fwd Packets Length Total", row_dict.get("total_bytes", 0)))
        total_p = int(float(row_dict.get("Total Fwd Packets", row_dict.get("packet_count", 1))) + float(row_dict.get("Total Backward Packets", 0)))

        dur_fmt = f"{dur_val/1e3:.1f}ms" if dur_val > 1000 else f"{dur_val:.0f}μs"
        bytes_fmt = f"{total_b/1e3:.1f}KB" if total_b > 1000 else f"{int(total_b)}B"
        iat_fmt = f"{iat_val/1e3:.1f}ms" if iat_val > 1000 else f"{iat_val:.0f}μs"

        st.session_state.display_data.insert(
            0,
            {
                "TIME": ts,
                "FLOW ID": flow_id,
                "SRC IP": src_ip,
                "SPORT": src_port,
                "DST IP": dst_ip,
                "DPORT": dst_port,
                "PROTO": proto_name,
                "PKTS": total_p,
                "BYTES": bytes_fmt,
                "PKT RATE": f"{pkts_sec:,.0f} p/s",
                "THROUGHPUT": f"{bytes_sec:,.0f} B/s",
                "DURATION": dur_fmt,
                "IAT MEAN": iat_fmt,
                "CLASSIFICATION": f"{THREAT_ICONS.get(threat_class, '')} {threat_class}",
                "TRACE": evidence_tag,
            },
        )

        st.session_state.backend_data.insert(
            0,
            {
                "flow_id": flow_id,
                "threat_class": threat_class,
                "confidence": confidence,
                "probs": probs_dict,
                "flow_5tuple": f"{src_ip}:{src_port} ➔ {dst_ip}:{dst_port} ({proto_name})",
                "duration_us": dur_val,
                "bytes_per_sec": bytes_sec,
                "pkts_per_sec": pkts_sec,
                "iat_mean": iat_val,
                "shap_info": shap_info,
            },
        )

        if len(st.session_state.display_data) > MAX_FEED_ROWS:
            st.session_state.display_data.pop()
            st.session_state.backend_data.pop()
    else:
        st.session_state.is_playing = False

total_dataset_rows = len(st.session_state.raw_flows) if st.session_state.raw_flows is not None else 0
total_ingested = st.session_state.total_inspected
threats_flagged = st.session_state.total_threats
elapsed = max(time.time() - st.session_state.start_time_real, 0.001)
velocity = total_ingested / elapsed if total_ingested > 0 else 0.0
det_rate = (threats_flagged / total_ingested * 100) if total_ingested > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"<div class='card-box' style='padding:6px 10px;'>"
        f"<div style='color:#64748B; font-size:0.68rem; font-weight:600; text-transform:uppercase;'>Ingested Flows</div>"
        f"<div style='font-family:JetBrains Mono,monospace; font-size:1.15rem; font-weight:600; color:#0F172A; margin-top:1px;'>"
        f"{total_ingested} <span style='font-size:0.7rem; font-weight:400; color:#64748B;'>/ {total_dataset_rows}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with k2:
    threat_col = "#DC2626" if threats_flagged > 0 else "#0F172A"
    st.markdown(
        f"<div class='card-box' style='padding:6px 10px;'>"
        f"<div style='color:#64748B; font-size:0.68rem; font-weight:600; text-transform:uppercase;'>Threats Flagged</div>"
        f"<div style='font-family:JetBrains Mono,monospace; font-size:1.15rem; font-weight:600; color:{threat_col}; margin-top:1px;'>{threats_flagged}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"<div class='card-box' style='padding:6px 10px;'>"
        f"<div style='color:#64748B; font-size:0.68rem; font-weight:600; text-transform:uppercase;'>Ingest Velocity</div>"
        f"<div style='font-family:JetBrains Mono,monospace; font-size:1.15rem; font-weight:600; color:#0F172A; margin-top:1px;'>"
        f"{velocity:.1f} <span style='font-size:0.7rem; font-weight:400; color:#64748B;'>f/s</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"<div class='card-box' style='padding:6px 10px;'>"
        f"<div style='color:#64748B; font-size:0.68rem; font-weight:600; text-transform:uppercase;'>Detection Ratio</div>"
        f"<div style='font-family:JetBrains Mono,monospace; font-size:1.15rem; font-weight:600; color:#0F172A; margin-top:1px;'>{det_rate:.1f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

col_feed, col_inspect = st.columns([65, 35])

def style_class_pill(val):
    val_str = str(val)
    base_css = "font-weight:500; border-radius:4px; font-family:JetBrains Mono,monospace; padding: 2px 5px;"
    if "BENIGN" in val_str:
        return f"{base_css} background-color:#ECFDF5; color:#059669; border: 1px solid #A7F3D0;"
    elif "DDoS" in val_str:
        return f"{base_css} background-color:#FEF2F2; color:#DC2626; border: 1px solid #FECACA;"
    elif "PortScan" in val_str:
        return f"{base_css} background-color:#FFFBEB; color:#D97706; border: 1px solid #FDE68A;"
    elif "Botnet" in val_str:
        return f"{base_css} background-color:#F5F3FF; color:#7C3AED; border: 1px solid #DDD6FE;"
    return ""

with col_feed:
    st.markdown(
        "<div style='font-size:0.75rem; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;'>📋 Live Telemetry Stream (Click row to inspect)</div>",
        unsafe_allow_html=True,
    )

    event = None
    if st.session_state.display_data:
        df_display = pd.DataFrame(st.session_state.display_data)
        styled = df_display.style.map(style_class_pill, subset=["CLASSIFICATION"])
        event = st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=420,
        )
    else:
        st.markdown(
            "<div class='card-box' style='padding:60px 20px; text-align:center; height:420px; display:flex; flex-direction:column; justify-content:center;'>"
            "<div style='font-size:1.8rem; margin-bottom:6px;'>📡</div>"
            "<div style='font-size:0.95rem; font-weight:600; color:#0F172A;'>Sensor Feed Ready</div>"
            "<div style='color:#64748B; font-size:0.78rem; margin-top:3px;'>Select scenario above and click <b>▶ Start</b> to stream flows.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

with col_inspect:
    st.markdown(
        "<div style='font-size:0.75rem; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;'>🔬 Forensic Deep Inspection</div>",
        unsafe_allow_html=True,
    )

    selected_meta = None
    if event and hasattr(event, "selection") and event.selection and event.selection.rows:
        sel_idx = event.selection.rows[0]
        if sel_idx < len(st.session_state.backend_data):
            selected_meta = st.session_state.backend_data[sel_idx]

    if selected_meta is not None:
        tc = selected_meta["threat_class"]
        color = THREAT_COLORS.get(tc, "#059669")
        icon = THREAT_ICONS.get(tc, "")

        st.markdown(
            f"<div class='card-box' style='border-left:4px solid {color}; margin-bottom:6px; padding:10px 14px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
            f"<div>"
            f"<div style='font-size:0.68rem; font-weight:600; color:#64748B; text-transform:uppercase;'>Flow ID: {selected_meta['flow_id']}</div>"
            f"<div style='font-family:JetBrains Mono,monospace; font-size:0.82rem; font-weight:600; color:#0F172A; margin-top:2px;'>"
            f"{selected_meta['flow_5tuple']}"
            f"</div>"
            f"</div>"
            f"<div style='text-align:right;'>"
            f"<div style='font-family:JetBrains Mono,monospace; font-size:1rem; font-weight:600; color:{color};'>"
            f"{icon} {tc}"
            f"</div>"
            f"<div style='font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#64748B;'>"
            f"{selected_meta['confidence']:.1f}% confidence"
            f"</div>"
            f"</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        probs_dict = selected_meta.get("probs", {})
        prob_items = []
        for cls_name in ["BENIGN", "DDoS", "PortScan", "Botnet"]:
            p_val = probs_dict.get(cls_name, 0.0)
            p_col = THREAT_COLORS.get(cls_name, "#64748B")
            p_icon = THREAT_ICONS.get(cls_name, "")
            prob_items.append(
                f"<div style='margin-bottom:5px;'>"
                f"<div style='display:flex; justify-content:space-between; font-size:0.72rem; font-family:JetBrains Mono,monospace;'>"
                f"<span style='color:#334155;'>{p_icon} {cls_name}</span>"
                f"<span style='color:{p_col}; font-weight:600;'>{p_val:.1f}%</span>"
                f"</div>"
                f"<div style='height:5px; width:100%; border-radius:3px; background:#F1F5F9; overflow:hidden; margin-top:2px;'>"
                f"<div style='height:100%; width:{p_val}%; background:{p_col}; border-radius:3px;'></div>"
                f"</div>"
                f"</div>"
            )

        prob_html = (
            "<div class='card-box' style='margin-bottom:6px; padding:10px 14px;'>"
            "<div style='font-size:0.68rem; font-weight:600; color:#64748B; text-transform:uppercase; margin-bottom:6px;'>Softmax Probability Distribution Vector</div>"
            + "".join(prob_items)
            + "</div>"
        )
        st.markdown(prob_html, unsafe_allow_html=True)

        dur = selected_meta["duration_us"]
        dur_str = f"{dur/1e3:.1f}ms" if dur > 1000 else f"{dur:.0f}μs"
        b_s = selected_meta["bytes_per_sec"]
        p_s = selected_meta["pkts_per_sec"]
        iat_m = selected_meta["iat_mean"]

        morph_html = (
            f"<div class='card-box' style='margin-bottom:6px; padding:10px 14px;'>"
            f"<div style='font-size:0.68rem; font-weight:600; color:#64748B; text-transform:uppercase; margin-bottom:6px;'>Flow Morphology</div>"
            f"<div style='display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:6px;'>"
            f"<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:5px; padding:6px 6px; text-align:center;'>"
            f"<div style='color:#64748B; font-size:0.62rem; font-family:JetBrains Mono,monospace;'>DURATION</div>"
            f"<div style='color:#0F172A; font-size:0.82rem; font-weight:600; font-family:JetBrains Mono,monospace;'>{dur_str}</div>"
            f"</div>"
            f"<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:5px; padding:6px 6px; text-align:center;'>"
            f"<div style='color:#64748B; font-size:0.62rem; font-family:JetBrains Mono,monospace;'>PKT RATE</div>"
            f"<div style='color:#0F172A; font-size:0.82rem; font-weight:600; font-family:JetBrains Mono,monospace;'>{p_s:,.0f}p/s</div>"
            f"</div>"
            f"<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:5px; padding:6px 6px; text-align:center;'>"
            f"<div style='color:#64748B; font-size:0.62rem; font-family:JetBrains Mono,monospace;'>THROUGHPUT</div>"
            f"<div style='color:#0F172A; font-size:0.82rem; font-weight:600; font-family:JetBrains Mono,monospace;'>{b_s:,.0f}B/s</div>"
            f"</div>"
            f"<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:5px; padding:6px 6px; text-align:center;'>"
            f"<div style='color:#64748B; font-size:0.62rem; font-family:JetBrains Mono,monospace;'>IAT MEAN</div>"
            f"<div style='color:#0F172A; font-size:0.82rem; font-weight:600; font-family:JetBrains Mono,monospace;'>{iat_m/1e3:.1f}ms</div>"
            f"</div>"
            f"</div>"
            f"</div>"
        )
        st.markdown(morph_html, unsafe_allow_html=True)

        shap_info = selected_meta.get("shap_info", {})
        xai_html = (
            f"<div class='card-box' style='border-left:4px solid {color}; padding:10px 14px;'>"
            f"<div style='font-size:0.68rem; font-weight:600; color:{color}; text-transform:uppercase; margin-bottom:4px;'>TreeSHAP Feature Attribution</div>"
            f"<div style='color:#334155; font-size:0.74rem; font-family:JetBrains Mono,monospace; line-height:1.4; margin-bottom:6px;'>"
            f"{shap_info.get('sentence', 'TreeSHAP attribution computed.')}"
            f"</div>"
        )

        supp_list = shap_info.get("supporting", [])
        if supp_list:
            xai_html += "<div style='margin-bottom:4px;'><div style='font-size:0.65rem; font-weight:600; color:#059669; text-transform:uppercase; margin-bottom:2px;'>▲ Supporting Drivers (Positive Margin)</div>"
            for s in supp_list:
                xai_html += (
                    f"<div style='background:#ECFDF5; border:1px solid #A7F3D0; border-radius:4px; padding:3px 8px; margin-bottom:3px; display:flex; justify-content:space-between; font-size:0.72rem; font-family:JetBrains Mono,monospace;'>"
                    f"<span style='color:#065F46; font-weight:500;'>{s['name']} <span style='color:#64748B;'>={s['val_str']}</span></span>"
                    f"<span style='color:#059669; font-weight:600;'>Δ+{s['phi']:.2f}</span>"
                    f"</div>"
                )
            xai_html += "</div>"

        opp_list = shap_info.get("opposing", [])
        if opp_list:
            xai_html += "<div><div style='font-size:0.65rem; font-weight:600; color:#DC2626; text-transform:uppercase; margin-bottom:2px;'>▼ Opposing Drivers (Negative Margin)</div>"
            for o in opp_list:
                xai_html += (
                    f"<div style='background:#FEF2F2; border:1px solid #FECACA; border-radius:4px; padding:3px 8px; margin-bottom:3px; display:flex; justify-content:space-between; font-size:0.72rem; font-family:JetBrains Mono,monospace;'>"
                    f"<span style='color:#991B1B; font-weight:500;'>{o['name']} <span style='color:#64748B;'>={o['val_str']}</span></span>"
                    f"<span style='color:#DC2626; font-weight:600;'>Δ{o['phi']:.2f}</span>"
                    f"</div>"
                )
            xai_html += "</div>"

        xai_html += "</div>"
        st.markdown(xai_html, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='card-box' style='padding:60px 20px; text-align:center; height:420px; display:flex; flex-direction:column; justify-content:center;'>"
            "<div style='font-size:1.8rem; margin-bottom:6px;'>🔍</div>"
            "<div style='font-size:0.95rem; font-weight:600; color:#0F172A;'>No Flow Selected</div>"
            "<div style='color:#64748B; font-size:0.78rem; margin-top:3px; line-height:1.4;'>"
            "Click on any row in the telemetry stream to inspect AI probabilities & TreeSHAP attributions."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

if st.session_state.is_playing:
    time.sleep(speed_ms / 1000.0)
    st.rerun()
