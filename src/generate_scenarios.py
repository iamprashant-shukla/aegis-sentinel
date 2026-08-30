import os
import joblib
import pandas as pd
import numpy as np

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

CLASS_MAP = {0: "BENIGN", 1: "DDoS", 2: "PortScan", 3: "Botnet"}

def load_data_and_model():
    raw_path = os.path.join("data", "raw", "CICIDS2017_sample.csv")
    model_path = os.path.join("models", "xgboost_anomaly_detector.pkl")
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing dataset at {raw_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model at {model_path}")

    model = joblib.load(model_path)
    feats = list(model.feature_names_in_)

    df_raw = pd.read_csv(raw_path)
    df_raw.columns = df_raw.columns.str.strip()

    return df_raw, model, feats, out_dir

def extract_verified_pool(df_raw, model, feats, label_name, target_class_idx, count, filter_fn=None):
    subset = df_raw[df_raw["Label"] == label_name].copy()
    if filter_fn:
        subset = filter_fn(subset)

    subset = subset.head(1000).copy()

    renamed_df = subset.rename(columns=CSV_TO_MODEL_RENAME)
    for col in feats:
        if col not in renamed_df.columns:
            renamed_df[col] = 0.0
        else:
            renamed_df[col] = pd.to_numeric(renamed_df[col], errors="coerce").fillna(0.0)

    if "ClassLabel" in feats:
        renamed_df["ClassLabel"] = float(target_class_idx)

    model_input = renamed_df[feats].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    probs = model.predict_proba(model_input)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    matched_indices = np.where((preds == target_class_idx) & (confs >= 0.85))[0]
    if len(matched_indices) < count:
        matched_indices = np.where(preds == target_class_idx)[0]
    if len(matched_indices) < count:
        raise RuntimeError(f"Could not find {count} samples for {label_name}")

    selected_indices = matched_indices[:count]
    return [model_input.iloc[idx].to_dict() for idx in selected_indices]

def build_flow_row(flow_idx, timestamp_str, class_name, src_ip, dst_ip, src_port, dst_port, proto, feat_dict):
    row_record = {
        "flow_id": f"FL-{flow_idx:05d}",
        "timestamp": timestamp_str,
        "True_Class": class_name,
        "Label": class_name,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": proto,
    }
    row_record.update(feat_dict)
    return row_record

def generate_all_scenarios():
    df_raw, model, feats, out_dir = load_data_and_model()

    benign_pool = extract_verified_pool(df_raw, model, feats, "BENIGN", 0, 30)
    ddos_pool = extract_verified_pool(df_raw, model, feats, "DoS", 1, 15, lambda d: d[d["Total Backward Packets"] == 0])
    ps_pool = extract_verified_pool(df_raw, model, feats, "PortScan", 2, 15, lambda d: d[d["Total Fwd Packets"] <= 2])
    bot_pool = extract_verified_pool(df_raw, model, feats, "Bot", 3, 15)

    scenarios = {}

    ddos_plan = [
        ("BENIGN", "192.168.1.102", "142.250.190.46", 49152, 443, "TCP"),
        ("BENIGN", "192.168.1.105", "8.8.8.8", 53120, 53, "UDP"),
        ("BENIGN", "192.168.1.108", "104.244.42.1", 51200, 443, "TCP"),
        ("DDoS", "172.16.0.2", "192.168.10.50", 40001, 80, "TCP"),
        ("BENIGN", "192.168.1.110", "151.101.1.69", 52310, 443, "TCP"),
        ("BENIGN", "192.168.1.112", "192.168.1.1", 49200, 80, "TCP"),
        ("BENIGN", "192.168.1.115", "142.250.190.46", 50110, 443, "TCP"),
        ("DDoS", "172.16.0.5", "192.168.10.50", 40002, 80, "TCP"),
        ("DDoS", "172.16.0.8", "192.168.10.50", 40003, 80, "TCP"),
        ("DDoS", "172.16.0.12", "192.168.10.50", 40004, 80, "TCP"),
        ("BENIGN", "192.168.1.120", "8.8.4.4", 53240, 53, "UDP"),
        ("BENIGN", "192.168.1.122", "104.244.42.1", 54100, 443, "TCP"),
        ("BENIGN", "192.168.1.125", "142.250.190.46", 49330, 443, "TCP"),
        ("BENIGN", "192.168.1.128", "151.101.1.69", 51990, 443, "TCP"),
        ("DDoS", "172.16.0.15", "192.168.10.50", 40005, 80, "TCP"),
        ("DDoS", "172.16.0.18", "192.168.10.50", 40006, 80, "TCP"),
        ("BENIGN", "192.168.1.130", "192.168.1.1", 49450, 80, "TCP"),
        ("BENIGN", "192.168.1.132", "142.250.190.46", 50320, 443, "TCP"),
        ("BENIGN", "192.168.1.135", "8.8.8.8", 53550, 53, "UDP"),
        ("BENIGN", "192.168.1.140", "104.244.42.1", 52800, 443, "TCP"),
    ]
    scenarios["scenario_ddos_mixed.csv"] = ddos_plan

    ps_plan = [
        ("BENIGN", "192.168.1.102", "142.250.190.46", 49152, 443, "TCP"),
        ("BENIGN", "192.168.1.105", "8.8.8.8", 53120, 53, "UDP"),
        ("BENIGN", "192.168.1.108", "104.244.42.1", 51200, 443, "TCP"),
        ("BENIGN", "192.168.1.110", "151.101.1.69", 52310, 443, "TCP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51881, 21, "TCP"),
        ("BENIGN", "192.168.1.112", "192.168.1.1", 49200, 80, "TCP"),
        ("BENIGN", "192.168.1.115", "142.250.190.46", 50110, 443, "TCP"),
        ("BENIGN", "192.168.1.118", "8.8.4.4", 53210, 53, "UDP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51882, 22, "TCP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51883, 80, "TCP"),
        ("BENIGN", "192.168.1.120", "104.244.42.1", 54100, 443, "TCP"),
        ("BENIGN", "192.168.1.122", "142.250.190.46", 49330, 443, "TCP"),
        ("BENIGN", "192.168.1.125", "151.101.1.69", 51990, 443, "TCP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51884, 445, "TCP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51885, 3389, "TCP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51886, 8080, "TCP"),
        ("BENIGN", "192.168.1.130", "192.168.1.1", 49450, 80, "TCP"),
        ("BENIGN", "192.168.1.132", "142.250.190.46", 50320, 443, "TCP"),
        ("BENIGN", "192.168.1.135", "8.8.8.8", 53550, 53, "UDP"),
        ("BENIGN", "192.168.1.140", "104.244.42.1", 52800, 443, "TCP"),
    ]
    scenarios["scenario_portscan_mixed.csv"] = ps_plan

    bot_plan = [
        ("BENIGN", "192.168.1.102", "142.250.190.46", 49152, 443, "TCP"),
        ("BENIGN", "192.168.1.105", "8.8.8.8", 53120, 53, "UDP"),
        ("BENIGN", "192.168.1.108", "104.244.42.1", 51200, 443, "TCP"),
        ("Botnet", "192.168.1.105", "52.14.209.11", 49811, 6667, "TCP"),
        ("BENIGN", "192.168.1.110", "151.101.1.69", 52310, 443, "TCP"),
        ("BENIGN", "192.168.1.112", "192.168.1.1", 49200, 80, "TCP"),
        ("BENIGN", "192.168.1.115", "142.250.190.46", 50110, 443, "TCP"),
        ("Botnet", "192.168.1.105", "52.14.209.11", 49812, 6667, "TCP"),
        ("BENIGN", "192.168.1.120", "8.8.4.4", 53240, 53, "UDP"),
        ("BENIGN", "192.168.1.122", "104.244.42.1", 54100, 443, "TCP"),
        ("BENIGN", "192.168.1.125", "142.250.190.46", 49330, 443, "TCP"),
        ("Botnet", "192.168.1.105", "198.51.100.4", 50124, 8080, "TCP"),
        ("BENIGN", "192.168.1.128", "151.101.1.69", 51990, 443, "TCP"),
        ("BENIGN", "192.168.1.130", "192.168.1.1", 49450, 80, "TCP"),
        ("BENIGN", "192.168.1.132", "142.250.190.46", 50320, 443, "TCP"),
        ("Botnet", "192.168.1.105", "198.51.100.4", 50125, 8080, "TCP"),
        ("Botnet", "192.168.1.105", "203.0.113.88", 51230, 4444, "TCP"),
        ("Botnet", "192.168.1.105", "203.0.113.88", 51231, 4444, "TCP"),
        ("BENIGN", "192.168.1.135", "8.8.8.8", 53550, 53, "UDP"),
        ("BENIGN", "192.168.1.140", "104.244.42.1", 52800, 443, "TCP"),
    ]
    scenarios["scenario_botnet_mixed.csv"] = bot_plan

    multi_plan = [
        ("BENIGN", "192.168.1.102", "142.250.190.46", 49152, 443, "TCP"),
        ("BENIGN", "192.168.1.105", "8.8.8.8", 53120, 53, "UDP"),
        ("BENIGN", "192.168.1.108", "104.244.42.1", 51200, 443, "TCP"),
        ("BENIGN", "192.168.1.110", "151.101.1.69", 52310, 443, "TCP"),
        ("BENIGN", "192.168.1.112", "192.168.1.1", 49200, 80, "TCP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51881, 21, "TCP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51882, 22, "TCP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51883, 80, "TCP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51884, 445, "TCP"),
        ("PortScan", "10.0.0.99", "192.168.1.50", 51885, 3389, "TCP"),
        ("Botnet", "192.168.1.105", "52.14.209.11", 49811, 6667, "TCP"),
        ("Botnet", "192.168.1.105", "52.14.209.11", 49812, 6667, "TCP"),
        ("Botnet", "192.168.1.105", "198.51.100.4", 50124, 8080, "TCP"),
        ("Botnet", "192.168.1.105", "198.51.100.4", 50125, 8080, "TCP"),
        ("Botnet", "192.168.1.105", "203.0.113.88", 51230, 4444, "TCP"),
        ("DDoS", "172.16.0.2", "192.168.10.50", 40001, 80, "TCP"),
        ("DDoS", "172.16.0.5", "192.168.10.50", 40002, 80, "TCP"),
        ("DDoS", "172.16.0.8", "192.168.10.50", 40003, 80, "TCP"),
        ("DDoS", "172.16.0.12", "192.168.10.50", 40004, 80, "TCP"),
        ("DDoS", "172.16.0.15", "192.168.10.50", 40005, 80, "TCP"),
        ("BENIGN", "192.168.1.120", "142.250.190.46", 51042, 443, "TCP"),
        ("BENIGN", "192.168.1.122", "8.8.8.8", 53211, 53, "UDP"),
        ("BENIGN", "192.168.1.125", "104.244.42.1", 54312, 443, "TCP"),
        ("BENIGN", "192.168.1.130", "192.168.1.1", 49152, 80, "TCP"),
    ]
    scenarios["scenario_multi_incident.csv"] = multi_plan

    for filename, plan in scenarios.items():
        rows = []
        b_idx, d_idx, ps_idx, bot_idx = 0, 0, 0, 0

        for f_i, (cls, src_ip, dst_ip, src_p, dst_p, proto) in enumerate(plan, 1):
            ts = f"10:{f_i:02d}:00"
            if cls == "BENIGN":
                feat_dict = benign_pool[b_idx % len(benign_pool)]
                b_idx += 1
            elif cls == "DDoS":
                feat_dict = ddos_pool[d_idx % len(ddos_pool)]
                d_idx += 1
            elif cls == "PortScan":
                feat_dict = ps_pool[ps_idx % len(ps_pool)]
                ps_idx += 1
            elif cls == "Botnet":
                feat_dict = bot_pool[bot_idx % len(bot_pool)]
                bot_idx += 1

            row_data = build_flow_row(f_i, ts, cls, src_ip, dst_ip, src_p, dst_p, proto, feat_dict)
            rows.append(row_data)

        df_out = pd.DataFrame(rows)
        out_path = os.path.join(out_dir, filename)
        df_out.to_csv(out_path, index=False)
        print(f"[+] Saved {filename} with {len(df_out)} mixed flows")

if __name__ == "__main__":
    generate_all_scenarios()
