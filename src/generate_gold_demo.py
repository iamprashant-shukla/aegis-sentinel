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

def generate_gold_dataset():
    raw_path = os.path.join("data", "raw", "CICIDS2017_sample.csv")
    out_dir = os.path.join("data", "processed")
    out_file = os.path.join(out_dir, "demo_telemetry_stream.csv")
    model_path = os.path.join("models", "xgboost_anomaly_detector.pkl")

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw dataset at {raw_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file at {model_path}")

    model = joblib.load(model_path)
    feats = list(model.feature_names_in_)

    df_raw = pd.read_csv(raw_path)
    df_raw.columns = df_raw.columns.str.strip()

    categories = [
        ("BENIGN", 0, lambda d: d[d["Label"] == "BENIGN"], [
            ("192.168.1.105", "142.250.190.46", 49210, 443, "TCP"),
            ("192.168.1.112", "8.8.8.8", 53211, 53, "UDP"),
            ("192.168.1.115", "104.244.42.1", 51042, 443, "TCP"),
            ("192.168.1.120", "151.101.1.69", 54312, 443, "TCP"),
            ("192.168.1.125", "192.168.1.1", 49152, 80, "TCP"),
        ]),
        ("DDoS", 1, lambda d: d[(d["Label"] == "DoS") & (d["Total Backward Packets"] == 0)], [
            ("172.16.0.2", "192.168.10.50", 40001, 80, "TCP"),
            ("172.16.0.5", "192.168.10.50", 40002, 80, "TCP"),
            ("172.16.0.8", "192.168.10.50", 40003, 80, "TCP"),
            ("172.16.0.12", "192.168.10.50", 40004, 80, "TCP"),
            ("172.16.0.15", "192.168.10.50", 40005, 80, "TCP"),
        ]),
        ("PortScan", 2, lambda d: d[(d["Label"] == "PortScan") & (d["Total Fwd Packets"] <= 2)], [
            ("10.0.0.99", "192.168.1.50", 51881, 21, "TCP"),
            ("10.0.0.99", "192.168.1.50", 51882, 22, "TCP"),
            ("10.0.0.99", "192.168.1.50", 51883, 80, "TCP"),
            ("10.0.0.99", "192.168.1.50", 51884, 445, "TCP"),
            ("10.0.0.99", "192.168.1.50", 51885, 3389, "TCP"),
        ]),
        ("Botnet", 3, lambda d: d[d["Label"] == "Bot"], [
            ("192.168.1.105", "52.14.209.11", 49811, 6667, "TCP"),
            ("192.168.1.105", "52.14.209.11", 49812, 6667, "TCP"),
            ("192.168.1.105", "198.51.100.4", 50124, 8080, "TCP"),
            ("192.168.1.105", "198.51.100.4", 50125, 8080, "TCP"),
            ("192.168.1.105", "203.0.113.88", 51230, 4444, "TCP"),
        ]),
    ]

    selected_rows = []
    flow_idx = 1

    for class_name, class_idx, filter_fn, net_tuples in categories:
        subset = filter_fn(df_raw)
        matched_samples = []

        for _, raw_row in subset.iterrows():
            renamed = {}
            for col, val in raw_row.items():
                renamed[CSV_TO_MODEL_RENAME.get(col, col)] = val

            feat_dict = {}
            for f in feats:
                val = renamed.get(f, 0)
                try:
                    val = float(val)
                except Exception:
                    val = 0.0
                feat_dict[f] = 0.0 if (pd.isna(val) or np.isinf(val)) else val

            if "ClassLabel" in feats:
                feat_dict["ClassLabel"] = float(class_idx)

            test_df = pd.DataFrame([feat_dict])[feats]
            probs = model.predict_proba(test_df)[0]
            pred = int(np.argmax(probs))

            if pred == class_idx and probs[pred] >= 0.85:
                matched_samples.append((raw_row, feat_dict, probs[pred]))
                if len(matched_samples) == 5:
                    break

        if len(matched_samples) < 5:
            raise RuntimeError(f"Could not find 5 high-confidence samples for {class_name}")

        for i, (raw_row, feat_dict, conf) in enumerate(matched_samples):
            src_ip, dst_ip, src_port, dst_port, proto = net_tuples[i]
            row_record = {
                "flow_id": f"FL-{flow_idx:05d}",
                "timestamp": f"10:{flow_idx:02d}:00",
                "True_Class": class_name,
                "Label": class_name,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": proto,
            }
            row_record.update(feat_dict)
            selected_rows.append(row_record)
            flow_idx += 1

    result_df = pd.DataFrame(selected_rows)
    result_df.to_csv(out_file, index=False)
    print(f"[+] Generated golden demo dataset with {len(result_df)} rows at {out_file}")
    return out_file

if __name__ == "__main__":
    generate_gold_dataset()
