import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib
import glob
import os

def load_data():
    files = glob.glob('data/processed/scenario_*.csv')
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No scenario CSV files found in data/processed/")
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def map_features(df):
    """Map legacy CICIDS-2017 style columns to the 14 robust metrics"""
    # Create the mapped dataframe
    mapped = pd.DataFrame()
    
    # Duration (convert micro to sec)
    mapped['duration'] = df.get('Flow Duration', df.get('flow_duration_sec', 0)) * 1e-6
    
    # Total Packets
    if 'Total Fwd Packets' in df and 'Total Backward Packets' in df:
        mapped['total_packets'] = df['Total Fwd Packets'] + df['Total Backward Packets']
    else:
        mapped['total_packets'] = df.get('packet_count', 0)
        
    # Total Bytes
    if 'Fwd Packets Length Total' in df and 'Bwd Packets Length Total' in df:
        mapped['total_bytes'] = df['Fwd Packets Length Total'] + df['Bwd Packets Length Total']
    else:
        mapped['total_bytes'] = df.get('total_bytes', 0)
        
    # Rates
    mapped['packet_rate'] = df.get('Flow Packets/s', df.get('pkts_per_sec', 0))
    mapped['byte_rate'] = df.get('Flow Bytes/s', df.get('bytes_per_sec', 0))
    
    # Packet lengths
    mapped['pkt_len_mean'] = df.get('Packet Length Mean', df.get('pkt_len_mean', 0))
    mapped['pkt_len_max'] = df.get('Packet Length Max', df.get('pkt_len_max', 0))
    mapped['pkt_len_std'] = df.get('Packet Length Std', df.get('pkt_len_std', 0))
    
    # IAT (convert micro to sec)
    mapped['iat_mean'] = df.get('Flow IAT Mean', df.get('iat_mean', 0)) * 1e-6
    mapped['iat_std'] = df.get('Flow IAT Std', df.get('iat_std', 0)) * 1e-6
    
    # Flags
    mapped['syn_count'] = df.get('SYN Flag Count', df.get('syn_count', 0))
    mapped['ack_count'] = df.get('ACK Flag Count', df.get('ack_count', 0))
    mapped['rst_count'] = df.get('RST Flag Count', df.get('rst_count', 0))
    
    # Protocol
    def map_proto(p):
        p_str = str(p).upper().strip()
        if p_str == 'TCP' or p_str == '6' or p_str == '6.0': return 6
        if p_str == 'UDP' or p_str == '17' or p_str == '17.0': return 17
        if p_str == 'ICMP' or p_str == '1' or p_str == '1.0': return 1
        return 0
        
    mapped['protocol'] = df.get('protocol', df.get('Protocol', 'TCP')).apply(map_proto)
    
    # Deal with NaNs/Infs
    mapped = mapped.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return mapped

CLASS_MAP = {"BENIGN": 0, "DDoS": 1, "PortScan": 2, "Botnet": 3}

def map_labels(df):
    labels = df.get('True_Class', df.get('Label', 'BENIGN'))
    # standardize strings
    labels = labels.astype(str).str.strip()
    
    y = np.zeros(len(labels), dtype=int)
    y[labels.str.contains('dos', case=False, na=False)] = 1
    y[labels.str.contains('portscan', case=False, na=False)] = 2
    y[labels.str.contains('bot', case=False, na=False)] = 3
    y[labels.str.contains('infiltration|web', case=False, na=False)] = 3
    # Default to 0 for everything else
    return y

def train():
    print("Loading datasets...")
    df = load_data()
    print(f"Loaded {len(df)} total rows.")
    
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    elif 'Time' in df.columns:
        df = df.sort_values('Time')
        
    X = map_features(df)
    y = map_labels(df)
    
    # Chronological Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    # Ensure ordered features
    FEATURES = ['duration', 'total_packets', 'total_bytes', 'packet_rate', 'byte_rate', 
                'pkt_len_mean', 'pkt_len_max', 'pkt_len_std', 'iat_mean', 'iat_std', 
                'syn_count', 'ack_count', 'rst_count', 'protocol']
                
    X_train = X_train[FEATURES]
    X_test = X_test[FEATURES]
    
    model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=5,
        learning_rate=0.08,
        objective="multi:softprob",
        eval_metric='mlogloss',
        random_state=42
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))
    
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "xgboost_anomaly_detector.pkl")
    joblib.dump(model, out_path)
    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    train()
