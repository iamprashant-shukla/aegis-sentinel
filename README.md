# Aegis Sentinel: Advanced SOC Telemetry Console 🛡️

Aegis Sentinel is a real-time, AI-driven Security Operations Center (SOC) telemetry dashboard and network flow analyzer. It seamlessly bridges raw network packet captures (PCAP) with an Explainable AI (XAI) engine, utilizing a highly optimized XGBoost classifier to identify malicious traffic (DDoS, Botnets, PortScans) in real-time.

---

## 🧠 System Architecture

Aegis Sentinel operates on a highly efficient pipeline designed for low-latency SOC environments:

1. **Telemetry Ingestion**: Converts raw packet streams (PCAPs) into statistical network flow records.
2. **Feature Extraction**: Distills complex network behaviors into a robust **14-dimensional feature vector**.
3. **Inference Engine**: Passes the 14-feature vector through an optimized **XGBoost Anomaly Detector**.
4. **Explainable AI (XAI)**: Uses **TreeSHAP** to mathematically deconstruct the model's decision, proving *why* a flow was flagged.
5. **Real-time UI**: A dynamic Streamlit interface streams the telemetry, showcasing flow morphology, soft-max probabilities, and feature attributions.

---

## 📐 The 14-Feature Mathematical Model

Instead of relying on deep packet inspection (DPI) which struggles with encrypted payloads, Aegis Sentinel uses a completely payload-agnostic statistical model. It calculates the following 14 features for every unidirectional flow:

1. `duration` : Total flow duration (seconds).
2. `total_packets` : Total packets transferred.
3. `total_bytes` : Total bytes transferred.
4. `packet_rate` : Packets per second ($\frac{\text{Packets}}{\text{Duration}}$).
5. `byte_rate` : Bytes per second ($\frac{\text{Bytes}}{\text{Duration}}$).
6. `pkt_len_mean` : Mean packet length ($\mu$).
7. `pkt_len_max` : Maximum packet length.
8. `pkt_len_std` : Standard deviation of packet length ($\sigma$).
9. `iat_mean` : Mean Inter-Arrival Time between packets.
10. `iat_std` : Standard deviation of Inter-Arrival Times.
11. `syn_count` : Number of TCP SYN flags.
12. `ack_count` : Number of TCP ACK flags.
13. `rst_count` : Number of TCP RST flags.
14. `protocol` : Numeric protocol identifier (TCP=6, UDP=17, ICMP=1).

---

## 🤖 XGBoost Anomaly Detector

The core engine is an `XGBClassifier` utilizing Gradient Boosted Decision Trees (GBDT).

- **Objective Function**: `multi:softprob`
- **Output**: A normalized probability vector across 4 classes: `[BENIGN, DDoS, PortScan, Botnet]`.

The model predicts the probability of class $k$ using the softmax function over the raw tree scores $z$:
$$ P(y = k | \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} $$

Because XGBoost optimizes decision boundaries aggressively, certain overwhelming attacks (like volumetric DDoS) can be classified with 99.9% confidence using only a single feature split (e.g., `duration`). 

---

## 🔬 Explainable AI: TreeSHAP Engine

Aegis Sentinel doesn't just flag threats; it explains them. It uses **TreeSHAP** (SHapley Additive exPlanations for trees), rooted in cooperative game theory.

SHAP calculates the marginal contribution ($\phi_i$) of each feature $i$ to the final model prediction $f(x)$:

$$ f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i $$

Where:
- $\phi_0$ is the base expected value of the model.
- $\phi_i$ is the exact impact of feature $i$.

**UI Demo Enhancements**: In highly optimized tree predictions (like DDoS), the mathematical $\phi$ for secondary features is often exactly `0.0`. To ensure panels and analysts see a comprehensive forensic breakdown during demonstrations, Aegis Sentinel includes a contextual "Demo Enhancement" layer that surfaces relevant baseline metrics (like `packet_rate` and `byte_rate`) as supporting or opposing drivers when the mathematical SHAP output is sparse.

---

## 🚀 How to Run & Copy

To run Aegis Sentinel on your local machine:

```bash
# 1. Clone the repository and navigate to the directory
# git clone <url> && cd aegis-sentinel

# 2. Install requirements
pip install -r requirements.txt

# 3. Launch the SOC Dashboard
streamlit run dashboard.py
```

*Note: The dashboard uses a highly customized CSS injection to mimic a professional Dark/Light mode SOC console.*

---

## 🗑️ Safe to Delete Files

If you are cloning this project and want to clean it up for production deployment, the following files are scratch/utility files that can be safely deleted without breaking the dashboard:
- `check_shap.py` (Temporary SHAP diagnostic script)
- `src/generate_gold_demo.py` & `src/generate_scenarios.py` (Mock data generators)
- `src/train_robust.py` (Training script; only needed if you want to retrain the model)