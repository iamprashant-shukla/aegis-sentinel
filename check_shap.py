import pandas as pd
import joblib
import shap
import numpy as np
from dashboard import load_model, prepare_model_input, compute_shap_explanation

model, explainer = load_model()

df = pd.read_csv("data/processed/scenario_ddos_mixed.csv")
row = df.iloc[3] # FL-00004 (DDoS)

model_input = prepare_model_input(row.to_dict())
probs = model.predict_proba(model_input)[0]
pred_idx = int(np.argmax(probs))

shap_info = compute_shap_explanation(model_input, pred_idx)
print(shap_info)

# Let's see raw phi values
EXPECTED_FEATURES = model.get_booster().feature_names
model_input = model_input.reindex(columns=EXPECTED_FEATURES).fillna(0.0)
shap_values = explainer(model_input)
class_shap = shap_values.values[0, :, pred_idx]
print("\nRaw PHI values:")
for i, col in enumerate(EXPECTED_FEATURES):
    if class_shap[i] != 0:
        print(f"{col}: {class_shap[i]}")
