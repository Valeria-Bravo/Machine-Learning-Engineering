import pandas as pd
from pycaret.classification import *
import mlflow
from scipy import interp

df = pd.read_csv("fintech_credit_approval.csv")

columnas_id = ["customer_id", "id", "client_id"]
df_model = df.drop(columns=[c for c in columnas_id if c in df.columns], errors="ignore")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fintech_credit_approval")

s = setup(
    data=df_model,
    target="approved",
    session_id=123,
    log_experiment=True,
    experiment_name="fintech_credit_approval",
    verbose=True,
    profile=False,
    use_gpu=False
)

best_model = compare_models(turbo=True)

# evaluate_model(best_model)   # ❌ ESTA LÍNEA BLOQUEA VS CODE

save_model(best_model, "modelos/modelo_final")

print("✔ Modelo entrenado y guardado en modelos/modelo_final.pkl")
