import pandas as pd
from pycaret.classification import *

df = pd.read_csv(r"C:\Users\vbravo\Downloads\EFinal_Taller4_ValeriaBravo\modelos\fintech_credit_approval.csv")

df_model = df.drop(columns=["user_id"])
#OBJETIVO 2: Registrar interpretabilidad SHAP en MLflow
s = setup(
    data=df_model,
    target="approved",
    session_id=123,
    verbose=True
)

best_model = compare_models()
evaluate_model(best_model)
save_model(best_model, "modelo_credito")

print("Modelo entrenado y guardado")
