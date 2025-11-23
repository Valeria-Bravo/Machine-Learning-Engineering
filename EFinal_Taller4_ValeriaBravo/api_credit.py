from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

app = FastAPI()

model = load_model("modelos/modelo_credito")

@app.get("/")
def home():
    return {"status": "API de crédito funcionando correctamente"}

class DatosEntrada(BaseModel):
    age: int
    monthly_income_usd: float
    app_usage_score: float
    digital_profile_strength: float
    num_contacts_uploaded: int
    residence_risk_zone: str
    political_event_last_month: int
    threshold: float

@app.post("/predict")
def predict(data: DatosEntrada):

    df = pd.DataFrame([data.dict()])


    df_model = df.drop(columns=["threshold"])
     
    # OBJETIVO 1: Prediga si un cliente será aprobado (1) o no (0)
    prediction = predict_model(model, data=df_model)

    prob = float(prediction["prediction_score"][0])
    pred_label = int(prediction["prediction_label"][0])

    # OBJETIVO 3: Exponer el modelo vía API con threshold ajustable
    pred_final = 1 if prob >= data.threshold else 0

    return {
        "input": df_model.to_dict(orient="records")[0],
        "probabilidad_modelo": prob,
        "prediccion_modelo": pred_label,
        "prediccion_final_con_threshold": pred_final
    }
