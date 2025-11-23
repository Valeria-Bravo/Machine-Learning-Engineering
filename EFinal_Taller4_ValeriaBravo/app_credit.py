import streamlit as st
import requests

st.title("Predicción de Crédito Digital")

age = st.number_input("Edad", 18, 80, 30)
income = st.number_input("Ingreso mensual USD", 0.0, 5000.0, 800.0)
app_score = st.slider("Uso de app (0–10)", 0.0, 10.0, 5.0)
digital = st.slider("Perfil digital (0–100)", 0, 100, 50)
contacts = st.number_input("Contactos subidos", 0, 500, 50)
risk = st.selectbox("Zona de riesgo", ["baja", "media", "alta"])
event = st.radio("Evento político reciente", [0, 1])
threshold = st.slider("Threshold", 0.0, 1.0, 0.5)

if st.button("Predecir"):

    payload = {
        "age": age,
        "monthly_income_usd": income,
        "app_usage_score": app_score,
        "digital_profile_strength": digital,
        "num_contacts_uploaded": contacts,
        "residence_risk_zone": risk,
        "political_event_last_month": event,
        "threshold": threshold
    }

    r = requests.post("http://localhost:8000/predict", json=payload)
    #OBJETIVO 4: Visualizar score y riesgo desde una interfaz en Streamlit
    if r.status_code == 200:
        st.json(r.json())
    else:
        st.error("Error en la API")
