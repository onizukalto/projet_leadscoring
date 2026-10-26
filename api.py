from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Prospect(BaseModel):
    secteur: str
    pays: str
    plateforme: str
    nb_employes: float
    anciennete_annees: float
    trafic_mensuel: int
    taux_conversion: float
    panier_moyen: float
    ca_annuel_estime_k: float
    nb_produits: int
    source_lead: str
    poste_contact: str
    pages_visitees: int
    temps_sur_site_min: float
    a_telecharge_livre_blanc: int
    a_demande_demo: int
    nb_emails_ouverts: int
    jours_depuis_dernier_contact: int
    utilise_concurrent: int


import joblib
import pandas as pd

saved = joblib.load("lead_model.joblib")
model, columns = saved["model"], saved["columns"]

@app.post("/score")
def scorer(prospect: Prospect):
    x = pd.DataFrame([prospect.model_dump()])
    x = pd.get_dummies(x, dtype=int)
    x = x.reindex(columns=columns, fill_value=0)
    score = model.predict_proba(x)[0, 1]
    return {"score": round(float(score), 3)}

