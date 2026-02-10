# Exécution rapide de l'application PredictiveCare

Ce petit guide explique comment installer les dépendances et lancer l'application Streamlit fournie dans ce dépôt.

## Prérequis
- Python 3.8+ installé
- Accès au dossier `multiple-disease-prediction-streamlit-app-main/saved_models/` contenant les modèles sérialisés:
  - `diabetes_model.sav`
  - `heart_disease_model.sav`
  - `parkinsons_model.sav`
  - `asthme.sav` 

## Installation (recommandé: environnement virtuel)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'application
 (Linux) :
```bash
streamlit run "multiple-disease-prediction-streamlit-app-main/app111.py"
```

Remarques:
- l appersue de l'app est dans le result 