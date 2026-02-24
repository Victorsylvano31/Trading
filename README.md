# 🩺 PredictiveCare - Système de Diagnostic Prédictif via l'IA

**PredictiveCare** est une application web intelligente conçue pour aider à la détection précoce de plusieurs maladies chroniques en utilisant la puissance du Machine Learning. Grâce à une interface intuitive, les utilisateurs peuvent obtenir des prédictions instantanées basées sur leurs données de santé.

---

## 🚀 Fonctionnalités
- **Analyse Multi-Maladies** : Diagnostic prédictif pour le Diabète, les Maladies Cardiaques, la Maladie de Parkinson et l'Asthme.
- **Interface Intuitive** : Navigation fluide via une barre latérale moderne (Streamlit).
- **Éducation Santé** : Sections d'aide détaillées expliquant chaque paramètre médical.
- **Conseils Personnalisés** : Recommandations basées sur les résultats des prédictions.

## 🛠️ Installation et Lancement

### 1. Prérequis
- **Python 3.8** ou version ultérieure installé sur votre système.

### 2. Clonage du projet
```bash
git clone https://github.com/Victorsylvano31/PredictiveCare-.git
cd PredictiveCare-
```

### 3. Installation des dépendances
Il est recommandé d'utiliser un environnement virtuel :
```bash
python -m venv .venv
# Sur Windows :
.venv\Scripts\activate
# Sur Linux/Mac :
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Configuration des Modèles (.sav)
> [!IMPORTANT]
> Pour que les prédictions fonctionnent, vous devez placer les fichiers de modèles entraînés dans le dossier suivant :
> `multiple-disease-prediction-streamlit-app-main/saved_models/`
>
> **Fichiers requis :**
> - `diabetes_model.sav`
> - `heart_disease_model.sav`
> - `parkinsons_model.sav`
> - `asthme.sav`

### 5. Lancer l'application
```bash
streamlit run "multiple-disease-prediction-streamlit-app-main/app111.py"
```

---

## 📁 Structure du Projet
- `app111.py` : Cœur de l'application (Interface et Logique).
- `saved_models/` : Répertoire contenant les modèles de Machine Learning.
- `dataset/` : Jeux de données utilisés pour l'entraînement.
- `requirements.txt` : Liste des dépendances Python.

---

## 📩 Contact
Développé par **Victor Sylvano**.
- **Facebook** : [Victor Sylvano](https://www.facebook.com/Victorsylvano.sylvano)
- **Email** : [victorsylvano31@gmail.com](mailto:victorsylvano31@gmail.com)

---
*" L'intelligence artificielle au service de la santé "*
