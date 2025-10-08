import streamlit as st
import pickle
import os
import numpy as np
from streamlit_option_menu import option_menu  # Nouveau module pour gérer les icônes dans la barre latérale

# Définir la page de titre
st.set_page_config(page_title="PredictiveCare", page_icon="💉", layout="centered")

# Charger les modèles
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'diabetes_model.sav'), 'rb'))
heart_disease_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'heart_disease_model.sav'), 'rb'))
parkinsons_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'parkinsons_model.sav'), 'rb'))

# Menu de navigation avec icônes dans la barre latérale
with st.sidebar:
    page = option_menu(
        "Système de Prédiction de Maladies Multiples",
        ["Accueil", "Prédiction de Diabète", "Prédiction de Maladies Cardiaques", "Prédiction de Parkinson", "Prédiction d'Asthme"],
        icons=["house", "activity", "heart", "person","lungs"],
        menu_icon="cast",
        default_index=0,
    )

# Page d'accueil
if page == "Accueil":
    st.markdown("<h1 style='text-align: center;'>Bienvenue sur PredictiveCare</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)  # Ouvrir une balise div centrée
    st.write(""" 
        Ce système permet de prédire plusieurs maladies à l'aide de modèles de machine learning. 
        Sélectionnez une option dans la barre latérale pour commencer.
    """)
    st.image("C:/Users/Vahoaka/multiple-disease-prediction-streamlit-app-main/image/predict.png", use_column_width=True)
    st.write("### Instructions :")
    st.write("1. Sélectionnez le type de maladie à prédire dans la barre latérale.")
    st.write("2. Remplissez les informations demandées dans le formulaire.")
    st.write("3. Cliquez sur le bouton pour obtenir le résultat de la prédiction.")
    st.markdown("</div>", unsafe_allow_html=True)  # Fermer la balise div centrée

# Page de prédiction du diabète
elif page == "Prédiction de Diabète":
    st.title("Prédiction du diabète")

    # Créer deux colonnes pour le formulaire de diabète
    col1, col2 = st.columns(2)

    with col1:
        grossesse = st.number_input("Nombre de Grossesses", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Niveau de glucose", min_value=0, max_value=200, step=1)
        pression = st.number_input("Pression artérielle (mmHg)", min_value=0, max_value=200, step=1)

    with col2:
        epaisseur_peau = st.number_input("Épaisseur de peau (mm)", min_value=0, max_value=100, step=1)
        insuline = st.number_input("Taux d'insuline (mu U/ml)", min_value=0, max_value=900, step=1)
        imc = st.number_input("Indice de Masse Corporelle (IMC)", min_value=0.0, max_value=100.0, step=0.1)
        pedigree = st.number_input("Fonction de Généalogie du Diabète", min_value=0.0, max_value=3.0, step=0.01,
                                   format="%.2f")
        age = st.number_input("Âge", min_value=0, max_value=120, step=1)

    # Bouton pour lancer la prédiction
    if st.button("Résultat du test de diabète"):
        # Préparation des données d'entrée pour le modèle
        input_data = np.array([[grossesse, glucose, pression, epaisseur_peau, insuline, imc, pedigree, age]])
        # Faire la prédiction
        prediction = diabetes_model.predict(input_data)
        st.success(f"Résultat de la prédiction du diabète : {'Diabétique' if prediction[0] == 1 else 'Non diabétique'}")

# Page de prédiction des maladies cardiaques
elif page == "Prédiction de Maladies Cardiaques":
    st.title("Prédiction des maladies cardiaques")

    # Créer deux colonnes pour le formulaire des maladies cardiaques
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", min_value=0, max_value=120, step=1)
        sexe = st.selectbox("Sexe (0 = Femme, 1 = Homme)", [0, 1])
        douleur_thoracique = st.number_input(
            "Type de douleur thoracique (0 = asymptomatique, 1 = douleur typique, 2 = douleur atypique, 3 = non angineuse)",
            min_value=0, max_value=3, step=1)
        pression_repos = st.number_input("Pression Artérielle au Repos (mmHg)", min_value=0, max_value=200, step=1)

    with col2:
        cholestérol = st.number_input("Cholestérol (mg/dl)", min_value=0, max_value=600, step=1)
        sucre = st.selectbox("Sucre à Jeun > 120 mg/dl (0 = Non, 1 = Oui)", options=[0, 1])
        ecg = st.selectbox(
            "Résultats Électrocardiographiques (0 = normal, 1 = anomalie onde ST, 2 = hypertrophie ventriculaire)",
            options=[0, 1, 2])
        fréquence_max = st.number_input("Fréquence cardiaque max", min_value=0, max_value=300, step=1)
        angine = st.selectbox("Angine Induite par Exercice (0 = Non, 1 = Oui)", options=[0, 1])
        oldpeak = st.number_input("Dépression ST induite par l'exercice par rapport au repos", min_value=0.0,
                                  max_value=10.0, step=0.1)
        pente = st.selectbox("Pente du Segment ST (0 = pente ascendante, 1 = plate, 2 = descendante)",
                             options=[0, 1, 2])
        vaisseaux = st.number_input("Nombre de vaisseaux principaux colorés par fluoroscopie (0-3)", min_value=0,
                                    max_value=3, step=1)
        thalassémie = st.selectbox("Thalassémie (1 = normal, 2 = défaut fixe, 3 = réversible)", options=[1, 2, 3])

    # Bouton pour lancer la prédiction
    if st.button("Résultat du test de maladies cardiaques"):
        # Préparation des données d'entrée pour le modèle
        input_data = np.array([[age, sexe, douleur_thoracique, pression_repos, cholestérol, sucre, ecg, fréquence_max,
                                angine, oldpeak, pente, vaisseaux, thalassémie]])
        # Faire la prédiction
        prediction = heart_disease_model.predict(input_data)
        st.success(
            f"Résultat de la prédiction des maladies cardiaques : {'Malade' if prediction[0] == 1 else 'Non malade'}")

# Page de prédiction de Parkinson
elif page == "Prédiction de Parkinson":
    st.title("Prédiction de Parkinson")

    # Créer deux colonnes pour le formulaire de Parkinson
    col1, col2 = st.columns(2)

    with col1:
        mdvp_fo = st.number_input("MDVP: Fo (Hz)", min_value=0.0, format="%.4f")
        mdvp_fhi = st.number_input("MDVP: Fhi (Hz)", min_value=0.0, format="%.4f")
        mdvp_flo = st.number_input("MDVP: Flo (Hz)", min_value=0.0, format="%.4f")
        mdvp_jitter_percent = st.number_input("MDVP: Jitter (%)", min_value=0.0, format="%.4f")
        mdvp_jitter_abs = st.number_input("MDVP: Jitter (Abs)", min_value=0.0, format="%.4f")
        mdvp_rap = st.number_input("MDVP: RAP", min_value=0.0, format="%.4f")
        mdvp_ppq = st.number_input("MDVP: PPQ", min_value=0.0, format="%.4f")

