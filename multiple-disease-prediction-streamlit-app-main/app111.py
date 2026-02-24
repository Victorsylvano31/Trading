import streamlit as st
import pickle
import os
import numpy as np
from streamlit_option_menu import option_menu  # Nouveau module pour gérer les icônes dans la barre latérale

# Définir la page de titre
st.set_page_config(page_title="PredictiveCare", page_icon="🩺", layout="centered")

# Charger les modèles
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(file_name):
    path = os.path.join(working_dir, 'saved_models', file_name)
    try:
        return pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        st.error(f"Erreur : Le modèle '{file_name}' est introuvable au chemin : {path}")
        return None
    except Exception as e:
        st.error(f"Une erreur est survenue lors du chargement de '{file_name}' : {e}")
        return None

diabetes_model = load_model('diabetes_model.sav')
heart_disease_model = load_model('heart_disease_model.sav')
parkinsons_model = load_model('parkinsons_model.sav')
asthma_model = load_model('asthme.sav')

# Menu de navigation avec icônes dans la barre latérale
with st.sidebar:
    page = option_menu(
        "Prédiction et Prévention des Maladies",
        ["Accueil", "Prédiction de Diabète", "Prédiction de Maladies Cardiaques", "Prédiction de Parkinson","Prédiction d'Asthme","Aide"],
        icons=["house", "activity", "heart", "person", "lungs","info"],
        menu_icon="cast",
        default_index=0,
    )

if page == "Accueil":
    # Titre centré
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Bienvenue sur PredictiveCare</h1>", unsafe_allow_html=True)

    # Texte centré
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)  # Ouvrir une balise div centrée
    st.write("""
        <h3 style='color: #333;'>Prédisez votre santé grâce au Machine Learning</h3>
        <p style='font-size: 18px;'>Ce système utilise des modèles de machine learning avancés pour prédire plusieurs maladies 
        et vous aider à prendre des décisions éclairées sur votre santé.</p>
        <p style='font-size: 18px;'>Sélectionnez une option dans la barre latérale pour commencer votre voyage vers une santé meilleure.</p>
    """, unsafe_allow_html=True)

    # Image ou illustration (ajoutez votre propre image)
    logo_path = os.path.join(working_dir, 'image', 'ccc.png')
    if os.path.exists(logo_path):
        st.image(logo_path, caption="'' L'intelligence artificielle au service de la santé ''", use_column_width=True)
    else:
        st.info("Image d'illustration non trouvée.")

    st.markdown("</div>", unsafe_allow_html=True)  # Fermer la balise div


    st.write("### Instructions :")
    st.write("1. Sélectionnez le type de maladie à prédire dans la barre latérale.")
    st.write("2. Remplissez les informations demandées dans le formulaire.")
    st.write("3. Cliquez sur le bouton pour obtenir le résultat de la prédiction.")

    st.markdown("</div>", unsafe_allow_html=True)  # Fermer la balise div centrée

    # Explications des maladies
    st.write("### Informations sur les maladies :")

    # Maladie Cardiaque
    st.write("#### Maladie Cardiaque :")
    st.write("Description :")
    st.write("""
        Les maladies cardiaques regroupent un ensemble de troubles affectant le cœur et les vaisseaux sanguins. 
        Les types les plus courants incluent :
        - **Cardiopathie Ischémique** : Lorsque le cœur ne reçoit pas suffisamment de sang, généralement à cause de l'accumulation de plaque dans les artères coronaires, ce qui peut entraîner des crises cardiaques.
        - **Insuffisance Cardiaque** : Une condition où le cœur ne peut pas pomper suffisamment de sang pour répondre aux besoins du corps, entraînant fatigue, essoufflement et œdème.
        - **Arythmies** : Des troubles du rythme cardiaque causant des battements irréguliers.
        - **Maladies Valvulaires** : Affectent les valves cardiaques, perturbant le flux sanguin normal.
    """, unsafe_allow_html=True)

    # Diabète
    st.write("#### Diabète :")
    st.write("""
        Le diabète est une maladie chronique où le corps ne peut pas produire ou utiliser efficacement l'insuline. 
        Les symptômes incluent une soif excessive, des mictions fréquentes, et une fatigue accrue. 
        Une gestion appropriée comprend une alimentation équilibrée, de l'exercice et, si nécessaire, des médicaments.
    """)

    # Maladie de Parkinson
    st.write("#### Maladie de Parkinson :")
    st.write("""
        La maladie de Parkinson est un trouble neurologique progressif affectant principalement le mouvement. 
        Les symptômes incluent des tremblements, une rigidité musculaire, et des difficultés à marcher. 
        Bien qu'il n'existe pas de cure, des traitements comme des médicaments et la thérapie physique peuvent aider à gérer les symptômes.
    """)
    # Asthme
    st.write ( "#### Asthme :" )
    st.write ( """
            L'asthme est une maladie respiratoire chronique caractérisée par une inflammation des voies respiratoires, 
            rendant la respiration difficile. Les symptômes incluent une toux, des sifflements, un essoufflement et une oppression thoracique. 
            Les déclencheurs courants peuvent être des allergènes, des infections, la pollution de l'air et l'exercice physique. 
            Une gestion appropriée de l'asthme inclut l'utilisation de médicaments tels que les bronchodilatateurs et les anti-inflammatoires, 
            ainsi que l'évitement des déclencheurs connus et le suivi régulier de la fonction pulmonaire.
        """ )

    # Utilité Générale de l'Application
    st.write("### Utilité Générale de l'Application :")
    st.write("""
        - **Prédiction Précoce** : Identifie les risques avant que les symptômes ne deviennent graves.
        - **Facilité d'Accès** : Interface conviviale permettant une saisie rapide des données.
        - **Conseils Personnalisés** : Suggestions basées sur les résultats pour encourager des consultations médicales.
        - **Éducation** : Sensibilisation sur les maladies et les mesures préventives.
    """)


# Page de prédiction du diabète
elif page == "Prédiction de Diabète":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prédiction du diabète 🩸</h1>", unsafe_allow_html=True)

    # Créer un formulaire pour entrer les données dans un expander
    with st.expander ( "Informations sur le diabète", expanded=True ):
        grossesse = st.number_input ( "Nombre de Grossesses", min_value=0, max_value=20, step=1 )

        glucose = st.number_input ( "Niveau de glucose", min_value=0, max_value=200, step=1 )

        pression = st.number_input ( "Pression artérielle (mmHg)", min_value=0, max_value=200, step=1 )

        epaisseur_peau = st.number_input ( "Épaisseur de peau (mm)", min_value=0, max_value=100, step=1 )

        insuline = st.number_input ( "Taux d'insuline (mu U/ml)", min_value=0, max_value=900, step=1 )

        imc = st.number_input ( "Indice de Masse Corporelle (IMC)", min_value=0.0, max_value=100.0, step=0.1 )

        pedigree = st.number_input ( "Fonction de Généalogie du Diabète", min_value=0.0, max_value=3.0, step=0.01,
                                     format="%.2f" )

        age = st.number_input("Âge", min_value=0, max_value=120, step=1)

    # Bouton pour lancer la prédiction
    if st.button("Résultat du test de diabète"):
        # Préparation des données d'entrée pour le modèle
        input_data = np.array([[
            grossesse, glucose, pression, epaisseur_peau,
            insuline, imc, pedigree, age
        ]])

        # Faire la prédiction
        prediction = diabetes_model.predict(input_data)

        # Correspondance avec les classes et affichage du résultat
        if prediction[0] == 1:  # Si l'utilisateur est diabétique
            st.success("Résultat de la prédiction du diabète : Diabétique")
            st.warning(
                "Conseil : Il est recommandé de consulter un professionnel de santé pour un suivi régulier et discuter des options de traitement.")
        else:  # Si l'utilisateur n'est pas diabétique
            st.success("Résultat de la prédiction du diabète : Non diabétique")
            st.info(
                "Conseil : Continuez à adopter un mode de vie sain en suivant une alimentation équilibrée et en faisant de l'exercice régulièrement.")


##################################################################################################""

# Page de prédiction des maladies cardiaques
elif page == "Prédiction de Maladies Cardiaques":
    st.markdown ( "<h1 style='text-align: center; color: #4CAF50;'>Prédiction des maladies cardiaques ❤️</h1>",
                  unsafe_allow_html=True )
    # Formulaire de saisie pour la prédiction des maladies cardiaques
    with st.expander ( "Informations sur la santé cardiaque", expanded=True ):
        age = st.number_input ( "Âge", min_value=0, max_value=120, step=1, value=45 )
        sexe = st.selectbox ( "Sexe (0 = Femme, 1 = Homme)", [0, 1], index=1 )
        douleur_thoracique = st.selectbox (
            "Type de douleur thoracique (0 = asymptomatique, 1 = douleur typique, 2 = douleur atypique, 3 = non angineuse)",
            [0, 1, 2, 3], index=1
        )
        pression_repos = st.number_input ( "Pression Artérielle au Repos (mmHg)", min_value=0, max_value=200, step=1,
                                           value=120 )
        cholestérol = st.number_input ( "Cholestérol (mg/dl)", min_value=0, max_value=600, step=1, value=200 )
        sucre = st.selectbox ( "Sucre à Jeun > 120 mg/dl (0 = Non, 1 = Oui)", options=[0, 1], index=0 )
        ecg = st.selectbox (
            "Résultats Électrocardiographiques (0 = normal, 1 = anomalie onde ST, 2 = hypertrophie ventriculaire)",
            options=[0, 1, 2], index=0 )
        fréquence_max = st.number_input ( "Fréquence cardiaque max", min_value=0, max_value=300, step=1, value=150 )
        angine = st.selectbox ( "Angine Induite par Exercice (0 = Non, 1 = Oui)", options=[0, 1], index=0 )
        oldpeak = st.number_input ( "Dépression ST induite par l'exercice par rapport au repos", min_value=0.0,
                                    max_value=10.0, step=0.1, value=0.0 )
        pente = st.selectbox ( "Pente du Segment ST (0 = pente ascendante, 1 = plate, 2 = descendante)",
                               options=[0, 1, 2], index=1 )
        vaisseaux = st.number_input ( "Nombre de vaisseaux principaux colorés par fluoroscopie (0-3)", min_value=0,
                                      max_value=3, step=1, value=1 )
        thalassémie = st.selectbox ( "Thalassémie (1 = normal, 2 = défaut fixe, 3 = réversible)", options=[1, 2, 3],
                                     index=0 )

    # Bouton pour lancer la prédiction
    if st.button("Résultat du test de maladies cardiaques"):
        # Préparation des données d'entrée pour le modèle
        input_data = np.array([[
            age, sexe, douleur_thoracique, pression_repos,
            cholestérol, sucre, ecg, fréquence_max,
            angine, oldpeak, pente, vaisseaux, thalassémie
        ]])

        # Faire la prédiction
        prediction = heart_disease_model.predict(input_data)

        # Affichage du résultat
        if prediction[0] == 1:  # Si l'utilisateur est malade
            st.success("Résultat de la prédiction des maladies cardiaques : Malade")
            st.warning("Conseil : Il est recommandé de consulter un professionnel de santé pour un suivi adapté.")
        else:  # Si l'utilisateur n'est pas malade
            st.success("Résultat de la prédiction des maladies cardiaques : Non malade")
            st.info(
                "Conseil : Continuez à adopter un mode de vie sain en suivant une alimentation équilibrée et en faisant de l'exercice régulièrement.")

        #####################################################################################################################################

# Page de prédiction de Parkinson
elif page == "Prédiction de Parkinson":
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Prédiction de Parkinson 🧠</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
   # Formulaire de saisie pour la prédiction de Parkinson
    with st.expander ( "Mesures MDVP", expanded=True ):
        col1, col2 = st.columns ( 2 )
        with col1:
            mdvp_fo = st.number_input ( "MDVP: Fo (Hz)", min_value=0.0, format="%.4f" )
            mdvp_fhi = st.number_input ( "MDVP: Fhi (Hz)", min_value=0.0, format="%.4f" )
            mdvp_flo = st.number_input ( "MDVP: Flo (Hz)", min_value=0.0, format="%.4f" )
            mdvp_jitter_percent = st.number_input ( "MDVP: Jitter (%)", min_value=0.0, format="%.4f" )
            mdvp_jitter_abs = st.number_input ( "MDVP: Jitter (Abs)", min_value=0.0, format="%.4f" )
            mdvp_rap = st.number_input ( "MDVP: RAP", min_value=0.0, format="%.4f" )

        with col2:
            mdvp_ppq = st.number_input ( "MDVP: PPQ", min_value=0.0, format="%.4f" )
            jitter_ddp = st.number_input ( "Jitter: DDP", min_value=0.0, format="%.4f" )
            mdvp_shimmer = st.number_input ( "MDVP: Shimmer", min_value=0.0, format="%.4f" )
            mdvp_shimmer_db = st.number_input ( "MDVP: Shimmer (dB)", min_value=0.0, format="%.4f" )
            shimmer_apq3 = st.number_input ( "Shimmer: APQ3", min_value=0.0, format="%.4f" )

    with st.expander ( "Mesures Shimmer", expanded=False ):
        col1, col2 = st.columns ( 2 )
        with col1:
            shimmer_apq5 = st.number_input ( "Shimmer: APQ5", min_value=0.0, format="%.4f" )
            mdvp_apq = st.number_input ( "MDVP: APQ", min_value=0.0, format="%.4f" )
            shimmer_dda = st.number_input ( "Shimmer: DDA", min_value=0.0, format="%.4f" )

        with col2:
            nhr = st.number_input ( "NHR", min_value=0.0, format="%.4f" )
            hnr = st.number_input ( "HNR", min_value=0.0, format="%.4f" )
            rpde = st.number_input ( "RPDE", min_value=0.0, format="%.4f" )

    with st.expander ( "Mesures DFA et Spread", expanded=False ):
        col1, col2 = st.columns ( 2 )
        with col1:
            dfa = st.number_input ( "DFA", min_value=0.0, format="%.4f" )
            spread1 = st.number_input ( "Spread1", min_value=0.0, format="%.4f" )

        with col2:
            spread2 = st.number_input ( "Spread2", min_value=0.0, format="%.4f" )
            d2 = st.number_input ( "D2", min_value=0.0, format="%.4f" )

    with st.expander ( "Mesures PPE", expanded=False ):
        ppe = st.number_input ( "PPE", min_value=0.0, format="%.4f" )

    # CSS pour centrer le bouton
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            display: block;
            margin: 0 auto;  /* Centrer le bouton */
        }
        </style>
        """, unsafe_allow_html=True)

    # Bouton pour lancer la prédiction
    if st.button("Résultat du test de Parkinson"):
        # Préparation des données d'entrée pour le modèle
        input_data = np.array([[
            mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent,
            mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp,
            mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3,
            shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ]])
        # Faire la prédiction
        prediction = parkinsons_model.predict(input_data)

        # Correspondance avec les classes et affichage du résultat
        if prediction[0] == 1:
            st.success("Résultat de la prédiction de Parkinson : Présent")
            st.warning("Conseil : Il est conseillé de consulter un professionnel de santé pour un suivi adapté.")
        else:
            st.success("Résultat de la prédiction de Parkinson : Absent")
            st.info("Conseil : Continuez à mener une vie active et saine.")

# Page de prédiction de l'asthme
elif page == "Prédiction d'Asthme":
        st.title ( "Prédiction d'Asthme 💨" )

        # Exemple de données à préremplir
        example_data = {
            "Age": 63,
            "Gender": 0,  # Use numeric representation (0 for Homme, 1 for Femme)
            "Ethnicity": 1,
            "EducationLevel": 0,
            "BMI": 15.8487444,
            "Smoking": 0,
            "PhysicalActivity": 0.894448309,
            "DietQuality": 5.488695585,
            "SleepQuality": 8.701002734,
            "PollutionExposure": 7.388480567,
            "PollenExposure": 2.855577785,
            "DustExposure": 0.974339383,
            "PetAllergy": 1,
            "FamilyHistoryAsthma": 1,
            "HistoryOfAllergies": 0,
            "Eczema": 0,
            "HayFever": 0,
            "GastroesophagealReflux": 0,
            "LungFunctionFEV1": 1.3690512,
            "LungFunctionFVC": 4.941205661,
            "Wheezing": 0,
            "ShortnessOfBreath": 0,
            "ChestTightness": 1,
            "Coughing": 0,
            "NighttimeSymptoms": 0,
            "ExerciseInduced": 1,
        }

        # Créer des sections pour le formulaire
        with st.expander ( "Informations personnelles", expanded=True ):
            Age = st.number_input ( "Âge", min_value=0, max_value=120, step=1, value=example_data["Age"] )
            Gender = st.selectbox ( "Genre (0 Homme, 1 Femme)", options=[0, 1], format_func=lambda x: "Homme" if x == 0 else "Femme", index=example_data["Gender"] )
            # Utilisation correcte de l'index pour l'ethnicité
            Ethnicity = st.selectbox ( "Ethnicité :0 - 1 - 2", ["0", "1", "2"],
                                       index=int ( example_data["Ethnicity"] ) )
            EducationLevel = st.selectbox ( "Niveau d'éducation", [0, 1, 2, 3, 4],index=example_data["EducationLevel"] )
            BMI = st.number_input ( "Indice de Masse Corporelle (BMI)", min_value=0.0, max_value=100.0, step=0.1,value=example_data["BMI"] )
            Smoking = st.selectbox ( "Fumeur", [0, 1], index=example_data["Smoking"] )

        with st.expander ( "Exposition et Qualité de Vie", expanded=False ):
            PhysicalActivity = st.number_input ( "Activité physique (heures par semaine)", min_value=0.0, step=0.1,
                                                 value=example_data["PhysicalActivity"] )
            DietQuality = st.slider ( "Qualité de l'alimentation", min_value=0.0, max_value=10.0, step=0.1,
                                      value=example_data["DietQuality"] )
            SleepQuality = st.slider ( "Qualité du sommeil", min_value=0.0, max_value=10.0, step=0.1,
                                       value=example_data["SleepQuality"] )
            PollutionExposure = st.slider ( "Exposition à la pollution", min_value=0.0, max_value=10.0, step=0.1,
                                            value=example_data["PollutionExposure"] )
            PollenExposure = st.slider ( "Exposition au pollen", min_value=0.0, max_value=10.0, step=0.1,
                                         value=example_data["PollenExposure"] )
            DustExposure = st.slider ( "Exposition à la poussière", min_value=0.0, max_value=10.0, step=0.1,
                                       value=example_data["DustExposure"] )

        with st.expander ( "Antécédents médicaux", expanded= False ):
            PetAllergy = st.selectbox ( "Allergie aux animaux", [0, 1], index=example_data["PetAllergy"] )
            FamilyHistoryAsthma = st.selectbox ( "Antécédents familiaux d'asthme", [0, 1],
                                                 index=example_data["FamilyHistoryAsthma"] )
            HistoryOfAllergies = st.selectbox ( "Antécédents d'allergies", [0, 1],
                                                index=example_data["HistoryOfAllergies"] )
            Eczema = st.selectbox ( "Eczéma", [0, 1], index=example_data["Eczema"] )
            HayFever = st.selectbox ( "Rhume des foins", [0, 1], index=example_data["HayFever"] )
            GastroesophagealReflux = st.selectbox ( "Reflux gastro-œsophagien", [0, 1],
                                                    index=example_data["GastroesophagealReflux"] )

        with st.expander ( "Fonction pulmonaire et Symptômes", expanded= False ):
            LungFunctionFEV1 = st.number_input ( "Lung Function FEV1 (L)", min_value=0.0, step=0.1,
                                                 value=example_data["LungFunctionFEV1"] )
            LungFunctionFVC = st.number_input ( "Lung Function FVC (L)", min_value=0.0, step=0.1,
                                                value=example_data["LungFunctionFVC"] )
            Wheezing = st.selectbox ( "Sifflements respiratoires", [0, 1], index=example_data["Wheezing"] )
            ShortnessOfBreath = st.selectbox ( "Essoufflement", [0, 1], index=example_data["ShortnessOfBreath"] )
            ChestTightness = st.selectbox ( "Oppression thoracique", [0, 1], index=example_data["ChestTightness"] )
            Coughing = st.selectbox ( "Toux", [0, 1], index=example_data["Coughing"] )
            NighttimeSymptoms = st.selectbox ( "Symptômes nocturnes", [0, 1], index=example_data["NighttimeSymptoms"] )
            ExerciseInduced = st.selectbox ( "Induit par l'exercice", [0, 1], index=example_data["ExerciseInduced"] )

        # CSS pour centrer le bouton
        st.markdown ( """
            <style>
            div.stButton > button:first-child {
                display: block;
                margin: 0 auto;  /* Centrer le bouton */
            }
            </style>
            """, unsafe_allow_html=True )

        # Bouton pour lancer la prédiction
        if st.button ( "Résultat de la prédiction d'asthme", key="predict_asthma" ):
            # Préparation des données d'entrée pour le modèle
            patient_data = [
                int ( Age ),
                int ( Gender ),  # Déjà converti par le selectbox
                int ( Ethnicity ),
                int ( EducationLevel ),
                float ( BMI ),
                int ( Smoking ),
                float ( PhysicalActivity ),
                float ( DietQuality ),
                float ( SleepQuality ),
                float ( PollutionExposure ),
                float ( PollenExposure ),
                float ( DustExposure ),
                int ( PetAllergy ),
                int ( FamilyHistoryAsthma ),
                int ( HistoryOfAllergies ),
                int ( Eczema ),
                int ( HayFever ),
                int ( GastroesophagealReflux ),
                float ( LungFunctionFEV1 ),
                float ( LungFunctionFVC ),
                int ( Wheezing ),
                int ( ShortnessOfBreath ),
                int ( ChestTightness ),
                int ( Coughing ),
                int ( NighttimeSymptoms ),
                int ( ExerciseInduced )
            ]

            # Convertir en tableau NumPy pour être compatible avec le modèle
            input_data = np.array ( [patient_data], dtype=float )

            # Faire la prédiction
            prediction = asthma_model.predict ( input_data )

            # Afficher le résultat de la prédiction
            if prediction[0] == 1:
                st.success ( "Résultat de la prédiction de l'asthme : Asthmatique" )
                st.warning("Après avoir été diagnostiqué asthmatique, consultez régulièrement votre médecin pour un suivi et un ajustement de votre traitement. Évitez les allergènes connus et gardez toujours votre inhalateur de secours à portée de main.")
            else:
                st.success ( "Résultat de la prédiction de l'asthme : Non asthmatique" )
                st.info ("continuez à mener un mode de vie actif et sain. Restez attentif aux facteurs environnementaux qui pourraient déclencher des allergies et consultez un professionnel de santé si des symptômes respiratoires apparaissent.")

# Page d'aide
elif page == "Aide":
            st.markdown ( "<h1 style='text-align: center;'>Page d'aide 💡 </h1>", unsafe_allow_html=True )

            st.markdown ( "<h2 ;'>Prédiction du Diabète </h2>", unsafe_allow_html=True )
                # Informations sur les paramètres
            st.write ( "### Explications des Valeurs :" )
            st.write ( "#### 1. Nombre de Grossesses" )
            st.write ( """
                Ce paramètre représente le nombre total de grossesses qu'une femme a eues.
                Un nombre élevé de grossesses peut être associé à un risque accru de diabète gestationnel, 
                une condition qui peut survenir pendant la grossesse et augmenter le risque de diabète de type 2 plus tard dans la vie.
                """ )
            st.write ( "#### 2. Niveau de Glucose" )
            st.write ( """
                Il s'agit du taux de glucose dans le sang, mesuré en milligrammes par décilitre (mg/dl).
                Un niveau élevé de glucose est un indicateur clé du diabète. 
                En général, un taux de glucose à jeun supérieur à 126 mg/dl est considéré comme un signe de diabète.
                """ )
            st.write ( "#### 3. Pression Artérielle (mmHg)" )
            st.write ( """
                C'est la mesure de la pression sanguine, exprimée en millimètres de mercure (mmHg).
                Une pression artérielle élevée est souvent associée à des problèmes de santé, 
                y compris le diabète, car elle peut entraîner des complications cardiaques.
                """ )
            st.write ( "#### 4. Épaisseur de Peau (mm)" )
            st.write ( """
                Ce paramètre mesure l'épaisseur de la peau, souvent à l'aide d'un dispositif à ultrasons, 
                et est généralement pris au niveau du triceps.
                Une épaisseur de peau accrue peut indiquer une résistance à l'insuline, 
                un facteur de risque pour le diabète.
            """ )
            st.write ( "#### 5. Taux d'Insuline (mu U/ml)" )
            st.write ( """
                C'est le niveau d'insuline dans le sang, mesuré en unités de micro-unité par millilitre (mu U/ml).
                Des niveaux d'insuline anormalement élevés ou bas peuvent indiquer des problèmes 
                avec le métabolisme du glucose, et donc un risque accru de diabète.
            """ )
            st.write ( "#### 6. Indice de Masse Corporelle (IMC)" )
            st.write ( """
                L'IMC est un indice calculé à partir du poids et de la taille d'une personne, 
                généralement exprimé en kg/m².
                Un IMC élevé est un indicateur de l'obésité, qui est un facteur de risque majeur pour le diabète de type 2.
            """ )
            st.write ( "#### 7. Fonction de Généalogie du Diabète" )
            st.write ( """
                Cela fait référence à l'historique familial de diabète, qui peut être évalué par un score ou un indicateur.
                Les antécédents familiaux de diabète augmentent le risque d'un individu de développer la maladie, 
                car des facteurs génétiques peuvent jouer un rôle.
            """ )
            st.write ( "#### 8. Âge" )
            st.write ( """
                L'âge de l'individu, généralement mesuré en années.
                Le risque de diabète augmente avec l'âge, en particulier après 45 ans. 
                Cela est souvent dû à des changements métaboliques et à une diminution de l'activité physique.
            """ )

        # Page de Prédiction des Maladies Cardiaques

            st.markdown ( "<h2>Prédiction des Maladies Cardiaques </h3>",
                          unsafe_allow_html=True )

            # Informations sur les paramètres
            st.write ( "### Explications des Valeurs :" )
            st.write ( "#### 1. Âge" )
            st.write ( """
                Âge de l'individu, exprimé en années. 
                L'âge est un facteur de risque important pour les maladies cardiaques, 
                car le risque augmente généralement avec l'âge.
            """ )
            st.write ( "#### 2. Sexe (0 = Femme, 1 = Homme)" )
            st.write ( """
                Indique le sexe de l'individu. 
                Le sexe peut influencer le risque de maladies cardiaques, 
                les hommes ayant tendance à être plus à risque à un âge plus jeune que les femmes.
            """ )
            st.write ( "#### 3. Type de Douleur Thoracique" )
            st.write ( """
                Cela classifie le type de douleur thoracique :
                - 0 : Asymptomatique
                - 1 : Douleur typique
                - 2 : Douleur atypique
                - 3 : Non angineuse
                Une douleur thoracique atypique peut parfois être associée à des problèmes cardiaques.
            """ )
            st.write ( "#### 4. Pression Artérielle au Repos (mmHg)" )
            st.write ( """
                Mesure de la pression sanguine au repos, exprimée en millimètres de mercure (mmHg).
                Une pression artérielle élevée est souvent un indicateur de maladies cardiovasculaires.
            """ )
            st.write ( "#### 5. Cholestérol (mg/dl)" )
            st.write ( """
                Taux de cholestérol total dans le sang, mesuré en milligrammes par décilitre (mg/dl).
                Des niveaux élevés de cholestérol sont liés à un risque accru de maladies cardiaques.
            """ )
            st.write ( "#### 6. Sucre à Jeun > 120 mg/dl (0 = Non, 1 = Oui)" )
            st.write ( """
                Indique si le niveau de glucose à jeun est supérieur à 120 mg/dl.
                Un taux élevé de glucose est associé à un risque accru de diabète et de maladies cardiaques.
            """ )
            st.write ( "#### 7. Résultats Électrocardiographiques" )
            st.write ( """
                Mesure de l'activité électrique du cœur :
                - 0 : Normal
                - 1 : Anomalie onde ST
                - 2 : Hypertrophie ventriculaire
                Les anomalies dans ces résultats peuvent indiquer des problèmes cardiaques.
            """ )
            st.write ( "#### 8. Fréquence Cardiaque Max" )
            st.write ( """
                C'est la fréquence cardiaque maximale atteinte par l'individu.
                Une fréquence cardiaque élevée peut être un indicateur de stress ou d'effort physique,
                mais des niveaux anormaux au repos peuvent indiquer des problèmes cardiaques.
            """ )
            st.write ( "#### 9. Angine Induite par Exercice (0 = Non, 1 = Oui)" )
            st.write ( """
                Indique si l'individu a des douleurs thoraciques provoquées par l'exercice.
                La présence d'angine peut signaler un risque accru de maladie cardiaque.
            """ )
            st.write ( "#### 10. Dépression ST Induite par l'Exercice" )
            st.write ( """
                Mesure de la dépression du segment ST pendant un test d'effort. 
                Une dépression ST peut indiquer une ischémie myocardique (réduction de l'apport sanguin au cœur).
            """ )
            st.write ( "#### 11. Pente du Segment ST" )
            st.write ( """
                Indique la pente du segment ST à l'effort :
                - 0 : Pente ascendante
                - 1 : Plate
                - 2 : Descendante
                La pente du segment ST peut fournir des indices sur la santé cardiaque.
            """ )
            st.write ( "#### 12. Nombre de Vaisseaux Principaux Colorés par Fluoroscopie (0-3)" )
            st.write ( """
                Représente le nombre de vaisseaux sanguins détectés par fluoroscopie.
                Un nombre plus élevé peut indiquer des obstructions et un risque accru de maladies cardiaques.
            """ )
            st.write ( "#### 13. Thalassémie" )
            st.write ( """
                Indique le type de thalassémie :
                - 1 : Normal
                - 2 : Défault fixe
                - 3 : Réversible
                La thalassémie peut affecter la santé cardiaque, en particulier en cas de carences en oxygène.
            """ )

        # Page de Prédiction de Parkinson

            st.markdown ( "<h2>Prédiction de Parkinson </h2>", unsafe_allow_html=True )

            # Informations sur les paramètres
            st.write ( "### Explications des Valeurs :" )
            st.write ( "#### 1. ID de la Patient" )
            st.write ( """
                    Un identifiant unique pour chaque patient dans la base de données. 
                    Cela permet de suivre les résultats et les historiques médicaux de chaque individu.
                """ )
            st.write ( "#### 2. Âge" )
            st.write ( """
                    L'âge de l'individu, généralement mesuré en années. 
                    Le risque de développer la maladie de Parkinson augmente généralement avec l'âge.
                """ )
            st.write ( "#### 3. Sexe (0 = Femme, 1 = Homme)" )
            st.write ( """
                    Indique le sexe de l'individu. 
                    Bien que la maladie de Parkinson puisse affecter tout le monde, des études ont montré des différences 
                    dans la prévalence selon le sexe.
                """ )
            st.write ( "#### 4. Années de Formation" )
            st.write ( """
                    Cela représente le nombre d'années d'éducation formelle que l'individu a reçues. 
                    Certaines recherches suggèrent qu'un niveau d'éducation plus élevé peut être associé à un risque réduit 
                    de maladie de Parkinson.
                """ )
            st.write ( "#### 5. Score UPDRS" )
            st.write ( """
                    Le score de l'échelle unifiée de la maladie de Parkinson (UPDRS) évalue la gravité des symptômes moteurs 
                    et non moteurs de la maladie. Un score plus élevé indique une gravité accrue des symptômes.
                """ )
            st.write ( "#### 6. Évaluation de la Motricité" )
            st.write ( """
                    Mesure des fonctions motrices, y compris la coordination et la vitesse des mouvements. 
                    Les difficultés motrices sont un symptôme majeur de la maladie de Parkinson.
                """ )
            st.write ( "#### 7. Évaluation de l'Humeur" )
            st.write ( """
                    Ce paramètre évalue l'état émotionnel et l'humeur générale de l'individu. 
                    Les troubles de l'humeur sont fréquents chez les patients atteints de Parkinson.
                """ )
            st.write ( "#### 8. Évaluation Cognitive" )
            st.write ( """
                    Évalue les fonctions cognitives de l'individu, y compris la mémoire, l'attention et le raisonnement. 
                    La maladie de Parkinson peut entraîner des problèmes cognitifs à mesure qu'elle progresse.
                """ )
            # Page de Prédiction d'Asthme

            st.markdown ( "<h2>Prédiction d'Asthme</h2>", unsafe_allow_html=True )

            # Informations sur les paramètres
            st.write ( "### Explications des Valeurs :" )
            st.write ( "#### 1. Âge" )
            st.write ( """
                    L'âge de l'individu, mesuré en années. 
                    L'incidence de l'asthme peut varier en fonction de l'âge, avec des pics d'incidence observés chez les enfants et les jeunes adultes.
                """ )

            st.write ( "#### 2. Genre (0 = Homme, 1 = Femme)" )
            st.write ( """
                    Indique le genre de l'individu. 
                    Certaines études montrent des différences dans la prévalence de l'asthme selon le genre.
                """ )

            st.write ( "#### 3. Ethnicité (0, 1, 2)" )
            st.write ( """
                    L'ethnicité peut influencer la susceptibilité à l'asthme et la gravité des symptômes. 
                    Les facteurs environnementaux et génétiques peuvent jouer un rôle dans cette variation.
                """ )

            st.write ( "#### 4. Niveau d'Éducation" )
            st.write ( """
                    Cela représente le niveau d'éducation formelle atteint par l'individu. 
                    Des études suggèrent qu'un niveau d'éducation plus élevé est associé à une meilleure compréhension et gestion des symptômes.
                """ )

            st.write ( "#### 5. Indice de Masse Corporelle (IMC)" )
            st.write ( """
                    L'IMC est un indicateur de la corpulence d'un individu. 
                    Un IMC élevé peut être associé à une augmentation de la gravité des symptômes de l'asthme.
                """ )

            st.write ( "#### 6. Exposition à la Pollution" )
            st.write ( """
                    Mesure l'exposition de l'individu à des polluants environnementaux. 
                    Une exposition accrue à la pollution de l'air peut exacerber les symptômes de l'asthme.
                """ )

            st.write ( "#### 7. Antécédents Familiaux d'Asthme" )
            st.write ( """
                    Indique si l'individu a des membres de la famille ayant des antécédents d'asthme. 
                    Les antécédents familiaux peuvent augmenter le risque de développer la maladie.
                """ )

            st.write ( "#### 8. Qualité du Sommeil" )
            st.write ( """
                    Évalue la qualité du sommeil de l'individu. 
                    Un sommeil de mauvaise qualité peut avoir un impact négatif sur la gestion de l'asthme et la santé globale.
                """ )

            st.write ( "#### 9. Symptômes Nocturnes" )
            st.write ( """
                    Indique la fréquence des symptômes d'asthme survenant pendant la nuit. 
                    Des symptômes nocturnes fréquents peuvent indiquer un asthme mal contrôlé.
                """ )

            st.write ( "#### 10. Activité Physique" )
            st.write ( """
                    Mesure le niveau d'activité physique de l'individu, en heures par semaine. 
                    Une activité physique régulière est généralement bénéfique pour la gestion de l'asthme, bien qu'elle puisse parfois déclencher des symptômes.
                """ )

# Ajouter un message de conclusion ou de contact
st.markdown("""
    ---
    ### Merci d'utiliser notre système de prédiction. 
    Si vous avez des questions, n'hésitez pas à nous contacter.

     Contact Facebook : [Victor Sylvano](https://www.facebook.com/Victorsylvano.sylvano)  
     Email : [victorsylvano31@gmail.com](mailto:victorsylvano31@gmail.com)
    
""")
st.markdown("""
    <style>
    .centered-text {
        text-align: center;
        font-size: 24px;  /* Taille de la police */
        font-weight: bold; /* Met le texte en gras */
        color: #4caf50;    /* Couleur du texte (modifiez selon le thème) */
        margin-top: 50px; /* Espace au-dessus du texte */
        margin-bottom: 20px; /* Espace en dessous du texte */
        font-family: 'Arial', sans-serif; /* Choix de la police */
    }
    </style>
""", unsafe_allow_html=True)

# Afficher le texte avec le style défini
st.markdown('<div class="centered-text">" L\'intelligence artificielle au service de la santé "</div>', unsafe_allow_html=True)