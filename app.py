import streamlit as st
import pandas as pd
from urllib.request import urlopen
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json
import requests
from streamlit_lottie import st_lottie
import pydeck as pdk
import snowflake.connector
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

# Layout
st.set_page_config(
    page_title="PulmoCare",
    layout="wide",
    initial_sidebar_state="expanded")

# Data Pull and Functions
st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


@st.cache_data
def pull_clean():
    master_zip = pd.read_csv('MASTER_ZIP.csv', dtype={'ZCTA5': str})
    master_city = pd.read_csv('MASTER_CITY.csv', dtype={'ZCTA5': str})
    return master_zip, master_city


def clear_image():
    st.session_state.image = None


def resize_image(img, img_size):
    return img.resize(img_size)


@st.cache_data
def classifier_image(image_path):
    # Redimensionner l'image
    img = resize_image(image_path, (224, 224))

    # Convertir l'image en tableau numpy
    img_array = image.img_to_array(img)
    # Ajouter une dimension pour représenter le batch (1 image)
    img_array = np.expand_dims(img_array, axis=0)
    # Normaliser l'image
    img_array = img_array / 255.0

    # Charger le modèle depuis le fichier .h5
    model = load_model("modele.h5")

    # Effectuer la prédiction
    predictions = model.predict(img_array)

    # Interpréter la prédiction
    class_labels = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']
    results = {class_labels[i]: float(predictions[0][i]*100) for i in range(4)}
    # for i, class_label in enumerate(class_labels):
    #     results.append((class_label, predictions[0][i] * 100))

    return results


# Options Menu
with st.sidebar:
    selected = option_menu('PulmoDetect', ["Acceuil", 'Documention', 'A propos'],
                           icons=['play-btn', "bi-file-earmark-bar-graph-fill", 'info-circle'], menu_icon='intersect', default_index=0)


# Intro Page
if selected == "Acceuil":
    # Header

    st.markdown(
        "<h1 style='color:#077485'>Bienvenue sur PulmoDetect</h1>", unsafe_allow_html=True)
    st.markdown(
        "<i><h5 style='color:#077485; font-weight:lighter'>Un avenir plus sain, plus proche. Diagnostique, trie, surveille, éduque, innove, consulte.<br/> Ensemble, pour des poumons plus forts.</h5></i>",
        unsafe_allow_html=True
    )

    st.divider()
    # Use Cases
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "<h3 style='color:#077485'>Charger l'image à analyser ici !</h3>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "", type=["jpg", "jpeg", "png"])
            with st.container():
                subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                with subcol1:
                    sub = st.button("Soumettre")
                with subcol2:
                    if st.button("Néttoyer"):
                        clear_image()
            if uploaded_file is not None:
                # Display the uploaded image
                custom_width = 500  # Modifier cette valeur selon vos besoins
                st.image(uploaded_file, caption='Uploaded Image', width=500)
                st.session_state.image = uploaded_file
        with col2:
            if sub:

                if 'image' in st.session_state:
                    img = st.session_state.image
                    img = image.load_img(uploaded_file)
                    img = img.copy()
                    prediction_result = classifier_image(img)
                    st.markdown(
                        "<h4 style='color:#077485;margin-left:100px;margin-top:20px'>Resultats de l'analyse</h4>", unsafe_allow_html=True)
                    for label, percentage in prediction_result.items():
                        st.write(
                            f"<span style='font-weight:bold; margin-left:100px'>{label}:</span> <span style='color:#1f77b4'>{percentage:.2f}%</span>", unsafe_allow_html=True)
                        # st.write(
                        #     f">  {label} -------------------------------------------------------- {percentage:.2f}%")
                else:
                    st.write("S'il vous plait charger une image .")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "<h1 style='color:#077485'>Cas d'utilisation</h1>", unsafe_allow_html=True)

            st.markdown(
                """
                -  _**Diagnostic précoce** : Identifier les maladies pulmonaires à un stade précoce, permettant un traitement rapide et efficace._

                -  _**Triage rapide** : Aider les professionnels de la santé à trier les patients en fonction de la gravité de leur maladie et de leur besoin de soins immédiats._

                -  _**Suivi des patients** : Surveiller la progression de la maladie chez les patients au fil du temps pour ajuster les traitements et évaluer l'efficacité des interventions._

                -  _**Éducation et sensibilisation** : Sensibiliser le public aux différents types de maladies pulmonaires en montrant visuellement les symptômes et les conséquences sur les images._

                -  _**Recherche médicale : Fournir des données visuelles pour la recherche médicale sur les maladies pulmonaires, telles que l'analyse de l'efficacité des traitements ou l'identification de nouveaux biomarqueurs._

                -  _**Téléconsultation** : Permettre aux médecins de consulter les patients à distance en visualisant les images pulmonaires, améliorant ainsi l'accès aux soins de santé, en particulier dans les zones rurales ou éloignées._
                                """
            )

    st.divider()

    # Tutorial Video
    # st.header('Tutorial Video')
    # video_file = open('Similo_Tutorial3_compressed.mp4', 'rb')
    # video_bytes = video_file.read()
    # st.video(video_bytes)

# Search Page
if selected == "Documention":
    st.write("## Utilisation de l'Application PulmoDetect")

    st.write("### 1. Exécution de l'Application")
    st.write("   Pour utiliser PulmoDetect, assurez-vous d'avoir le script Python contenant l'application Streamlit. Ensuite, exécutez la commande suivante dans votre terminal :")
    st.code("streamlit run app.py")

    st.write("### 2. Téléchargement de l'Image")
    st.write("   Lorsque l'application est lancée, vous serez redirigé vers la page d'accueil de PulmoDetect.")
    st.write(
        "   - Sur la page d'accueil, vous verrez une section permettant de télécharger une image.")
    st.write("   - Cliquez sur le bouton 'Parcourir les fichiers' pour sélectionner l'image que vous souhaitez analyser.")

    st.write("### 3. Analyse de l'Image")
    st.write("   Après avoir téléchargé l'image, vous pouvez cliquer sur le bouton 'Submit' pour lancer l'analyse.")
    st.write("   - Les résultats de l'analyse seront affichés sur la page, indiquant les pourcentages associés à chaque classe de maladie pulmonaire détectée.")

    st.write("### 4. Nettoyage de l'Image")
    st.write("   Si vous souhaitez supprimer l'image téléchargée et recommencer, vous pouvez cliquer sur le bouton 'Clear'. Cela effacera l'image chargée et vous pourrez télécharger une nouvelle image.")

    st.write("## Remarque")
    st.write("Assurez-vous d'avoir le modèle de détection de maladie pulmonaire (.h5) dans le même répertoire que le script Python de l'application.")

# About Page
if selected == 'A propos':
    st.write("## À PROPOS DU MODÈLE")

    st.write("Ce modèle a été conçu pour la classification d'images de tumeurs cérébrales en utilisant le dataset masoudnickparvar/brain-tumor-mri-dataset de Kaggle.")

    st.write("### PERFORMANCE DU MODÈLE")
    st.write(
        "Notre modèle EfficientNetB0 a atteint les performances suivantes lors des tests :")
    st.write("- Précision: 98%")
    st.write("- Rappel moyen: 98%")
    st.write("- Score F1 moyen: 98%")
    st.write("- Nombre total d'échantillons: 1311")
    st.write("- Nombre d'erreurs trouvées: 23")

    st.write("### ARCHITECTURE DU MODÈLE")
    st.write(
        "L'architecture de notre modèle EfficientNetB0 est configurée comme suit :")
    st.write("- Paramètres totaux: 4,418,375 (16.85 MB)")
    st.write("- Paramètres entraînables: 3,688,004 (1.41 MB)")
    st.write("- Paramètres non-entraînables: 4,045,971 (15.45 MB)")
    st.write("Le modèle comprend des couches convolutives personnalisées en plus des couches de pooling et une couche dense finale pour la classification en 4 classes.")

    st.write("### DATASET UTILISÉ")
    st.write("Les ensembles de données utilisés pour l'entraînement et la validation sont les suivants :")
    st.write("- Fichiers d'entraînement: 5712 fichiers, répartis en 4 classes.")
    st.write("- Fichiers de validation: 1311 fichiers, répartis en 4 classes.")
    st.write("Note: Les données ont été préparées et réparties pour assurer une répartition équilibrée, ce qui est crucial pour l'entraînement d'un modèle de classification robuste.")
    st.divider()
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image("loss.png", caption="Courbe de Loss",
                     use_column_width=True)

        with col2:
            st.image("accuracy.png", caption="Courbe d'Accuracy",
                     use_column_width=True)
