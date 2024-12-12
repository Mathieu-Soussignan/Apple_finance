import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

# Charger les données nettoyées
def load_cleaned_data():
    """
    Charge les données nettoyées depuis un fichier CSV.
    Renomme les colonnes pour les rendre compatibles avec Prophet ('ds' pour les dates, 'y' pour les valeurs).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Chemin du fichier courant
    filepath = os.path.join(base_dir, '../../data/cleaned/cleaned_apple_stock_data.csv')  # Chemin du fichier de données
    data = pd.read_csv(filepath, parse_dates=['Date'])  # Charge les données en parsant la colonne 'Date'
    data = data.rename(columns={'Date': 'ds', 'Adj Close': 'y'})  # Renomme les colonnes pour Prophet
    return data

# Ajuster Prophet
def train_prophet_model(data):
    """
    Entraîne un modèle Prophet sur les données fournies.
    """
    model = Prophet()  # Initialise un modèle Prophet
    model.fit(data)  # Entraîne le modèle sur les données
    return model

# Faire des prévisions
def forecast_prophet(model, periods):
    """
    Prédit les valeurs futures sur un nombre de jours spécifié.
    """
    future = model.make_future_dataframe(periods=periods)  # Crée un DataFrame avec des dates futures
    forecast = model.predict(future)  # Génère les prévisions
    return forecast

# Calculer les métriques
def calculate_metrics(true_values, predictions):
    """
    Calcule les métriques de performance (MAE et RMSE) pour évaluer la précision des prévisions.
    """
    mae = mean_absolute_error(true_values, predictions)  # Erreur absolue moyenne
    rmse = np.sqrt(mean_squared_error(true_values, predictions))  # Erreur quadratique moyenne
    return mae, rmse

# Interface Streamlit
# Configuration de la page
st.set_page_config(page_title="Prédictions des Actions Apple avec Prophet", layout="wide")

# Titre principal
st.title("Prédictions des Actions Apple avec Prophet")

# Section des options (barre latérale)
st.sidebar.header("Options")
forecast_period = st.sidebar.slider("Horizon de prévision (jours)", 30, 180, step=30)  # Slider pour choisir la période de prévision

# Charger les données
cleaned_data = load_cleaned_data()  # Charge les données nettoyées

# Diviser les données en entraînement et test
train_data = cleaned_data.iloc[:-forecast_period]  # Données d'entraînement (toutes sauf les derniers jours)
test_data = cleaned_data.iloc[-forecast_period:]  # Données de test (les derniers jours pour validation)

# Bouton pour générer les prévisions
if st.button("Générer Prévisions"):  # Affiche un bouton pour lancer les prévisions
    # Entraîner le modèle Prophet
    st.write("Entraînement du modèle...")
    model = train_prophet_model(train_data)  # Entraîne le modèle Prophet avec les données d'entraînement

    # Faire les prévisions
    st.write("Calcul des prévisions...")
    forecast = forecast_prophet(model, periods=forecast_period)  # Génère les prévisions

    # Visualiser les prévisions
    st.subheader("Graphique des Prévisions")
    fig = model.plot(forecast)  # Génère le graphique des prévisions
    fig.set_size_inches(10, 5)  # Ajuste la taille du graphique
    st.pyplot(fig)  # Affiche le graphique dans Streamlit

    # Calculer les métriques
    true_values = test_data['y'].values  # Valeurs réelles des données de test
    predicted_values = forecast['yhat'][-forecast_period:].values  # Valeurs prédites pour la période de test
    mae, rmse = calculate_metrics(true_values, predicted_values)  # Calcule les métriques de performance

    # Ajouter une description des résultats
    st.subheader("Résumé des Résultats")
    st.markdown(f"""
    - **Horizon de prévision :** {forecast_period} jours
    - **MAE (Erreur absolue moyenne) :** {mae:.2f}
    - **RMSE (Erreur quadratique moyenne) :** {rmse:.2f}

    Les prévisions montrent une tendance générale qui suit le comportement historique des actions Apple. La bande bleue représente l'incertitude des prévisions, avec une couverture de 95 %. 
    Un MAE de {mae:.2f} signifie que, en moyenne, la différence entre les prévisions et les valeurs réelles est de {mae:.2f} $. Le RMSE de {rmse:.2f} indique que la majorité des erreurs de prévision se situent autour de cette valeur.
    """)

else:
    # Message par défaut si le bouton n'a pas été cliqué
    st.write("Cliquez sur le bouton **Générer Prévisions** pour voir les prévisions.")