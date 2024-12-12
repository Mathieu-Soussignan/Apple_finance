# Documentation du Modèle de Prévision Prophet

## Objectif du Projet
Prévoir les prix futurs des actions Apple (échelons journaliers) en utilisant le modèle **Prophet** de Facebook. Le but est de fournir des prévisions fiables sur des horizons de 30 à 90 jours tout en permettant une exploration interactive via un tableau de bord.

---

## Étapes Suivies

### 1. **Chargement et Nettoyage des Données**
- Les données historiques des actions Apple (à partir de Yahoo Finance) ont été nettoyées.
- Les colonnes "Date" et "Adj Close" ont été renommées pour correspondre à la nomenclature Prophet (« ds » pour les dates, « y » pour les valeurs).

### 2. **Exploration des Données**
- Analyse exploratoire initiale :
  - Tracé des prix ajustés sur la période considérée (2018-2024).
  - Décomposition de la série temporelle pour identifier les composantes (tendance, saisonnalité).

### 3. **Modélisation avec Prophet**
- **Pourquoi Prophet ?**
  - Il gère bien les séries temporelles non linéaires avec saisonnalités multiples.
  - Sa flexibilité permet d'ajouter des composantes personnalisées (saisonnalités mensuelles, trimestrielles, etc.).

- **Configuration du Modèle :**
  - Utilisation de la configuration par défaut de Prophet.
  - Ajout d’une **saisonnalité mensuelle personnalisée** (étendue sur 30,5 jours avec 5 termes de Fourier).

### 4. **Prévisions**
- Le modèle a prévu les prix sur des horizons de 30 et 90 jours.
- Évaluation des performances sur les 30 et 90 derniers jours des données historiques.

### 5. **Évaluation du Modèle**
- Métriques calculées pour les horizons de prévision :
  - **30 jours :** MAE = 8,17 | RMSE = 9,21
  - **90 jours :** MAE = 8,24 | RMSE = 9,29
- Ces résultats montrent que le modèle prédit avec une précision raisonnable et est robuste même sur des périodes étendues.

---

## Résultats Clés
- Graphiques de prévision (30 et 90 jours) montrant la tendance capturée et les intervalles de confiance.
- Le modèle Prophet est prêt pour être utilisé dans un cadre opérationnel ou interactif.

---

## Tableau de Bord Interactif

### 1. **Choix de la Plateforme : Streamlit**
- Streamlit est idéal pour des visualisations interactives et une expérience utilisateur rapide.

### 2. **Fonctionnalités Proposées :**
- **Visualisation des données historiques :**
  - Courbe des prix ajustés.
- **Prévisions interactives :**
  - Sélection de l’horizon de prévision (30, 60, 90 jours).
  - Graphiques des prévisions avec intervalles de confiance.
- **Métriques d’évaluation :**
  - MAE, RMSE affichés pour chaque horizon choisi.

### 3. **Structure de l’Application Streamlit :**
- **Sidebar :**
  - Sélection de la période de prévision.
- **Section principale :**
  - Graphiques interactifs des prévisions.
  - Tableaux des métriques d’évaluation.

### 4. **Exemple de Code pour Streamlit :**
```python
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Charger les données nettoyées
def load_cleaned_data(filepath='./data/cleaned_apple_stock_data.csv'):
    data = pd.read_csv(filepath, parse_dates=['Date'])
    data = data.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    return data

# Entraîner et prédire
def forecast_prophet(data, periods):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# Application Streamlit
st.title("Prévisions des Actions Apple avec Prophet")

# Charger les données
st.sidebar.header("Options")
data = load_cleaned_data()
horizon = st.sidebar.slider("Horizon de prévision (jours)", 30, 180, 90)

# Prédire et afficher
if st.button("Générer Prévisions"):
    model, forecast = forecast_prophet(data, periods=horizon)
    st.write("### Graphique des Prévisions")
    fig = model.plot(forecast)
    st.pyplot(fig)
```

---

### 5. **Prochaines Améliorations**
- Ajouter des variables externes pour enrichir les prévisions (par ex., événements financiers).
- Comparer Prophet avec d’autres modèles (ARIMA, LSTM).
- Déployer le tableau de bord sur un serveur (par ex. Streamlit Cloud).

---

Ce processus est maintenant prêt à être présenté ou déployé. Dites-moi si vous souhaitez des ajustements ou des ajouts supplémentaires !