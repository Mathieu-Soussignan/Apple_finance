import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Charger les données nettoyées
def load_cleaned_data(filepath='../data/cleaned/cleaned_apple_stock_data.csv'):
    """Charge les données nettoyées et prépare pour Prophet."""
    data = pd.read_csv(filepath, parse_dates=['Date'])
    data = data.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    return data

# Ajuster Prophet avec saisonnalité personnalisée
def train_prophet_with_custom_seasonality(data):
    """Entraîne un modèle Prophet avec saisonnalité personnalisée."""
    model = Prophet()
    # Ajouter une saisonnalité mensuelle personnalisée
    model.add_seasonality(name='mensuelle', period=30.5, fourier_order=5)
    model.fit(data)
    return model

# Faire des prévisions
def forecast_prophet(model, periods):
    """Prédire les valeurs futures avec Prophet."""
    future = model.make_future_dataframe(periods=periods)  # Ajoute des jours pour la prévision
    forecast = model.predict(future)
    return forecast

# Visualiser les résultats
def visualize_forecast(data, forecast, model_name="Prophet avec saisonnalité personnalisée"):
    """Affiche les données historiques et les prévisions."""
    model.plot(forecast)
    plt.title(f"Prévisions avec {model_name}")
    plt.show()

# Calculer les métriques
def calculate_metrics(true_values, predictions, model_name="Prophet avec saisonnalité personnalisée"):
    """Calcule MAE et RMSE pour évaluer les performances."""
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    # Charger les données nettoyées
    cleaned_data = load_cleaned_data()

    # Diviser en données d'entraînement et de test
    train_data = cleaned_data.iloc[:-90]  # Tout sauf les 90 derniers jours
    test_data = cleaned_data.iloc[-90:]  # Les 90 derniers jours pour le test

    # Ajuster le modèle Prophet avec saisonnalité personnalisée
    print("Entraînement du modèle Prophet avec saisonnalité personnalisée...")
    model = train_prophet_with_custom_seasonality(train_data)

    # Faire des prévisions sur 90 jours
    print("Prévision sur 90 jours...")
    forecast = forecast_prophet(model, periods=90)

    # Visualiser les prévisions
    visualize_forecast(cleaned_data, forecast)

    # Évaluer les prévisions sur les 90 derniers jours
    true_values = test_data['y'].values  # Valeurs réelles
    predicted_values = forecast['yhat'][-90:].values  # Derniers 90 jours prévus
    calculate_metrics(true_values, predicted_values)