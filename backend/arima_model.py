import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet

# Charger les données nettoyées
def load_cleaned_data(filepath='../data/cleaned/cleaned_apple_stock_data.csv'):
    """Charge les données nettoyées depuis un fichier CSV."""
    data = pd.read_csv(filepath, parse_dates=['Date'])
    return data

# Appliquer une différenciation
def apply_differencing(data, column='Adj Close'):
    """Applique une différenciation à la série temporelle."""
    diff_column = f'{column}_diff'
    if diff_column not in data.columns:
        data[diff_column] = data[column].diff()
    return data

# Tracer ACF et PACF
def plot_acf_pacf(data, column='Adj Close_diff'):
    """Affiche les graphiques ACF et PACF pour déterminer les paramètres ARIMA."""
    plt.figure(figsize=(12, 6))
    plot_acf(data[column].dropna(), lags=40)
    plt.title("Autocorrélation (ACF)")
    plt.show()

    plt.figure(figsize=(12, 6))
    plot_pacf(data[column].dropna(), lags=40)
    plt.title("Autocorrélation partielle (PACF)")
    plt.show()

# Implémenter et entraîner ARIMA
def fit_arima_model(data, order):
    """Entraîne un modèle ARIMA avec les paramètres spécifiés."""
    model = ARIMA(data['Adj Close'], order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

# Implémenter et entraîner SARIMA
def fit_sarima_model(data, order, seasonal_order):
    """Entraîne un modèle SARIMA avec les paramètres spécifiés."""
    model = SARIMAX(data['Adj Close'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

# Prévoir les valeurs futures
def forecast_model(model_fit, steps):
    """Prédit les valeurs futures à partir d'un modèle ajusté."""
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Implémenter Prophet
def fit_prophet_model(data):
    """Entraîne un modèle Prophet et génère des prévisions."""
    df = data.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    model = Prophet()
    model.fit(df)
    forecast = model.predict(model.make_future_dataframe(periods=30))
    model.plot(forecast)
    plt.show()
    return forecast

if __name__ == "__main__":
    # Charger les données nettoyées
    cleaned_data = load_cleaned_data()

    # Appliquer une différenciation pour rendre la série stationnaire
    cleaned_data = apply_differencing(cleaned_data)

    # Vérifier si la colonne différenciée existe
    if 'Adj Close_diff' not in cleaned_data.columns or cleaned_data['Adj Close_diff'].isnull().all():
        raise ValueError("La série différenciée ('Adj Close_diff') est vide ou n'a pas été créée correctement.")

    # Tracer les graphiques ACF et PACF
    print("Tracer les graphiques ACF et PACF...")
    plot_acf_pacf(cleaned_data)

    # Tester plusieurs modèles ARIMA
    print("Tester ARIMA(1,1,1)...")
    arima_model_1 = fit_arima_model(cleaned_data, order=(1, 1, 1))
    print("Tester ARIMA(2,1,2)...")
    arima_model_2 = fit_arima_model(cleaned_data, order=(2, 1, 2))

    # Prévisions avec le meilleur modèle ARIMA
    print("Prévisions avec ARIMA(2,1,2)...")
    forecast_arima = forecast_model(arima_model_2, steps=30)
    print("Prévisions sur 30 jours (ARIMA) :")
    print(forecast_arima)

    # Tester SARIMA
    print("Tester SARIMA(1,1,1)(1,1,1,12)...")
    sarima_model = fit_sarima_model(cleaned_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

    # Prévisions avec SARIMA
    print("Prévisions avec SARIMA...")
    forecast_sarima = forecast_model(sarima_model, steps=30)
    print("Prévisions sur 30 jours (SARIMA) :")
    print(forecast_sarima)

    # Tester Prophet
    print("Tester Prophet...")
    forecast_prophet = fit_prophet_model(cleaned_data)

    # Comparer les prévisions
    plt.figure(figsize=(12, 6))
    plt.plot(cleaned_data['Date'], cleaned_data['Adj Close'], label='Historique')
    plt.plot(pd.date_range(cleaned_data['Date'].iloc[-1], periods=30, freq='B'), forecast_arima, label='ARIMA', color='orange')
    plt.plot(pd.date_range(cleaned_data['Date'].iloc[-1], periods=30, freq='B'), forecast_sarima, label='SARIMA', color='green')
    plt.title("Comparaison des prévisions")
    plt.legend()
    plt.show()