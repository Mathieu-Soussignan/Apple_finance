import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Charger les données nettoyées
def load_cleaned_data(filepath='../data/cleaned/cleaned_apple_stock_data.csv'):
    """Charge les données nettoyées depuis un fichier CSV."""
    data = pd.read_csv(filepath, parse_dates=['Date'])
    return data

# Visualisation des données
def plot_time_series(data):
    """Affiche la série temporelle des prix ajustés."""
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Adj Close'], label='Prix ajusté (Adj Close)')
    plt.title('Prix ajusté des actions Apple')
    plt.xlabel('Date')
    plt.ylabel('Prix ($)')
    plt.legend()
    plt.show()

# Décomposition de la série temporelle
def decompose_time_series(data, model='multiplicative'):
    """Décompose la série temporelle en tendance, saisonnalité et résidus."""
    decomposition = seasonal_decompose(data['Adj Close'], model=model, period=252)
    decomposition.plot()
    plt.show()

# Tester la stationnarité
def test_stationarity(data, column='Adj Close'):
    """Teste la stationnarité de la série avec le test ADF."""
    result = adfuller(data[column].dropna())
    print("Test de stationnarité (ADF):")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] < 0.05:
        print("La série est stationnaire.")
    else:
        print("La série n'est pas stationnaire.")
    return result[1] < 0.05  # Retourne True si stationnaire, False sinon

# Appliquer une différenciation
def apply_differencing(data, column='Adj Close'):
    """Applique une différenciation à la série temporelle."""
    data[f'{column}_diff'] = data[column].diff()
    return data


if __name__ == "__main__":
    # Charger les données nettoyées
    cleaned_data = load_cleaned_data()

    # Étape 1 : Analyse exploratoire
    print("Visualisation des données...")
    plot_time_series(cleaned_data)

    # Décomposition multiplicative de la série temporelle
    print("Décomposition multiplicative des séries temporelles...")
    decompose_time_series(cleaned_data, model='multiplicative')

    # Décomposition additive de la série temporelle
    print("Décomposition additive des séries temporelles...")
    decompose_time_series(cleaned_data, model='additive')

    # Test de stationnarité
    print("Test de stationnarité...")
    is_stationary = test_stationarity(cleaned_data)

    # Si la série n'est pas stationnaire, appliquer une différenciation
    if not is_stationary:
        print("La série n'est pas stationnaire. Application d'une différenciation...")
        cleaned_data = apply_differencing(cleaned_data)

        # Test de stationnarité après différenciation
        print("Test de stationnarité après différenciation...")
        is_stationary = test_stationarity(cleaned_data, column='Adj Close_diff')

        # Visualisation de la série différenciée
        plt.figure(figsize=(12, 6))
        plt.plot(cleaned_data['Date'], cleaned_data['Adj Close_diff'])
        plt.title("Série différenciée")
        plt.xlabel("Date")
        plt.ylabel("Différence du prix ajusté")
        plt.show()