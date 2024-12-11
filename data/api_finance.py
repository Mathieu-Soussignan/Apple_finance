import yfinance as yf

# Télécharger les données d'Apple pour une période spécifique
data = yf.download("AAPL", start="2018-01-01", end="2023-12-31")

# Enregistrer les données dans un fichier CSV
data.to_csv("apple_stock_data.csv")

# Afficher un message de confirmation
print("Les données ont été enregistrées dans 'apple_stock_data.csv'")