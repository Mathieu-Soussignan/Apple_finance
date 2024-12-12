import pandas as pd
from prophet import Prophet

data = pd.read_csv('../data/cleaned/Cleaned_Apple_Stock_Data.csv')


df = data[['Date', 'Close']]
df['Date']= pd.to_datetime(df['Date'])

df.plot(x='Date', y='Close')


df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)  # 30 jours de prévision
forecast = model.predict(future)

