import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import logging
logging.basicConfig(level=logging.INFO)

logging.info("Loading trained model...")
filename = "Controllers/finalized_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
logging.info("trained model loaded")

logging.info("Loading input data...")
df = pd.read_csv('Controllers/test.csv')
df = df.drop(columns=['nic'])
X = df.loc[:, df.columns != 'churn']
logging.info("input data loaded")

logging.info("Drop and fill")
for col in df.columns:
    df.dropna(thresh=10, how='all', inplace=True)
    df.fillna(df.median(), inplace=True)

logging.info("Convert all to int")
for col_name in df.columns:
    if df[col_name].dtype == np.number:
        df[col_name] = df[col_name].astype(int)
        continue
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    df[col_name] = df[col_name].astype(int)

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df.values), columns=X.columns)
logging.info("Precict churn")
result = loaded_model.predict(df)
print(result)
