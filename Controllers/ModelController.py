import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

filename = "Controllers/finalized_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv('Controllers/test.csv')
df = df.drop(columns=['nic'])
X=df.loc[:, df.columns != 'churn']

for col in df.columns:
    #drop rows having more than half missing values in the dataset (Dont Keep rows with more than 10 missing values.)
    df.dropna(thresh=10,how='all', inplace = True)

    #fill the missing values with the median value of the entier attribute values.
    df.fillna(df.median(), inplace= True)

for col_name in df.columns:
    if df[col_name].dtype==np.number:
        df[col_name] = df[col_name].astype(int)
        continue
    df[col_name] =pd.to_numeric(df[col_name],errors='coerce').fillna(0)
    df[col_name] = df[col_name].astype(int)

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df.values), columns=X.columns)
result = loaded_model.predict(df)
print(result)
