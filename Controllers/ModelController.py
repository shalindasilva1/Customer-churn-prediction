import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import logging
logging.basicConfig(level=logging.INFO)

#############################################################
logging.info("Read arguments...")
inputFile = sys.argv[1]
sessionId = sys.argv[2]
#############################################################

#############################################################
logging.info("Loading trained model...")
filename = r"PythonModelHelpFiles/finalized_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
#############################################################

#############################################################
logging.info("Loading input data...")
df = pd.read_csv(inputFile)
tempdf = df
df = df.drop(columns=['nic'])
X = df.loc[:, df.columns != 'churn']
#############################################################

#############################################################
logging.info("Drop and fill")
for col in df.columns:
    df.dropna(thresh=10, how='all', inplace=True)
    tempdf.dropna(thresh=10, how='all', inplace=True)
    df.fillna(df.median(), inplace=True)
#############################################################

#############################################################
logging.info("Convert all to int")
for col_name in df.columns:
    if df[col_name].dtype == np.number:
        df[col_name] = df[col_name].astype(int)
        continue
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    df[col_name] = df[col_name].astype(int)
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df.values), columns=X.columns)
#############################################################

#############################################################
logging.info("Precict churn")
result = loaded_model.predict(df)
#############################################################

#############################################################
logging.info("Add prediction")
tempdf["churn"] = result
#############################################################

directory = sessionId
parent_dir = r"C:\Users\Shalinda\source\repos\shalindasilva1\ML-Project\Web\web-app\src\assets\sessionOutputFiles"
path = os.path.join(parent_dir, directory)
os.mkdir(path)
#############################################################
logging.info("Save to outputfiles")
tempdf.to_csv(path + r"/" +"output.csv")
#############################################################
