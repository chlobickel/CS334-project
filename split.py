from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.impute import SimpleImputer


df = pd.read_csv("processed_data.csv")
x = df.drop(columns=['Numeric Severity']) 
y = df['Numeric Severity']

imputer = SimpleImputer(strategy='mean')
ximputed = imputer.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(ximputed, y, test_size=0.3, random_state=42)

xtrain = pd.DataFrame(xtrain, columns=x.columns)
xtest = pd.DataFrame(xtest, columns=x.columns)

xtrain.to_csv("xtrain.csv", index=False) 
xtest.to_csv("xtest.csv", index=False) 
ytrain.to_csv("ytrain.csv", index=False) 
ytest.to_csv("ytest.csv", index=False) 
