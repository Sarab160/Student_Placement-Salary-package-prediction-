import pandas as pd

df=pd.read_csv("student.csv")

print(df.head(5))
print(df.columns)
print(df.info())
print(df.describe())