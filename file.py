import pandas as pd

df=pd.read_csv("student.csv")

print(df["salary_package_lpa"].min())
print(df["salary_package_lpa"].max())