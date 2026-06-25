import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

df = pd.read_csv(
    "adult.csv",
    header=None,
    names=columns,
    skipinitialspace=True,
    na_values="?"
)

df.to_csv("adult_clean.csv", index=False)

print(df.head())
print(df.isnull().sum())